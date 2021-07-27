use std::sync::{Mutex, MutexGuard, Arc};

use anyhow::{format_err, Result};
use hwloc2::{Bitmap, ObjectType, Topology, TopologyObject, CpuBindFlags, CpuSet};
use lazy_static::lazy_static;
use log::{debug, info, warn};
use storage_proofs_core::settings::SETTINGS;
use super::utils::{env_lock_p2_cores, p2_binding_policy, binding_use_locality, P2BoundPolicy};

pub type CoreGroup = Vec<CoreIndex>;

lazy_static! {
    pub static ref TOPOLOGY: Mutex<Topology> = Mutex::new(Topology::new().unwrap());
    pub static ref CORE_GROUPS: Option<Vec<Mutex<CoreGroup>>> = {
        let num_producers = &SETTINGS.multicore_sdr_producers;
        let cores_per_unit = num_producers + 1;

        core_groups(cores_per_unit)
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// `CoreIndex` is a simple wrapper type for indexes into the set of vixible cores. A `CoreIndex` should only ever be
/// created with a value known to be less than the number of visible cores.
pub struct CoreIndex(pub usize);

pub fn checkout_core_group() -> Option<MutexGuard<'static, CoreGroup>> {
    match &*CORE_GROUPS {
        Some(groups) => {
            for (i, group) in groups.iter().enumerate() {
                match group.try_lock() {
                    Ok(guard) => {
                        debug!("checked out core group {}", i);
                        return Some(guard);
                    }
                    Err(_) => debug!("core group {} locked, could not checkout", i),
                }
            }
            None
        }
        None => None,
    }
}

pub fn get_p1_core_group() -> Option<Vec<MutexGuard<'static, CoreGroup>>> {
    match &*CORE_GROUPS {
        Some(groups) => {
            let total_size = &SETTINGS.multicore_sdr_producers + 1;
            let mut current_size: usize = 0;
            let mut res = vec![];
            for (i, group) in groups.iter().enumerate() {
                match group.try_lock() {
                    Ok(guard) => {
                        let n = guard.len();
                        res.push(guard);
                        current_size += n;
                        if current_size >= total_size {
                            return Some(res);
                        }
                    }
                    Err(_) => debug!("core group {} locked, could not checkout", i),
                }
            }
            if res.len() > 0 {
                info!("not enough free cores, P1 uses only {}", current_size);
                return Some(res);
            }
            None
        }
        None => None,
    }
}

pub fn get_p2_core_group() -> Option<Vec<MutexGuard<'static, CoreGroup>>> {
    match &*CORE_GROUPS {
        Some(groups) => {
            let binding_policy = p2_binding_policy();
            if binding_policy == P2BoundPolicy::NoBinding {
                return None;
            }

            let total_size = env_lock_p2_cores();
            let mut current_size: usize = 0;
            let mut res = vec![];
            for (i, group) in groups.iter().enumerate() {
                match group.try_lock() {
                    Ok(guard) => {
                        let n = guard.len();
                        res.push(guard);
                        current_size += n;
                        if current_size >= total_size {
                            return Some(res);
                        }
                    }
                    Err(_) => debug!("core group {} locked, could not checkout", i),
                }
            }
            if res.len() < total_size && binding_policy == P2BoundPolicy::Strict {
                info!("not enough free cores, Strict bound policy implies not use binding");
                return None;
            }
            if res.len() > 0 {
                info!("not enough free cores, Weak bound policy, P2 uses only {}", current_size);
                return Some(res);
            }
            None
        }
        None => None,
    }
}

#[cfg(not(target_os = "windows"))]
pub type ThreadId = libc::pthread_t;

#[cfg(target_os = "windows")]
pub type ThreadId = winapi::winnt::HANDLE;

/// Helper method to get the thread id through libc, with current rust stable (1.5.0) its not
/// possible otherwise I think.
#[cfg(not(target_os = "windows"))]
fn get_thread_id() -> ThreadId {
    unsafe { libc::pthread_self() }
}

#[cfg(target_os = "windows")]
fn get_thread_id() -> ThreadId {
    unsafe { kernel32::GetCurrentThread() }
}

pub struct Cleanup {
    tid: ThreadId,
    prior_state: Option<Bitmap>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(prior) = self.prior_state.take() {
            let child_topo = &TOPOLOGY;
            let mut locked_topo = child_topo.lock().expect("poisded lock");
            let _ = locked_topo.set_cpubind_for_thread(self.tid, prior, CpuBindFlags::CPUBIND_THREAD);
        }
    }
}

pub fn bind_core(core_index: CoreIndex) -> Result<Cleanup> {
    let child_topo = &TOPOLOGY;
    let tid = get_thread_id();
    let mut locked_topo = child_topo.lock().expect("poisoned lock");
    let core = get_core_by_index(&locked_topo, core_index)
        .map_err(|err| format_err!("failed to get core at index {}: {:?}", core_index.0, err))?;

    let cpuset = core
        .cpuset()
        .ok_or_else(|| format_err!("no allowed cpuset for core at index {}", core_index.0,))?;
    debug!("allowed cpuset: {:?}", cpuset);
    let bind_to = cpuset;

    // Thread binding before explicit set.
    let before = locked_topo.get_cpubind_for_thread(tid, CpuBindFlags::CPUBIND_THREAD);

    debug!("binding to {:?}", bind_to);
    // Set the binding.
    let result = locked_topo
        .set_cpubind_for_thread(tid, bind_to, CpuBindFlags::CPUBIND_THREAD)
        .map_err(|err| format_err!("failed to bind CPU: {:?}", err));

    if result.is_err() {
        warn!("error in bind_core, {:?}", result);
    }

    Ok(Cleanup {
        tid,
        prior_state: before,
    })
}

pub fn bind_core_set(core_set: Arc<Vec<CoreIndex>>) -> Result<Cleanup> {
    let child_topo = &TOPOLOGY;
    let tid = get_thread_id();
    let mut locked_topo = child_topo.lock().expect("poisoned lock");

    let mut cpu_sets = Vec::new();
    for i in 0..core_set.len() {
        let core_index = core_set[i].clone();
        let core = get_core_by_index(&locked_topo, core_index)
            .map_err(|err| format_err!("failed to get core at index {}: {:?}", core_index.0, err))?;
        let cpuset = core
            .cpuset()
            .ok_or_else(|| format_err!("no allowed cpuset for core at index {}", core_index.0,))?;
        cpu_sets.push(cpuset);
    }
    let mut acc_cpuset = CpuSet::new();
    for x in cpu_sets {
        acc_cpuset = CpuSet::or(acc_cpuset, x);
    }
    debug!("allowed cpuset: {:?}", acc_cpuset);
    let bind_to = acc_cpuset;

    // Thread binding before explicit set.
    let before = locked_topo.get_cpubind_for_thread(tid, CpuBindFlags::CPUBIND_THREAD);

    debug!("binding to {:?}", bind_to);
    // Set the binding.
    let result = locked_topo
        .set_cpubind_for_thread(tid, bind_to, CpuBindFlags::CPUBIND_THREAD)
        .map_err(|err| format_err!("failed to bind CPU: {:?}", err));

    if result.is_err() {
        warn!("error in bind_core, {:?}", result);
    }

    Ok(Cleanup {
        tid,
        prior_state: before,
    })
}

fn get_core_by_index(topo: &Topology, index: CoreIndex) -> Result<&TopologyObject> {
    let idx = index.0;

    match topo.objects_with_type(&ObjectType::PU) {
        Ok(all_cores) if idx < all_cores.len() => Ok(all_cores[idx]),
        Ok(all_cores) => Err(format_err!(
            "idx ({}) out of range for {} cores",
            idx,
            all_cores.len()
        )),
        _e => Err(format_err!("failed to get core by index {}", idx,)),
    }
}

fn core_groups(cores_per_unit: usize) -> Option<Vec<Mutex<Vec<CoreIndex>>>> {
    let topo = TOPOLOGY.lock().expect("poisoned lock");

    let core_depth = match topo.depth_or_below_for_type(&ObjectType::Core) {
        Ok(depth) => depth,
        Err(_) => return None,
    };
    let all_cores = topo
        .objects_with_type(&ObjectType::Core)
        .expect("objects_with_type failed");
    let core_count = all_cores.len();

    let all_pu = topo
        .objects_with_type(&ObjectType::PU)
        .expect("objects_with_type failed");
    let pu_count = all_pu.len();

    let pu_per_core = pu_count / core_count;

    let mut cache_depth = core_depth;
    let mut cache_count = 1;

    while cache_depth > 0 {
        let objs = topo.objects_at_depth(cache_depth);
        let obj_count = objs.len();
        if obj_count < core_count {
            cache_count = obj_count;
            break;
        }

        cache_depth -= 1;
    }

    assert_eq!(0, core_count % cache_count);
    let mut group_size = (core_count / cache_count) * pu_per_core;
    let mut group_count = cache_count;

    if !binding_use_locality() {
        group_size = 1;
        group_count = pu_count;
        debug!(
            "Cores: {}, cores are not grouped per cache.",
            pu_count,
        );
    } else {
        if cache_count <= 1 {
            // If there are not more than one shared caches, there is no benefit in trying to group cores by cache.
            // In that case, prefer more groups so we can still bind cores and also get some parallelism.
            // Create as many full groups as possible. The last group may not be full.
            group_count = pu_count / cores_per_unit;
            group_size = cores_per_unit;
    
            info!(
                "found only {} shared cache(s), heuristically grouping cores into {} groups",
                cache_count, group_count
            );
        } else {
            debug!(
                "Cores: {}, Shared Caches: {}, cores per cache (group_size): {}",
                pu_count, cache_count, group_size
            );
        }
    }

    let core_groups = (0..group_count)
        .map(|i| {
            (0..group_size)
                .map(|j| {
                    let core_index = i * group_size + j;
                    assert!(core_index < pu_count);
                    CoreIndex(core_index)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Some(
        core_groups
            .iter()
            .map(|group| Mutex::new(group.clone()))
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cores() {
        core_groups(2);
    }

    #[test]
    #[cfg(feature = "single-threaded")]
    fn test_checkout_cores() {
        let checkout1 = checkout_core_group();
        dbg!(&checkout1);
        let checkout2 = checkout_core_group();
        dbg!(&checkout2);

        // This test might fail if run on a machine with fewer than four cores.
        match (checkout1, checkout2) {
            (Some(c1), Some(c2)) => assert!(*c1 != *c2),
            _ => panic!("failed to get two checkouts"),
        }
    }
}
