use log::*;
use std::sync::{Arc};
use storage_proofs_core::settings::SETTINGS;
use num_cpus;

use super::super::utils::{P2BoundPolicy, p2_binding_policy, env_lock_p2_cores};

const MEMORY_PADDING: f64 = 0.35f64;

pub fn get_memory_padding() -> f64 {
    std::env::var("FIL_PROOFS_GPU_MEMORY_PADDING")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_PROOFS__GPU_MEMORY_PADDING! Defaulting to {}", MEMORY_PADDING);
                Ok(MEMORY_PADDING)
            }
        })
        .unwrap_or(MEMORY_PADDING)
        .max(0f64)
        .min(1f64)
}

pub fn get_gpu_for_parallel_tree_r() -> usize {
    std::env::var("FIL_PROOFS_GPU_FOR_PARALLEL_TREE_R")
                .and_then(|v| match v.parse() {
                    Ok(val) => Ok(val),
                    Err(_) => {
                        error!("Invalid FIL_PROOFS_TREE_PER_GPU! Defaulting to {}", SETTINGS.gpu_for_parallel_tree_r);
                        Ok(SETTINGS.gpu_for_parallel_tree_r)
                    }
                })
                .unwrap_or(SETTINGS.gpu_for_parallel_tree_r) as usize
}

pub fn get_core_pool(core_group: Arc<Vec<usize>>) -> rayon::ThreadPool {
    let pool;
    let binding_policy = p2_binding_policy();
    if binding_policy == P2BoundPolicy::Weak || (binding_policy == P2BoundPolicy::Strict && core_group.len() >= env_lock_p2_cores()) {
        pool = thread_binder::ThreadPoolBuilder::new_with_core_set(core_group.clone()).build().expect("failed creating pool for P2");
    } else {
        let cpus = num_cpus::get();
        pool = rayon::ThreadPoolBuilder::new().num_threads(cpus).build().expect("failed creating pool for P2");
    }
    pool
}
