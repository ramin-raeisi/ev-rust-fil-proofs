use log::*;
use std::sync::{Arc};
use storage_proofs_core::settings::SETTINGS;
use num_cpus;

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
    if core_group.len() > 0 {
        pool = thread_binder::ThreadPoolBuilder::new_with_core_set(core_group.clone()).build().expect("failed creating core pool");
    } else {
        let cpus = num_cpus::get();
        pool = rayon::ThreadPoolBuilder::new().num_threads(cpus).build().expect("failed creating core pool");
    }
    pool
}
