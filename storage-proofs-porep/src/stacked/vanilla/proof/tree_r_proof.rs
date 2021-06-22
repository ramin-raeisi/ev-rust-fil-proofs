use std::fs::OpenOptions;
use std::io::Write;
use std::path::{PathBuf};
use std::sync::{mpsc, Arc, Mutex,
    atomic::{AtomicU64, Ordering::SeqCst}};
use std::thread;
use std::time::Duration;

use anyhow::Context;
use bellperson::bls::Fr;
use filecoin_hashers::{Domain, Hasher, PoseidonArity};
use generic_array::typenum::{self, Unsigned};
use log::*;
use merkletree::merkle::{
    get_merkle_tree_cache_size, get_merkle_tree_leafs,
};
use merkletree::store::{StoreConfig};
use rayon::prelude::*;
use storage_proofs_core::{
    data::Data,
    error::Result,
    merkle::*,
    settings,
    util::{NODE_SIZE},
};

use super::super::{
    params::{
        LabelsCache,
    },
    proof::StackedDrg,
    cores::{bind_core_set, get_p2_core_group, CoreIndex, Cleanup},
    utils::{P2BoundPolicy, p2_binding_policy, p2_binding_use_same_set}
};

use neptune::batch_hasher::BatcherType;
use neptune::tree_builder::{TreeBuilder, TreeBuilderTrait};
use fr32::fr_into_bytes;

use rust_gpu_tools::opencl;

use crate::encode::{encode};

use bellperson::gpu::{scheduler};
use super::utils::{get_memory_padding, get_gpu_for_parallel_tree_r, get_p2_pool};

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> { 
    pub fn generate_tree_r_last_gpu<TreeArity>(
        data: &mut Data<'_>,
        nodes_count: usize,
        tree_count: usize,
        tree_r_last_config: StoreConfig,
        replica_path: PathBuf,
        labels: &LabelsCache<Tree>,
    ) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        TreeArity: PoseidonArity,
    {
        let (configs, replica_config) = split_config_and_replica(
            tree_r_last_config.clone(),
            replica_path,
            nodes_count,
            tree_count,
        )?;

        data.ensure_data()?;
        let last_layer_labels = labels.labels_for_last_layer()?;

        info!("[tree_r_last] generating tree r last using the GPU");
        let max_gpu_tree_batch_size = settings::SETTINGS.max_gpu_tree_batch_size as usize;

        let mut batchertype_gpus = Vec::new();
        let all_bus_ids = opencl::Device::all()
            .iter()
            .map(|d| d.bus_id().unwrap())
            .collect::<Vec<_>>();
        let bus_num = all_bus_ids.len();
        assert!(bus_num > 0);

        let tree_r_gpu = get_gpu_for_parallel_tree_r();
        let mut start_idx = 0;
        if tree_r_gpu > 0 { // tree_r_lats will be calculated in parallel with tree_c using tree_r_gpu GPU
            assert!(tree_r_gpu < bus_num, 
                "tree_r_last are calculating in parallel with tree_c. There is not free GPU for tree_c. Try to decrease gpu_for_parallel_tree_r constant.");
            info!("[tree_r_last] are calculating in paralle with tree_c. It uses {}/{} GPU", tree_r_gpu, bus_num);

            // tree_r_last uses last indexes of the GPU list
            start_idx = bus_num - tree_r_gpu; 
        }

        let mut bus_num = bus_num - start_idx;

        let trees_per_gpu: usize = std::env::var("FIL_PROOFS_TREE_PER_GPU")
                .and_then(|v| match v.parse() {
                    Ok(val) => Ok(val),
                    Err(_) => {
                        error!("Invalid FIL_PROOFS_TREE_PER_GPU! Defaulting to {}", 0);
                        Ok(0)
                    }
                })
                .unwrap_or(0);
            
        if trees_per_gpu != 0 {
            assert!(trees_per_gpu * bus_num >= configs.len(), "wrong FIL_PROOFS_TREE_PER_GPU value");
            bus_num = ((configs.len() as f64) / (trees_per_gpu as f64)).ceil() as usize;
        }

        //let batchers_per_gpu = configs.len() / bus_num + 1;
        for gpu_idx in start_idx..start_idx + bus_num {
            batchertype_gpus.push(BatcherType::CustomGPU
                (opencl::GPUSelector::BusId(all_bus_ids[gpu_idx])));
        }

        // ================= CPU POOL ===============
        let groups = get_p2_core_group();
        let mut core_group: Vec<CoreIndex> = vec![];
        let mut core_group_usize: Vec<usize> = vec![];
        if let Some(groups) = groups {
            for cg in groups {
                for core_id in 0..cg.len() {
                    let core_index = cg.get(core_id);
                    if let Some(core_index) = core_index {
                        core_group.push(core_index.clone());
                        core_group_usize.push(core_index.0)
                    }
                }
            }
        }
        
        let core_group = Arc::new(core_group);
        let core_group_usize = Arc::new(core_group_usize);

        let binding_policy = p2_binding_policy();
        let bind_thread = || -> Option<Result<Cleanup>> 
        {
            if binding_policy != P2BoundPolicy::NoBinding && core_group.len() > 0 {
                return Some(bind_core_set(core_group.clone()));
            }
            None
        };
        // =====
        
        let mut builders_rx_by_gpu = Vec::new();
        let mut builders_tx = Vec::new();
        for _i in 0..bus_num {
            builders_rx_by_gpu.push(Vec::new());
        }

        for config_idx in 0..configs.len() {
            // This channel will receive batches of columns and add them to the ColumnTreeBuilder.
            // Each config has own channel
            let (builder_tx, builder_rx) = mpsc::sync_channel(0);
            builders_tx.push(builder_tx);
            builders_rx_by_gpu[config_idx % bus_num].push(builder_rx);
        }


        let bus_num = batchertype_gpus.len();
        assert!(bus_num > 0);

        let mem_one_thread = 800000000;
        let gpu_memory_padding = get_memory_padding();

        let last_layer_labels = Arc::new(Mutex::new(last_layer_labels));

        let config_count = configs.len(); // Don't move config into closure below.
        let configs = &configs;
        let tree_r_last_config = &tree_r_last_config;
        crossbeam::scope(|s| {
            let mut main_threads = Vec::new();

            // This channel will receive the finished tree data to be written to disk.
            let mut writers_tx = Vec::new();
            let mut writers_rx = Vec::new();
            for _i in 0..configs.len() {
                let (writer_tx, writer_rx) = mpsc::sync_channel::<Vec<Fr>>(0);
                writers_tx.push(writer_tx);
                writers_rx.push(writer_rx);
            }

            let data_raw = data.as_mut();
            
            main_threads.push(s.spawn(move |_| {
                let _cleanup_handle_prepare = bind_thread();
                crossbeam::scope(|s2| {
                    let mut threads = Vec::new();

                    for ((&i, builder_tx), data) in (0..config_count).collect::<Vec<_>>().iter()
                        .zip(builders_tx.into_iter())
                        .zip(data_raw.chunks_mut(nodes_count * NODE_SIZE))
                        {
                            
                        let last_layer_labels = last_layer_labels.clone();
                        let core_group_usize = core_group_usize.clone();
                        threads.push(s2.spawn(move |_| {
                            let _cleanup_handle_prepare_i = bind_thread();
                            let mut node_index = 0;
                            while node_index != nodes_count {
                                let chunked_nodes_count =
                                    std::cmp::min(nodes_count - node_index, max_gpu_tree_batch_size);
                                let start = node_index;
                                let end = start + chunked_nodes_count;
                                trace!(
                                    "processing config {}/{} with leaf nodes {} [{}, {}, {}-{}]",
                                    i + 1,
                                    tree_count,
                                    chunked_nodes_count,
                                    node_index,
                                    nodes_count,
                                    start,
                                    end,
                                );
                                
                                let encoded_data = {
                                    use fr32::bytes_into_fr;

                                    let mut layer_bytes =
                                        vec![0u8; (end - start) * std::mem::size_of::<Fr>()];

                                    {
                                        let last_layer_labels = last_layer_labels.lock().unwrap();
                                        let labels_start = i * nodes_count + node_index;
                                        let labels_end = labels_start + chunked_nodes_count;
                                        last_layer_labels
                                            .read_range_into(labels_start, labels_end, &mut layer_bytes)
                                            .expect("failed to read layer bytes");
                                    }

                                    let pool = get_p2_pool(core_group_usize.clone());
                                    pool.install(|| {
                                        let res = layer_bytes
                                            .into_par_iter() // TODO CROSSBEAM
                                            .chunks(std::mem::size_of::<Fr>())
                                            .map(|chunk| {
                                                bytes_into_fr(&chunk).expect("Could not create Fr from bytes.")
                                            })
                                            .zip(
                                                data.as_mut()[(start * NODE_SIZE)..(end * NODE_SIZE)]
                                                    .par_chunks_mut(NODE_SIZE),
                                            )
                                            .map(|(key, data_node_bytes)| {
                                                let data_node =
                                                    <Tree::Hasher as Hasher>::Domain::try_from_bytes(
                                                        data_node_bytes,
                                                    )
                                                    .expect("try_from_bytes failed");

                                                let encoded_node = encode::<<Tree::Hasher as Hasher>::Domain>(
                                                    key.into(),
                                                    data_node,
                                                );
                                                data_node_bytes
                                                    .copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                                                encoded_node
                                            });
                                        res
                                    })
                                };

                                node_index += chunked_nodes_count;
                                trace!(
                                    "node index {}/{}/{}",
                                    node_index,
                                    chunked_nodes_count,
                                    nodes_count,
                                );

                                let encoded: Vec<_> =
                                    encoded_data.into_par_iter().map(|x| x.into()).collect();

                                let is_final = node_index == nodes_count;
                                builder_tx
                                    .send((encoded, is_final))
                                    .expect("failed to send encoded");
                            }
                        }));
                    } // threads loop

                    for t in threads {
                        t.join().unwrap();
                    }
                }).unwrap(); // scope s2
            })); // spawn
            let batchertype_gpus = &batchertype_gpus;
            let gpu_indexes: Vec<usize> = (0.. bus_num).collect();

            //Parallel tuning GPU computing
            main_threads.push(s.spawn(move |_| {
                let _cleanup_handle_gpu = bind_thread();
                crossbeam::scope(|s2| {
                    let mut gpu_threads = Vec::new();

                    let writers_tx = Arc::new(writers_tx);

                    for (&gpu_index, builders_rx) in gpu_indexes.iter()
                        .zip(builders_rx_by_gpu.into_iter())
                        {

                        let writers_tx = writers_tx.clone();

                        gpu_threads.push(s2.spawn(move |_| {
                            let _cleanup_handle_gpu_i = bind_thread();
                            let mut locked_gpu: i32 = -1;
                            let lock = loop {
                                let lock_inner = scheduler::get_next_device_second_pool().lock().unwrap();
                                let target_bus_id = lock_inner.device().bus_id().unwrap();
                                
                                for idx in 0..batchertype_gpus.len() {
                                    match &batchertype_gpus[idx] {
                                        BatcherType::CustomGPU(selector) => {
                                            let bus_id = selector.get_device().unwrap().bus_id().unwrap();
                                            if bus_id == target_bus_id {
                                                locked_gpu = idx as i32;
                                            }

                                        }
                                        _default => {
                                            info!("Run ColumnTreeBuilder on non-CustromGPU batcher");
                                        }
                                    }
                                }

                                if locked_gpu != -1 {
                                    break lock_inner;
                                }
                                else {
                                    drop(lock_inner);
                                    info!("GPU was excluded from the avaiable GPUs by settings, wait the next one");
                                }
                            };

                            assert!(locked_gpu >= 0);
                            let locked_gpu: usize = locked_gpu as usize;

                            let mut mem_total: u64 = 0;
                            let mem_used = AtomicU64::new(0);
                            
                            let tree_r_last_config = &tree_r_last_config;
                            let batchertype_gpus = &batchertype_gpus;

                            match &batchertype_gpus[locked_gpu] {
                                BatcherType::CustomGPU(selector) => {
                                    mem_total = selector.get_device().unwrap().memory();

                                    info!("[tree_r_last] Run TreeBuilder over indexes i % gpu_num = {} on {} (buis_id: {})",
                                    gpu_index,
                                    selector.get_device().unwrap().name(),
                                    selector.get_device().unwrap().bus_id().unwrap(),
                                    );
                                }
                                _default => {
                                    info!("Run ColumnTreeBuilder on non-CustromGPU batcher");
                                }
                            }

                            // Loop until all trees for all configs have been built.
                            let config_ids: Vec<_> = (gpu_index..config_count).step_by(bus_num).collect();

                            crossbeam::scope(|s3| {
                                let mut config_threads = Vec::new();
                                let writers_tx = Arc::new(writers_tx);
                                let mem_used = Arc::new(mem_used);

                                // Loop until all trees for all configs have been built.
                                for (&i, builder_rx) in config_ids.iter()
                                    .zip(builders_rx.into_iter())
                                    {
                                    let writers_tx  = writers_tx.clone();
                                    let mem_used = mem_used.clone();
                                    
                                    config_threads.push(s3.spawn(move |_| {
                                        let _cleanup_handle_gpu_inner = bind_thread();
                                        let mut printed = false;
                                        let mut mem_used_val = mem_used.load(SeqCst);
                                        while (mem_used_val + mem_one_thread) as f64 >= (1.0 - gpu_memory_padding) * (mem_total as f64) {
                                            if !printed {
                                                info!("gpu memory shortage on {}, waiting ({})...", locked_gpu, i);
                                                printed = true;
                                            }
                                            thread::sleep(Duration::from_secs(1));
                                            mem_used_val = mem_used.load(SeqCst);
                                        }
                                        mem_used.fetch_add(mem_one_thread, SeqCst);

                                        if printed {
                                            info!("continue on {} ({})", locked_gpu, i);
                                        }

                                        let mut tree_builder = TreeBuilder::<Tree::Arity>::new(
                                            Some(batchertype_gpus[locked_gpu].clone()),
                                            nodes_count,
                                            max_gpu_tree_batch_size,
                                            tree_r_last_config.rows_to_discard,
                                        )
                                        .expect("failed to create TreeBuilder");
                                        
                                        loop {
                                            let (encoded, is_final) =
                                                builder_rx.recv().expect("failed to recv encoded data");
    
                                            // Just add non-final leaf batches.
                                            if !is_final {
                                                tree_builder
                                                    .add_leaves(&encoded)
                                                    .expect("failed to add leaves");
                                                continue;
                                            };
    
                                            // If we get here, this is a final leaf batch: build a sub-tree.
                                            let (_, tree_data) = tree_builder
                                                .add_final_leaves(&encoded)
                                                .expect("failed to add final leaves");
                    
    
                                            mem_used.fetch_sub(mem_one_thread, SeqCst);
                                            let writer_tx = writers_tx[i].clone();
                                            writer_tx.send(tree_data).expect("failed to send tree_data");
                                            break;
                                        }
                                    })); // spawn
                                }

                                for t in config_threads {
                                    t.join().unwrap();
                                }
                            }).unwrap(); // scope s3

                            drop(lock);
                            trace!("[tree_c] set gpu idle={}", locked_gpu);
                        })); // spawn
                    } // gpu loop
                    for t in gpu_threads {
                        t.join().unwrap();
                    }
                }).unwrap(); // scope s2
            }));

            main_threads.push(s.spawn(move |_| {
                let _cleanup_handle_write = bind_thread();
                configs.iter().enumerate()
                    .zip(writers_rx.iter())
                    .for_each(|((_i, config), writer_rx)| {

                    let tree_data = writer_rx
                        .recv()
                        .expect("failed to receive tree_data for tree_r_last");

                    let tree_data_len = tree_data.len();
                    let cache_size = get_merkle_tree_cache_size(
                        get_merkle_tree_leafs(
                            config.size.expect("config size failure"),
                            Tree::Arity::to_usize(),
                        )
                        .expect("failed to get merkle tree leaves"),
                        Tree::Arity::to_usize(),
                        config.rows_to_discard,
                    )
                    .expect("failed to get merkle tree cache size");
                    assert_eq!(tree_data_len, cache_size);

                    let flat_tree_data: Vec<_> = tree_data
                        .into_par_iter()
                        .flat_map(|el| fr_into_bytes(&el))
                        .collect();

                    // Persist the data to the store based on the current config.
                    let tree_r_last_path = StoreConfig::data_path(&config.path, &config.id);
                    trace!(
                        "persisting tree r of len {} with {} rows to discard at path {:?}",
                        tree_data_len,
                        config.rows_to_discard,
                        tree_r_last_path
                    );
                    let mut f = OpenOptions::new()
                        .create(true)
                        .write(true)
                        .open(&tree_r_last_path)
                        .expect("failed to open file for tree_r_last");
                    f.write_all(&flat_tree_data)
                        .expect("failed to wrote tree_r_last data");
                });
            })); //spawn

            for t in main_threads {
                t.join().unwrap();
            }
        }).unwrap(); // scope

        create_lc_tree::<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
            tree_r_last_config.size.expect("config size failure"),
            &configs,
            &replica_config,
        )
    }

    pub fn generate_tree_r_last_cpu<TreeArity>(
        data: &mut Data<'_>,
        nodes_count: usize,
        tree_count: usize,
        tree_r_last_config: StoreConfig,
        replica_path: PathBuf,
        labels: &LabelsCache<Tree>,
    ) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        TreeArity: PoseidonArity,
    {
        info!("generating tree r last using the CPU");

        // ================= CPU POOL ===============
        let groups = get_p2_core_group();
        let mut core_group: Vec<CoreIndex> = vec![];
        let mut core_group_usize: Vec<usize> = vec![];
        
        let use_same_set = p2_binding_use_same_set();
        if use_same_set {
            if let Some(groups) = groups {
                for cg in groups {
                    for core_id in 0..cg.len() {
                        let core_index = cg.get(core_id);
                        if let Some(core_index) = core_index {
                            core_group.push(core_index.clone());
                            core_group_usize.push(core_index.0)
                        }
                    }
                }
            }
        } else {
            if let Some(ref groups) = groups {
                for cg in groups {
                    for core_id in 0..cg.len() {
                        let core_index = cg.get(core_id);
                        if let Some(core_index) = core_index {
                            core_group.push(core_index.clone());
                            core_group_usize.push(core_index.0)
                        }
                    }
                }
            }
        }

        let core_group_usize = Arc::new(core_group_usize);
        // =====

        let pool = get_p2_pool(core_group_usize.clone());
        pool.install(|| {

            let (configs, replica_config) = split_config_and_replica(
                tree_r_last_config.clone(),
                replica_path,
                nodes_count,
                tree_count,
            )?;

            data.ensure_data()?;
            let last_layer_labels = labels.labels_for_last_layer()?;

            let size = Store::len(last_layer_labels);

            let mut start = 0;
            let mut end = size / tree_count;

            for (i, config) in configs.iter().enumerate() {
                let encoded_data = last_layer_labels
                    .read_range(start..end)?
                    .into_par_iter()
                    .zip(
                        data.as_mut()[(start * NODE_SIZE)..(end * NODE_SIZE)]
                            .par_chunks_mut(NODE_SIZE),
                    )
                    .map(|(key, data_node_bytes)| {
                        let data_node =
                            <Tree::Hasher as Hasher>::Domain::try_from_bytes(data_node_bytes)
                                .expect("try from bytes failed");
                        let encoded_node =
                            encode::<<Tree::Hasher as Hasher>::Domain>(key, data_node);
                        data_node_bytes.copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                        encoded_node
                    });

                info!(
                    "building base tree_r_last with CPU {}/{}",
                    i + 1,
                    tree_count
                );
                LCTree::<Tree::Hasher, Tree::Arity, typenum::U0, typenum::U0>::from_par_iter_with_config(encoded_data, config.clone()).with_context(|| format!("failed tree_r_last CPU {}/{}", i + 1, tree_count))?;

                start = end;
                end += size / tree_count;
            }

            create_lc_tree::<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
                tree_r_last_config.size.expect("config size failure"),
                &configs,
                &replica_config,
            )
        })
    }
}