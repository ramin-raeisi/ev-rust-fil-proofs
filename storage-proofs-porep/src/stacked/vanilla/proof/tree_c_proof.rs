use std::sync::{mpsc, Arc, RwLock, Mutex,
    atomic::{AtomicU64, Ordering::SeqCst}};
use std::thread;
use std::time::Duration;

use bellperson::bls::Fr;
use filecoin_hashers::{Hasher, PoseidonArity};
use generic_array::typenum::{self, Unsigned};
use log::*;
use merkletree::store::{DiskStore, StoreConfig};
use rayon::prelude::*;
use storage_proofs_core::{
    error::Result,
    measurements::{
        measure_op,
        Operation::{GenerateTreeC},
    },
    merkle::*,
    settings,
    util::{NODE_SIZE},
};

use super::super::{
    hash::hash_single_column,
    params::{
        LabelsCache
    },
    proof::StackedDrg,
    cores::{bind_core, get_p2_core_group, CoreIndex, Cleanup},
};

use super::utils::{get_memory_padding, get_gpu_for_parallel_tree_r};

use generic_array::{GenericArray};
use neptune::batch_hasher::BatcherType;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use fr32::{bytes_into_fr, fr_into_bytes};

use rust_gpu_tools::opencl;
use bellperson::gpu::{scheduler};

use thread_binder;


impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> {
    #[allow(clippy::needless_range_loop)]
    pub fn generate_tree_c_gpu<ColumnArity, TreeArity>(
        layers: usize,
        nodes_count: usize,
        tree_count: usize,
        configs: Vec<StoreConfig>,
        labels: &LabelsCache<Tree>,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: 'static + PoseidonArity,
        TreeArity: PoseidonArity,
    {
        info!("generating tree c using the GPU");
        // Build the tree for CommC
        measure_op(GenerateTreeC, || {
            info!("Building column hashes");

            // NOTE: The max number of columns we recommend sending to the GPU at once is
            // 400000 for columns and 700000 for trees (conservative soft-limits discussed).
            //
            // 'column_write_batch_size' is how many nodes to chunk the base layer of data
            // into when persisting to disk.
            //
            // Override these values with care using environment variables:
            // FIL_PROOFS_MAX_GPU_COLUMN_BATCH_SIZE, FIL_PROOFS_MAX_GPU_TREE_BATCH_SIZE, and
            // FIL_PROOFS_COLUMN_WRITE_BATCH_SIZE respectively.
            let max_gpu_column_batch_size = settings::SETTINGS.max_gpu_column_batch_size as usize;
            let max_gpu_tree_batch_size = settings::SETTINGS.max_gpu_tree_batch_size as usize;
            let column_write_batch_size = settings::SETTINGS.column_write_batch_size as usize;

            //  ============== GPU POOL ================
            let mut batchertype_gpus = Vec::new();
            //let mut builders_rx = Vec::new();
            let all_bus_ids = opencl::Device::all()
                .iter()
                .map(|d| d.bus_id().unwrap())
                .collect::<Vec<_>>();
            let bus_num = all_bus_ids.len();
            assert!(bus_num > 0);

            let tree_r_gpu = get_gpu_for_parallel_tree_r();
            let mut last_idx = bus_num;
            if tree_r_gpu > 0 { // tree_r_lats will be calculated in parallel with tree_c using tree_r_gpu GPU
                assert!(tree_r_gpu < bus_num, 
                    "tree_r_last are calculating in parallel with tree_c. There is not free GPU for tree_c. Try to decrease gpu_for_parallel_tree_r constant.");
                info!("[tree_c] are calculating in paralle with tree_r_last. It uses {}/{} GPU", bus_num - tree_r_gpu, bus_num);
    
                // tree_c uses first indexes of the GPU list
                last_idx = bus_num - tree_r_gpu;
            }

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
                assert!(trees_per_gpu * last_idx >= configs.len(), "wrong FIL_PROOFS_TREE_PER_GPU value");
                last_idx = ((configs.len() as f64) / (trees_per_gpu as f64)).ceil() as usize;
            }

            let bus_num = last_idx;

            //let batchers_per_gpu = configs.len() / bus_num + 1;
            for gpu_idx in 0..bus_num {
                batchertype_gpus.push(BatcherType::CustomGPU(opencl::GPUSelector::BusId(all_bus_ids[gpu_idx])));
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
                            core_group_usize.push(core_index.0);
                        }
                    }
                }
            }
            let total_cores = core_group.len();
            let core_group = if total_cores > 0 {
                Some(core_group)
            } else {
                None
            };
            let core_group = Arc::new(core_group);
            let core_group_usize = Arc::new(core_group_usize);
            let current_core: usize = 0;
            let current_core = Arc::new(Mutex::new(current_core));

            let get_core_index = |i: usize| -> Option<CoreIndex> {
                if let Some(cg) = &*core_group {
                    Some(cg[i])
                } else {
                    None
                }
            };

            let increment_core = || {
                let mut cc = current_core.lock().unwrap();
                *cc = (*cc + 1) % total_cores;
            };

            let bind_thread = || -> Option<Result<Cleanup>> {
                let cleanup_handle = get_core_index(*current_core.lock().unwrap()).map(
                    |core_index| bind_core(core_index)
                );
                increment_core();
                cleanup_handle
            };
            // =====

            let _cleanup_handle = bind_thread();



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

            let config_count = configs.len(); // Don't move config into closure below.

            let labels = Arc::new(Mutex::new(labels));

            //let size_fr = std::mem::size_of::<Fr>() as u64;
            //let size_state: u64 = size_fr * (layers as u64); // state_{width} = ColumnArity = layers
            //let threads_num: u64 = max_gpu_column_batch_size as u64;
            /*let mut mem_column_add: u64 = 0;
            mem_column_add = mem_column_add + size_fr * ((max_gpu_column_batch_size * layers) as u64); // preimages buffer
            mem_column_add = mem_column_add + size_fr * ((max_gpu_column_batch_size * layers) as u64); // digests buffer
            mem_column_add = mem_column_add + size_state * threads_num; // states per thread*/
            //let mem_column_add = 858993459;
            let mem_column_add = 850000000;
            let gpu_memory_padding = get_memory_padding();

            let configs =  Arc::new(configs);
            crossbeam::scope(|s| {
                let mut main_threads = Vec::new();
                // This channel will receive the finished tree data to be written to disk.
                let mut writers_tx = Vec::new();
                let mut writers_rx = Vec::new();
                for _i in 0..config_count {
                    let (writer_tx, writer_rx) = mpsc::sync_channel::<(Vec<Fr>, Vec<Fr>)>(1);
                    writers_tx.push(writer_tx);
                    writers_rx.push(writer_rx);
                }

                main_threads.push(s.spawn(move |_| {
                    let _cleanup_handle_prepare = bind_thread();
                    crossbeam::scope(|s2| {
                        let mut threads = Vec::new();
                        for (&i, builder_tx) in (0..config_count).collect::<Vec<_>>().iter()
                        .zip(builders_tx.into_iter())
                        {
                            if i != 0 {
                                thread::sleep(Duration::from_secs(4));
                            }
                            let labels = labels.clone();
                            debug!("start spawn tree_c {}", i + 1);
                            let core_group_usize = core_group_usize.clone();
                            threads.push(s2.spawn(move |_| {
                                let _cleanup_handle_prepare_i = bind_thread();
                                let mut node_index = 0;
                                while node_index != nodes_count {
                                    debug!("while tree_c {}, node_index = {}", i + 1, node_index);
                                    let chunked_nodes_count =
                                        std::cmp::min(nodes_count - node_index, max_gpu_column_batch_size);
                                    trace!(
                                        "processing config {}/{} with column nodes {}",
                                        i + 1,
                                        tree_count,
                                        chunked_nodes_count,
                                    );

                                    let columns: Vec<
                                        GenericArray<Fr, ColumnArity>,
                                    > = {
                                        debug!("columns, tree_c {}, node_index = {}", i + 1, node_index);

                                        // Allocate layer data array and insert a placeholder for each layer.
                                        let mut layer_data: Vec<Vec<u8>> =
                                            vec![
                                                vec![0u8; chunked_nodes_count * std::mem::size_of::<Fr>()];
                                                layers
                                            ];

                                        debug!("loop 1, tree_c {}, node_index = {}", i + 1, node_index);
                                        for (layer_index, mut layer_bytes) in
                                            layer_data.iter_mut().enumerate()
                                        {
                                            let labels = labels.lock().unwrap();
                                            trace!("loop 1 into, tree_c {}, node_index = {}, layer_index = {}", i + 1, node_index, layer_index);
                                            let store = labels.labels_for_layer(layer_index + 1);
                                            let start = (i * nodes_count) + node_index;
                                            let end = start + chunked_nodes_count;

                                            store
                                                .read_range_into(start, end, &mut layer_bytes)
                                                .expect("failed to read store range");
                                            debug!("loop 1 store, tree_c {}, node_index = {}, layer_index = {}", i + 1, node_index, layer_index);
                                        }
                                        debug!("loop 1 end, tree_c {}, node_index = {}", i + 1, node_index);

                                        debug!("loop 2, tree_c {}, node_index = {}", i + 1, node_index);
                                        let pool = thread_binder::ThreadPoolBuilder::new_with_core_set(core_group_usize.clone()).build().unwrap();
                                        pool.install(|| {
                                            let res = (0..chunked_nodes_count)
                                                .into_par_iter()
                                                .map(|index| {
                                                    (0..layers)
                                                        .map(|layer_index| {
                                                            trace!("loop 2 into, tree_c {}, node_index = {}, layer_index = {}", i + 1, node_index, layer_index);
                                                            bytes_into_fr(
                                                            &layer_data[layer_index][std::mem::size_of::<Fr>()
                                                                * index
                                                                ..std::mem::size_of::<Fr>() * (index + 1)],
                                                        )
                                                        .expect("Could not create Fr from bytes.")
                                                        })
                                                        .collect::<GenericArray<Fr, ColumnArity>>()
                                                })
                                                .collect();
                                            debug!("loop 2 end, tree_c {}, node_index = {}", i + 1, node_index);
                                            res
                                        })
                                    }; // columns

                                    node_index += chunked_nodes_count;
                                    trace!(
                                        "node index {}/{}/{}",
                                        node_index,
                                        chunked_nodes_count,
                                        nodes_count,
                                    );

                                    let is_final = node_index == nodes_count;
                                    debug!("tree_c {}, new node_index = {}, is_final = {}", i + 1, node_index, is_final);
                                    builder_tx
                                        .send((columns, is_final))
                                        .expect("failed to send columns");
                                    debug!("tree_c {}, new node_index = {}, data was sent", i + 1, node_index);
                                } // while loop
                            }));
                        } // threads loop

                        for t in threads {
                            t.join().unwrap();
                        }
                    }).unwrap(); // scope s2

                    debug!("end spawn");
                })); // spawn
                
                let batchertype_gpus = &batchertype_gpus;
                let gpu_indexes: Vec<usize> = (0.. bus_num).collect();

                //Parallel tuning GPU computing
                main_threads.push(s.spawn(move |_| {
                    let _cleanup_handle_gpu = bind_thread();

                    crossbeam::scope(|s2| {
                        let mut gpu_threads = Vec::new();

                        let writers_tx = Arc::new(writers_tx);

                        //debug!("start spawn2");
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

                                    match &batchertype_gpus[locked_gpu] {
                                        BatcherType::CustomGPU(selector) => {
                                            mem_total = selector.get_device().unwrap().memory();

                                            info!("[tree_c] Run ColumnTreeBuilder over indexes i % gpu_num = {} on {} (buis_id: {}, memory: {})",
                                            gpu_index,
                                            selector.get_device().unwrap().name(),
                                            selector.get_device().unwrap().bus_id().unwrap(),
                                            mem_total,
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
                                        debug!("run spawn2 inner loop");
                                        for (&i, builder_rx) in config_ids.iter()
                                            .zip(builders_rx.into_iter())
                                            {
                                            if i != 0 {
                                                thread::sleep(Duration::from_secs(5));
                                            }
                                            let writers_tx  = writers_tx.clone();
                                            let mem_used = mem_used.clone();
                                            config_threads.push(s3.spawn(move |_| {
                                                let _cleanup_handle_gpu_inner = bind_thread();
                                                let mut printed = false;
                                                let mut mem_used_val = mem_used.load(SeqCst);
                                                while (mem_used_val + mem_column_add) as f64 >= (1.0 - gpu_memory_padding) * (mem_total as f64) {
                                                    if !printed {
                                                        info!("gpu memory shortage on {}, waiting ({})...", locked_gpu, i);
                                                        printed = true;
                                                    }
                                                    thread::sleep(Duration::from_secs(1));
                                                    mem_used_val = mem_used.load(SeqCst);
                                                }
                                                mem_used.fetch_add(mem_column_add, SeqCst);
                                                if printed {
                                                    info!("continue on {} ({})", locked_gpu, i);
                                                    thread::sleep(Duration::from_secs(i as u64));
                                                }

                                                //debug!("create column_tree_builder, tree_c {}", i + 1);
                                                let mut column_tree_builder = ColumnTreeBuilder::<ColumnArity, TreeArity>::new(
                                                    Some(batchertype_gpus[locked_gpu].clone()),
                                                    nodes_count,
                                                    max_gpu_column_batch_size,
                                                    max_gpu_tree_batch_size,
                                                )
                                                .expect("failed to create ColumnTreeBuilder");
                                                
                                                //debug!("loop, tree_c {}", i + 1);
                                                loop {
                                                    //debug!("get columns, tree_c {}", i + 1);
                                                    let (columns, is_final): (Vec<GenericArray<Fr, ColumnArity>>, bool) =
                                                        builder_rx.recv().expect("failed to recv columns");
                                                    //debug!("got columns, tree_c {}, is_final = {}", i + 1, is_final);
                                                    // Just add non-final column batches.
                                                    if !is_final {
                                                        
                                                        //debug!("Use {}/{} GB", (mem_used.load(SeqCst) as f64 / (1024 * 1024 * 1024) as f64), (mem_total as f64 / (1024 * 1024 * 1024) as f64));
                                                        column_tree_builder
                                                            .add_columns(&columns)
                                                            .expect("failed to add columns");
                                                        continue;
                                                    };

                                                    // If we get here, this is a final column: build a sub-tree.
                                                    let (base_data, tree_data) = column_tree_builder
                                                        .add_final_columns(&columns)
                                                        .expect("failed to add final columns");
                                                    trace!(
                                                        "base data len {}, tree data len {}",
                                                        base_data.len(),
                                                        tree_data.len()
                                                    );

                                                    let tree_len = base_data.len() + tree_data.len();

                                                    info!(
                                                        "persisting base tree_c {}/{} of length {}",
                                                        i + 1,
                                                        tree_count,
                                                        tree_len,
                                                    );

                                                    let writer_tx = writers_tx[i].clone();

                                                    mem_used.fetch_sub(mem_column_add, SeqCst);
                                                    writer_tx
                                                        .send((base_data, tree_data))
                                                        .expect("failed to send base_data, tree_data");
                                                    break;
                                                }
                                            }));
                                        } // configs loop

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
                    }).unwrap(); //scope s2
                    debug!("end spawn2");
                }));

                let configs = configs.clone();
                main_threads.push(s.spawn(move |_| {
                    let _cleanup_handle_write = bind_thread();
                    configs.iter().enumerate()
                        .zip(writers_rx.iter())
                        .for_each(|((_i, config), writer_rx)| {
                        //debug!("writing tree_c {}", i + 1);
                        let (base_data, tree_data) = writer_rx
                            .recv()
                            .expect("failed to receive base_data, tree_data for tree_c");
                        let tree_len = base_data.len() + tree_data.len();

                        assert_eq!(base_data.len(), nodes_count);
                        assert_eq!(tree_len, config.size.expect("config size failure"));

                        // Persist the base and tree data to disk based using the current store config.
                        let tree_c_store =
                            DiskStore::<<Tree::Hasher as Hasher>::Domain>::new_with_config(
                                tree_len,
                                Tree::Arity::to_usize(),
                                config.clone(),
                            )
                            .expect("failed to create DiskStore for base tree data");

                        let store = Arc::new(RwLock::new(tree_c_store));
                        let batch_size = std::cmp::min(base_data.len(), column_write_batch_size);
                        let flatten_and_write_store = |data: &Vec<Fr>, offset| {
                            data.into_par_iter()
                                .chunks(batch_size)
                                .enumerate()
                                .try_for_each(|(index, fr_elements)| {
                                    let mut buf = Vec::with_capacity(batch_size * NODE_SIZE);

                                    for fr in fr_elements {
                                        buf.extend(fr_into_bytes(&fr));
                                    }
                                    store
                                        .write()
                                        .expect("failed to access store for write")
                                        .copy_from_slice(&buf[..], offset + (batch_size * index))
                                })
                        };

                        trace!(
                            "flattening tree_c base data of {} nodes using batch size {}",
                            base_data.len(),
                            batch_size
                        );
                        flatten_and_write_store(&base_data, 0)
                            .expect("failed to flatten and write store");

                        let base_offset = base_data.len();
                        trace!("flattening tree_c tree data of {} nodes using batch size {} and base offset {}", tree_data.len(), batch_size, base_offset);
                        flatten_and_write_store(&tree_data, base_offset)
                            .expect("failed to flatten and write store");
                        trace!("done flattening tree_c tree data");

                        store
                            .write()
                            .expect("failed to access store for sync")
                            .sync()
                            .expect("store sync failure");
                        trace!("done writing tree_c store data");
                    });
                }));

                for t in main_threads {
                    t.join().unwrap();
                }
            }).unwrap(); // scope
            
            create_disk_tree::<
                DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
            >(configs[0].size.expect("config size failure"), &configs)
        })
    }

    pub fn generate_tree_c_cpu<ColumnArity, TreeArity>(
        layers: usize,
        nodes_count: usize,
        tree_count: usize,
        configs: Vec<StoreConfig>,
        labels: &LabelsCache<Tree>,
    ) -> Result<DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
    where
        ColumnArity: PoseidonArity,
        TreeArity: PoseidonArity,
    {
        info!("generating tree c using the CPU");
        measure_op(GenerateTreeC, || {
            info!("Building column hashes");

            let mut trees = Vec::with_capacity(tree_count);
            for (i, config) in configs.iter().enumerate() {
                let mut hashes: Vec<<Tree::Hasher as Hasher>::Domain> =
                    vec![<Tree::Hasher as Hasher>::Domain::default(); nodes_count];

                rayon::scope(|s| {
                    let n = num_cpus::get();

                    // only split if we have at least two elements per thread
                    let num_chunks = if n > nodes_count * 2 { 1 } else { n };

                    // chunk into n chunks
                    let chunk_size = (nodes_count as f64 / num_chunks as f64).ceil() as usize;

                    // calculate all n chunks in parallel
                    for (chunk, hashes_chunk) in hashes.chunks_mut(chunk_size).enumerate() {
                        let labels = &labels;

                        s.spawn(move |_| {
                            for (j, hash) in hashes_chunk.iter_mut().enumerate() {
                                let data: Vec<_> = (1..=layers)
                                    .map(|layer| {
                                        let store = labels.labels_for_layer(layer);
                                        let el: <Tree::Hasher as Hasher>::Domain = store
                                            .read_at((i * nodes_count) + j + chunk * chunk_size)
                                            .expect("store read_at failure");
                                        el.into()
                                    })
                                    .collect();

                                *hash = hash_single_column(&data).into();
                            }
                        });
                    }
                });

                info!("building base tree_c {}/{}", i + 1, tree_count);
                trees.push(DiskTree::<
                    Tree::Hasher,
                    Tree::Arity,
                    typenum::U0,
                    typenum::U0,
                >::from_par_iter_with_config(
                    hashes.into_par_iter(), config.clone()
                ));
            }

            assert_eq!(tree_count, trees.len());
            create_disk_tree::<
                DiskTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>,
            >(configs[0].size.expect("config size failure"), &configs)
        })
    }
}