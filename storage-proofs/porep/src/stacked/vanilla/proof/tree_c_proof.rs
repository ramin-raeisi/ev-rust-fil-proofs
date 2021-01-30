use std::sync::{mpsc, Arc, RwLock};
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
};

use ff::Field;
use generic_array::{sequence::GenericSequence, GenericArray};
use neptune::batch_hasher::BatcherType;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use fr32::fr_into_bytes;

use rust_gpu_tools::opencl;



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

            let mut batchertype_gpus = Vec::new();
            //let mut builders_rx = Vec::new();
            let all_bus_ids = opencl::Device::all()
                .iter()
                .map(|d| d.bus_id().unwrap())
                .collect::<Vec<_>>();
            let _bus_num = all_bus_ids.len();
            assert!(_bus_num > 0);

            let tree_r_gpu = settings::SETTINGS.gpu_for_parallel_tree_r as usize;
            let mut last_idx = _bus_num;
            if tree_r_gpu > 0 { // tree_r_lats will be calculated in parallel with tree_c using tree_r_gpu GPU
                assert!(tree_r_gpu < _bus_num, 
                    "tree_r_last are calculating in parallel with tree_c. There is not free GPU for tree_c. Try to decrease gpu_for_parallel_tree_r constant.");
                info!("[tree_c] are calculating in paralle with tree_r_last. It uses {}/{} GPU", _bus_num - tree_r_gpu, _bus_num);
    
                // tree_c uses first indexes of the GPU list
                last_idx = _bus_num - tree_r_gpu;
            }

            for gpu_idx in 0..last_idx {
                batchertype_gpus.push(BatcherType::CustomGPU(opencl::GPUSelector::BusId(all_bus_ids[gpu_idx])));
            }

            let mut builders_rx_by_gpu = Vec::new();
            let mut builders_tx = Vec::new();
            for _i in 0.._bus_num {
                builders_rx_by_gpu.push(Vec::new());
            }

            for config_idx in 0..configs.len() {
                // This channel will receive batches of columns and add them to the ColumnTreeBuilder.
                // Each config has own channel
                let (builder_tx, builder_rx) = mpsc::sync_channel(0);
                builders_tx.push(builder_tx);
                builders_rx_by_gpu[config_idx % _bus_num].push(builder_rx);
            }

            // This channel will receive the finished tree data to be written to disk.
            let mut writers_tx = Vec::new();
            let mut writers_rx = Vec::new();
            for _i in 0..configs.len() {
                let (writer_tx, writer_rx) = mpsc::sync_channel::<(Vec<Fr>, Vec<Fr>)>(0);
                writers_tx.push(writer_tx);
                writers_rx.push(writer_rx);
            }

            let _bus_num = batchertype_gpus.len();
            assert!(_bus_num > 0);

            // Use this set of read-write locks to control GPU threads
            let mut gpu_busy_flag = Vec::new();
            for _ in 0.._bus_num {
                gpu_busy_flag.push(Arc::new(RwLock::new(0)))
            }

            let config_count = configs.len(); // Don't move config into closure below.
            rayon::scope(|s| {
                s.spawn(move |_| {
                    (0..config_count).collect::<Vec<_>>().par_iter()
                        .zip(builders_tx.into_par_iter())
                        .for_each( |(&i, builder_tx)| {
                        
                        let mut node_index = 0;
                        while node_index != nodes_count {
                            let chunked_nodes_count =
                                std::cmp::min(nodes_count - node_index, max_gpu_column_batch_size);
                            trace!(
                                "processing config {}/{} with column nodes {}",
                                i + 1,
                                tree_count,
                                chunked_nodes_count,
                            );
                            let mut columns: Vec<GenericArray<Fr, ColumnArity>> = vec![
                                GenericArray::<Fr, ColumnArity>::generate(|_i: usize| Fr::zero());
                                chunked_nodes_count
                            ];

                            // Allocate layer data array and insert a placeholder for each layer.
                            let mut layer_data: Vec<Vec<Fr>> =
                                vec![Vec::with_capacity(chunked_nodes_count); layers];

                            rayon::scope(|s| {
                                // capture a shadowed version of layer_data.
                                let layer_data: &mut Vec<_> = &mut layer_data;

                                // gather all layer data in parallel.
                                s.spawn(move |_| {
                                    for (layer_index, layer_elements) in
                                        layer_data.iter_mut().enumerate()
                                    {
                                        let store = labels.labels_for_layer(layer_index + 1);
                                        let start = (i * nodes_count) + node_index;
                                        let end = start + chunked_nodes_count;
                                        let elements: Vec<<Tree::Hasher as Hasher>::Domain> = store
                                            .read_range(std::ops::Range { start, end })
                                            .expect("failed to read store range");
                                        layer_elements.extend(elements.into_iter().map(Into::into));
                                    }
                                });
                            });

                            // Copy out all layer data arranged into columns.
                            for layer_index in 0..layers {
                                for index in 0..chunked_nodes_count {
                                    columns[index][layer_index] = layer_data[layer_index][index];
                                }
                            }

                            drop(layer_data);

                            node_index += chunked_nodes_count;
                            trace!(
                                "node index {}/{}/{}",
                                node_index,
                                chunked_nodes_count,
                                nodes_count,
                            );

                            let is_final = node_index == nodes_count;
                            builder_tx
                                .send((columns, is_final))
                                .expect("failed to send columns");
                        }
                    });
                }); // spawn

                let batchertype_gpus = &batchertype_gpus;
                let gpu_indexes: Vec<usize> = (0.. _bus_num).collect();

                //Parallel tuning GPU computing
                s.spawn(move |_| {
                    gpu_indexes.par_iter()
                        .zip(builders_rx_by_gpu.into_par_iter())
                        .for_each( |(&gpu_index, builders_rx)| {
                        
                        let gpu_busy_flag = gpu_busy_flag.clone();
                        // TODO-Ryan: find_idle_gpu
                        let mut find_idle_gpu: i32 = -1;
                        loop {
                            for i in 0.._bus_num {
                                if *gpu_busy_flag[i].read().unwrap() == 0 {
                                    *gpu_busy_flag[i].write().unwrap() = 1;
                                    find_idle_gpu = i as i32;

                                    trace!("[tree_c] find_idle_gpu={}, gpu_index={}", find_idle_gpu, gpu_index);
                                    break;
                                }
                            }

                            if find_idle_gpu == -1 {
                                thread::sleep(Duration::from_millis(1));
                            } else {
                                break;
                            }
                        }

                        assert!(find_idle_gpu >= 0);
                        let find_idle_gpu: usize = find_idle_gpu as usize;

                        match &batchertype_gpus[find_idle_gpu] {
                            BatcherType::CustomGPU(selector) => {
                                info!("[tree_c] Run ColumnTreeBuilder over indexes i*{} on {} (buis_id: {})",
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
                        let config_ids: Vec<_> = (0 + gpu_index..config_count).step_by(_bus_num).collect();

                        config_ids.par_iter()
                            .zip(builders_rx.into_par_iter())
                            .for_each( |(&i, builder_rx)| {

                            let mut column_tree_builder = ColumnTreeBuilder::<ColumnArity, TreeArity>::new(
                                Some(batchertype_gpus[find_idle_gpu].clone()),
                                nodes_count,
                                max_gpu_column_batch_size,
                                max_gpu_tree_batch_size,
                            )
                            .expect("failed to create ColumnTreeBuilder");

                            info!("Run builder for tree_c {}/{}", i + 1, tree_count);
                            
                            loop {
                                let (columns, is_final): (Vec<GenericArray<Fr, ColumnArity>>, bool) =
                                    builder_rx.recv().expect("failed to recv columns");

                                // Just add non-final column batches.
                                if !is_final {
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

                                writer_tx
                                    .send((base_data, tree_data))
                                    .expect("failed to send base_data, tree_data");
                                break;
                            }
                        }); // configs loop

                        *gpu_busy_flag[find_idle_gpu].write().unwrap() = 0; // TODO-Ryan: After the store is completed, enter the preparation for the next tree (adopted by the amd platform)
                        trace!("[tree_c] set gpu idle={}", find_idle_gpu);
                    }); // gpu loop
                });
            }); // rayon::scope

            
            configs.iter()
                .zip(writers_rx.iter())
                .for_each(|(config, writer_rx)| {

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
                trace!("done flattening tree_c base data");

                let base_offset = base_data.len();
                trace!("flattening tree_c tree data of {} nodes using batch size {} and base offset {}", tree_data.len(), batch_size, base_offset);
                flatten_and_write_store(&tree_data, base_offset)
                    .expect("failed to flatten and write store");
                trace!("done flattening tree_c tree data");

                trace!("writing tree_c store data");
                store
                    .write()
                    .expect("failed to access store for sync")
                    .sync()
                    .expect("store sync failure");
                trace!("done writing tree_c store data");
            });
            
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