use std::fs::OpenOptions;
use std::io::Write;
use std::path::{PathBuf};
use std::sync::{mpsc, Arc, RwLock};
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
    measurements::{
        measure_op,
        Operation::{GenerateTreeRLast},
    },
    merkle::*,
    settings,
    util::{NODE_SIZE},
};

use super::super::{
    params::{
        LabelsCache,
    },
    proof::StackedDrg,
};

use neptune::batch_hasher::BatcherType;
use neptune::tree_builder::{TreeBuilder, TreeBuilderTrait};
use fr32::fr_into_bytes;

use rust_gpu_tools::opencl;

use crate::encode::{encode};

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> { 
    // Internally, it has been changed to use GPU in parallel (after tree_c is built in parallel, tree_r_last is built in parallel)
    //#[cfg(feature = "tree_c-serial-tree_r_last")]
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

        // Ryan 
        let mut batchertype_gpus = Vec::new();
        let mut builders_rx = Vec::new();
        let mut builders_tx = Vec::new();
        let all_bus_ids = opencl::Device::all()
            .iter()
            .map(|d| d.bus_id().unwrap())
            .collect::<Vec<_>>();
        let _bus_num = all_bus_ids.len();
        assert!(_bus_num > 0);

        let tree_r_gpu = settings::SETTINGS.gpu_for_parallel_tree_r as usize;
        if (tree_r_gpu > 0) { // tree_r_lats will be calculated in parallel with tree_c using tree_r_gpu GPU
            assert!(tree_r_gpu < _bus_num, 
                "tree_r_last are calculating in parallel with tree_c. There is not free GPU for tree_c. Try to decrease gpu_for_parallel_tree_r constant.");
            info!("[tree_r_last] are calculating in paralle with tree_c. It uses {}/{} GPU", tree_r_gpu, _bus_num);

            // tree_r_last uses last indexes of the GPU list
            let start_idx = _bus_num - tree_r_gpu;
            for gpu_index in start_idx.._bus_num {
                batchertype_gpus.push(Some(BatcherType::CustomGPU
                    (opencl::GPUSelector::BusId(all_bus_ids[gpu_index]))));
    
                // These channels will receive batches of leaf nodes and add them to the TreeBuilder.
                // Each GPU has own channel
                let (builder_tx, builder_rx) = mpsc::sync_channel::<(Vec<Fr>, bool)>(0);
                builders_tx.push(builder_tx);
                builders_rx.push(builder_rx);
            };
        } else {
            for gpu_index in 0.._bus_num {
                batchertype_gpus.push(Some(BatcherType::CustomGPU
                    (opencl::GPUSelector::BusId(all_bus_ids[gpu_index]))));
    
                // These channels will receive batches of leaf nodes and add them to the TreeBuilder.
                // Each GPU has own channel
                let (builder_tx, builder_rx) = mpsc::sync_channel::<(Vec<Fr>, bool)>(0);
                builders_tx.push(builder_tx);
                builders_rx.push(builder_rx);
            };
        }

        let _bus_num = batchertype_gpus.len();
        assert!(_bus_num > 0);

        // Use this set of read-write locks to control GPU threads
        let mut gpu_busy_flag = Vec::new();
        for _ in 0.._bus_num {
            gpu_busy_flag.push(Arc::new(RwLock::new(0)))
        }
        // Ryan End

        let config_count = configs.len(); // Don't move config into closure below.
        let configs = &configs;
        let tree_r_last_config = &tree_r_last_config;
        rayon::scope(|s| {
            // This channel will receive the finished tree data to be written to disk.
            let (writer_tx, writer_rx) = mpsc::sync_channel::<Vec<Fr>>(0);
            
            s.spawn(move |_| {
                for i in (0..config_count).step_by(_bus_num) {
                    // loop over _bus_num sync channels
                    for gpu_index in 0.._bus_num {
                        let i = i + gpu_index;
                        if i >= config_count {
                            break;
                        }

                        let builder_tx = builders_tx[gpu_index].clone();

                        let mut node_index = 0;
                        while node_index != nodes_count {
                            let chunked_nodes_count =
                                std::cmp::min(nodes_count - node_index, max_gpu_tree_batch_size);
                            let start = (i * nodes_count) + node_index;
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

                            let encoded_data = last_layer_labels
                                .read_range(start..end)
                                .expect("failed to read layer range")
                                .into_par_iter()
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
                                    let encoded_node =
                                        encode::<<Tree::Hasher as Hasher>::Domain>(key, data_node);
                                    data_node_bytes
                                        .copy_from_slice(AsRef::<[u8]>::as_ref(&encoded_node));

                                    encoded_node
                                });

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
                    } // gpu loop
                } // outer loop
            });
            let batchertype_gpus = &batchertype_gpus;
            let gpu_indexes: Vec<usize> = (0.. _bus_num).collect();

            //Parallel tuning GPU computing
            s.spawn(move |_| {
                gpu_indexes.par_iter()
                    .map(|gpu_index| { *gpu_index } )
                    .zip(builders_rx.into_par_iter())
                    .for_each( |(gpu_index, builder_rx)| {
                    
                    let gpu_busy_flag = gpu_busy_flag.clone();
                    // TODO-Ryan: find_idle_gpu
                    info!("[tree_r_last] begin to find idle gpu");
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
                    info!("[tree_r_last] Use multi GPUs, total_gpu={}, use_gpu_index={}", _bus_num, gpu_index);
                    
                    let tree_r_last_config = &tree_r_last_config;
                    let batchertype_gpus = &batchertype_gpus;
                    
                    let writer_tx = writer_tx.clone();

                    info!("[tree_r_last] Use multi GPUs, total_gpu={}, use_gpu_index={}", _bus_num, gpu_index);
                    let mut tree_builder = TreeBuilder::<Tree::Arity>::new(
                        batchertype_gpus[find_idle_gpu].clone(),
                        nodes_count,
                        max_gpu_tree_batch_size,
                        tree_r_last_config.rows_to_discard,
                    )
                    .expect("failed to create TreeBuilder");

                    // Loop until all trees for all configs have been built.
                    for i in (0..config_count).step_by(_bus_num) {
                        // Move on to the next config.
                        let i = i + gpu_index;
                        if i >= config_count {
                            break;
                        }

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
                            info!(
                                "building base tree_r_last with GPU {}/{}",
                                i + 1,
                                tree_count
                            );
                            let (_, tree_data) = tree_builder
                                .add_final_leaves(&encoded)
                                .expect("failed to add final leaves");

                            writer_tx.send(tree_data).expect("failed to send tree_data");
                            break;
                        }
                    }

                    *gpu_busy_flag[find_idle_gpu].write().unwrap() = 0; // TODO-Ryan: After the store is completed, enter the preparation for the next tree (adopted by the amd platform)
                    trace!("[tree_c] set gpu idle={}", find_idle_gpu);
                }); // gpu loop
            });

            for config in configs.iter() {
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
            }
        });

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
    }
}