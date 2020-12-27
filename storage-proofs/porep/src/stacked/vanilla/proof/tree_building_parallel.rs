use std::marker::PhantomData;
use std::path::{PathBuf};

use anyhow::Context;
use filecoin_hashers::{HashFunction, Hasher};
use generic_array::typenum::{self, Unsigned};
use log::*;
use merkletree::merkle::{
    get_merkle_tree_len,
    is_merkle_tree_size_valid,
};
use merkletree::store::{StoreConfig};
use rayon::prelude::*;
use storage_proofs_core::{
    cache_key::CacheKey,
    data::Data,
    drgraph::Graph,
    error::Result,
    measurements::{
        measure_op,
        Operation::{CommD, GenerateTreeRLast},
    },
    merkle::*,
    util::{default_rows_to_discard, NODE_SIZE},
};
use typenum::{U11, U2, U8};

use super::super::{
    challenges::LayerChallenges,
    graph::StackedBucketGraph,
    params::{
        Labels, LabelsCache, PersistentAux,
        Tau, TemporaryAux, TransformedLayers, BINARY_ARITY,
    },
    proof::StackedDrg,
};

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedDrg<'a, Tree, G> {
    //FIXME-Ryan: tree_r_last(Only one GPU is used internally) Built in parallel with tree_c
    // The calculation logic of Precommit2 is implemented (tree_c and tree_r_last are parallel) (When there are more than 8 graphics cards, enable this paragraph, and use the last card for tree_r_last (that is, the 9th card))
    pub(crate) fn transform_and_replicate_layers_inner_parallel(
        graph: &StackedBucketGraph<Tree::Hasher>,
        layer_challenges: &LayerChallenges,
        mut data: Data<'_>,
        data_tree: Option<BinaryMerkleTree<G>>,
        config: StoreConfig,
        replica_path: PathBuf,
        label_configs: Labels<Tree>,
    ) -> Result<TransformedLayers<Tree, G>> {
        trace!("transform_and_replicate_layers");
        let nodes_count = graph.size();

        assert_eq!(data.len(), nodes_count * NODE_SIZE);
        trace!("nodes count {}, data len {}", nodes_count, data.len());

        let tree_count = get_base_tree_count::<Tree>();
        let nodes_count = graph.size() / tree_count;

        // Ensure that the node count will work for binary and oct arities.
        let binary_arity_valid = is_merkle_tree_size_valid(nodes_count, BINARY_ARITY);
        let other_arity_valid = is_merkle_tree_size_valid(nodes_count, Tree::Arity::to_usize());
        trace!(
            "is_merkle_tree_size_valid({}, BINARY_ARITY) = {}",
            nodes_count,
            binary_arity_valid
        );
        trace!(
            "is_merkle_tree_size_valid({}, {}) = {}",
            nodes_count,
            Tree::Arity::to_usize(),
            other_arity_valid
        );
        assert!(binary_arity_valid);
        assert!(other_arity_valid);

        let layers = layer_challenges.layers();
        assert!(layers > 0);

        // Generate all store configs that we need based on the
        // cache_path in the specified config.
        let mut tree_d_config = StoreConfig::from_config(
            &config,
            CacheKey::CommDTree.to_string(),
            Some(get_merkle_tree_len(nodes_count, BINARY_ARITY)?),
        );
        tree_d_config.rows_to_discard = default_rows_to_discard(nodes_count, BINARY_ARITY);

        let mut tree_r_last_config = StoreConfig::from_config(
            &config,
            CacheKey::CommRLastTree.to_string(),
            Some(get_merkle_tree_len(nodes_count, Tree::Arity::to_usize())?),
        );

        // A default 'rows_to_discard' value will be chosen for tree_r_last, unless the user overrides this value via the
        // environment setting (FIL_PROOFS_ROWS_TO_DISCARD).  If this value is specified, no checking is done on it and it may
        // result in a broken configuration.  Use with caution.  It must be noted that if/when this unchecked value is passed
        // through merkle_light, merkle_light now does a check that does not allow us to discard more rows than is possible
        // to discard.
        tree_r_last_config.rows_to_discard =
            default_rows_to_discard(nodes_count, Tree::Arity::to_usize());
        trace!(
            "tree_r_last using rows_to_discard={}",
            tree_r_last_config.rows_to_discard
        );

        let mut tree_c_config = StoreConfig::from_config(
            &config,
            CacheKey::CommCTree.to_string(),
            Some(get_merkle_tree_len(nodes_count, Tree::Arity::to_usize())?),
        );
        tree_c_config.rows_to_discard =
            default_rows_to_discard(nodes_count, Tree::Arity::to_usize());

        let labels =
            LabelsCache::<Tree>::new(&label_configs).context("failed to create labels cache")?;
        let configs = split_config(tree_c_config.clone(), tree_count)?;

        match fdlimit::raise_fd_limit() {
            Some(res) => {
                info!("Building trees [{} descriptors max available]", res);
            }
            None => error!("Failed to raise the fd limit"),
        };

        // FIXME-Ryan: The following 2 steps in P2 can be parallel
        let mut tree_c_root: <Tree::Hasher as Hasher>::Domain = <Tree::Hasher as Hasher>::Domain::default();
        let mut tree_d_root: <G as filecoin_hashers::Hasher>::Domain = <G as filecoin_hashers::Hasher>::Domain::default();
        let mut tree_r_last_root: <Tree::Hasher as Hasher>::Domain = <Tree::Hasher as Hasher>::Domain::default();

        rayon::scope(|s| {

            // capture a shadowed version of datas.
            let tree_c_root = &mut tree_c_root;
            let tree_d_root = &mut tree_d_root;
            let tree_r_last_root = &mut tree_r_last_root;

            let labels = &labels;
            let tree_d_config = &mut tree_d_config;
            let tree_r_last_config = &tree_r_last_config;

            // 1)[gpu] Column Hash calculation
            s.spawn(move |_| {
                info!("[tree_c] building tree_c in parallel with tree_r");
                *tree_c_root = match layers {
                    2 => {
                        let tree_c = Self::generate_tree_c::<U2, Tree::Arity>(
                            layers,
                            nodes_count,
                            tree_count,
                            configs,
                            &labels,
                        ).expect("failed to generate_tree_c U2");
                        tree_c.root()
                    }
                    8 => {
                        let tree_c = Self::generate_tree_c::<U8, Tree::Arity>(
                            layers,
                            nodes_count,
                            tree_count,
                            configs,
                            &labels,
                        ).expect("failed to generate_tree_c U8");
                        tree_c.root()
                    }
                    11 => {
                        let tree_c = Self::generate_tree_c::<U11, Tree::Arity>(
                            layers,
                            nodes_count,
                            tree_count,
                            configs,
                            &labels,
                        ).expect("failed to generate_tree_c U11");
                        tree_c.root()
                    }
                    _ => panic!("Unsupported column arity"),
                };
                info!("tree_c done");
            });

            s.spawn(move |_| {
                // 2) [cpu] Build the MerkleTree over the original data (if needed).
                let tree_d = match data_tree {
                    Some(t) => {
                        trace!("using existing original data merkle tree");
                        assert_eq!(t.len(), 2 * (data.len() / NODE_SIZE) - 1);

                        t
                    }
                    None => {
                        trace!("building merkle tree for the original data");
                        data.ensure_data().expect("failed to data.ensure_data");
                        measure_op(CommD, || {
                            Self::build_binary_tree::<G>(data.as_ref(), tree_d_config.clone())
                        }).expect("failed to tree_d measure_op")
                    }
                };
                tree_d_config.size = Some(tree_d.len());
                assert_eq!(
                    tree_d_config.size.expect("config size failure"),
                    tree_d.len()
                );
                *tree_d_root = tree_d.root();
                drop(tree_d);

                // You have to wait for the second step to be executed before the third step
                // 3) [gpu] Encode original data into the last layer.
                info!("[tree_r_last] building tree_r_last in parallel with tree_c");
                let tree_r_last = measure_op(GenerateTreeRLast, || {
                    Self::generate_tree_r_last::<Tree::Arity>(
                        &mut data,
                        nodes_count,
                        tree_count,
                        tree_r_last_config.clone(),
                        replica_path.clone(),
                        &labels,
                    )
                    .context("failed to generate tree_r_last")
                }).expect("failed to generate tree_r_last");
                info!("tree_r_last done");

                *tree_r_last_root = tree_r_last.root();
                drop(tree_r_last);

                data.drop_data(); 
            });
        });

        // comm_r = H(comm_c || comm_r_last)
        let comm_r: <Tree::Hasher as Hasher>::Domain =
            <Tree::Hasher as Hasher>::Function::hash2(&tree_c_root, &tree_r_last_root);

        Ok((
            Tau {
                comm_d: tree_d_root,
                comm_r,
            },
            PersistentAux {
                comm_c: tree_c_root,
                comm_r_last: tree_r_last_root,
            },
            TemporaryAux {
                labels: label_configs,
                tree_d_config,
                tree_r_last_config,
                tree_c_config,
                _g: PhantomData,
            },
        ))
    }

    #[cfg(feature = "tree_c-parallel-tree_r_last")]
    fn generate_tree_r_last<TreeArity>(
        data: &mut Data,
        nodes_count: usize,
        tree_count: usize,
        tree_r_last_config: StoreConfig,
        replica_path: PathBuf,
        labels: &LabelsCache<Tree>,
    ) -> Result<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>
        where
            TreeArity: PoseidonArity,
    {
        info!("[generate_tree_r_last] tree_c-parallel-tree_r_last");
        let (configs, replica_config) = split_config_and_replica(
            tree_r_last_config.clone(),
            replica_path,
            nodes_count,
            tree_count,
        )?;

        data.ensure_data()?;
        let last_layer_labels = labels.labels_for_last_layer()?;

        if settings::SETTINGS.use_gpu_tree_builder {    // generate_tree_r_last
            info!("[tree_r_last] generating tree r last using the GPU");
            let max_gpu_tree_batch_size =
                settings::SETTINGS.max_gpu_tree_batch_size as usize;

            // Ryan
            let mut batchertype_gpus = Vec::new(); // FIXME-Ryan: batchertype_gpus
            if settings::SETTINGS.use_gpu_tree_builder {
                let all_bus_ids = opencl::Device::all()
                    .unwrap()
                    .iter()
                    .map(|d| d.bus_id())
                    .collect::<Vec<_>>();
                let _bus_num = all_bus_ids.len();
                assert!(_bus_num > 0);
                for gpu_index in 0.._bus_num {
                    batchertype_gpus.push(Some(BatcherType::CustomGPU(opencl::GPUSelector::BusId(all_bus_ids[gpu_index]))));
                };
            }

            let _bus_num = batchertype_gpus.len();
            assert!(_bus_num > 0);
            let batchertype_gpu = batchertype_gpus[_bus_num - 1];  // FIXME-Ryan: //Use the last GPU

            // let all_bus_ids = opencl::Device::all()
            //                     .unwrap()
            //                     .iter()
            //                     .map(|d| d.bus_id())
            //                     .collect::<Vec<_>>();
            // let _bus_num = all_bus_ids.len();
            // assert!(_bus_num>0);
            // let batchertype_gpu = match _bus_num {
            //     1 => Some(BatcherType::GPU),
            //     x => Some(BatcherType::CustomGPU(opencl::GPUSelector::BusId(all_bus_ids[x-1]))),
            // };
            // Ryan End

            // This channel will receive batches of leaf nodes and add them to the TreeBuilder.
            let (builder_tx, builder_rx) = mpsc::sync_channel::<(Vec<Fr>, bool)>(0);
            let config_count = configs.len(); // Don't move config into closure below.
            let configs = &configs;
            rayon::scope(|s| {
                // This is CPU operation, prepare for GPU operation
                s.spawn(move |_| {
                    for i in 0..config_count {
                        let mut node_index = 0;
                        while node_index != nodes_count {
                            let chunked_nodes_count =
                                std::cmp::min(nodes_count - node_index, max_gpu_tree_batch_size);
                            let start = (i * nodes_count) + node_index;
                            let end = start + chunked_nodes_count;
                            trace!(
                                "[tree_r_last] processing config {}/{} with leaf nodes {} [{}, {}, {}-{}]",
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
                                "[tree_r_last] leaf node index {}/{}/{}",
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
                    }
                });

                { // Parallel tuning GPU computing
                    let tree_r_last_config = &tree_r_last_config;
                    s.spawn(move |_| {
                        let mut tree_builder = TreeBuilder::<Tree::Arity>::new(       // GPU construction of Merkle tree neptune
                                                                                      // Some(BatcherType::GPU),
                                                                                      batchertype_gpu,  // FIXME-Ryan: //Use the last GPU    for `generate_tree_r_last`
                                                                                      nodes_count,
                                                                                      max_gpu_tree_batch_size,
                                                                                      tree_r_last_config.rows_to_discard,
                        )
                            .expect("failed to create TreeBuilder");

                        let mut i = 0;
                        let mut config = &configs[i];

                        // Loop until all trees for all configs have been built.
                        while i < configs.len() {
                            let (encoded, is_final) =
                                builder_rx.recv().expect("failed to recv encoded data");

                            // Just add non-final leaf batches.
                            if !is_final {
                                // info!("[tree_r_last] generating tree r last using the GPU - begin to use GPU");
                                tree_builder
                                    .add_leaves(&encoded)
                                    .expect("failed to add leaves");
                                continue;
                            };

                            // If we get here, this is a final leaf batch: build a sub-tree.
                            info!(
                                "[tree_r_last] building base tree_r_last with GPU {}/{}",
                                i + 1,
                                tree_count
                            );
                            // info!("[tree_r_last] generating tree r last using the GPU - begin to use GPU (final)");
                            let (_, tree_data) = tree_builder
                                .add_final_leaves(&encoded)
                                .expect("failed to add final leaves");
                            let tree_data_len = tree_data.len();
                            let cache_size = get_merkle_tree_cache_size(
                                get_merkle_tree_leafs(
                                    config.size.unwrap(),
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
                                "[tree_r_last] persisting tree r of len {} with {} rows to discard at path {:?}",
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

                            // Move on to the next config.
                            i += 1;
                            if i == configs.len() {
                                break;
                            }
                            config = &configs[i];
                        }
                    });
                }
            });

            info!("[tree_r_last] generating tree r last using the GPU done");
        } else {
            info!("generating tree r last using the CPU");
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
                LCTree::<Tree::Hasher, Tree::Arity, typenum::U0, typenum::U0>::from_par_iter_with_config(encoded_data, config.clone())?;

                start = end;
                end += size / tree_count;
            }
        };

        create_lc_tree::<LCTree<Tree::Hasher, Tree::Arity, Tree::SubTreeArity, Tree::TopTreeArity>>(
            tree_r_last_config.size.unwrap(),
            &configs,
            &replica_config,
        )
    }
}