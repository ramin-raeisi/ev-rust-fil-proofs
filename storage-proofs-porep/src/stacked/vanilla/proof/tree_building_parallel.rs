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

                data.drop_data().unwrap(); 
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
}