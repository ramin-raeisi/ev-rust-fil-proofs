use std::fs::{metadata};
use std::path::{Path};

use anyhow::{ensure, Result};
use log::{info};
use merkletree::store::{StoreConfig};
use storage_proofs_core::{
    cache_key::CacheKey,
    compound_proof::{self, CompoundProof},
    merkle::{MerkleTreeTrait},
    sector::SectorId,
    util::default_rows_to_discard,
};
use storage_proofs_porep::stacked::{
    generate_replica_id, StackedCompound, StackedDrg,
};

use crate::{
    api::{get_base_tree_leafs, get_base_tree_size},
    constants::{
        DefaultBinaryTree, DefaultPieceHasher,
    },
    parameters::setup_params,
    types::{
        PaddedBytesAmount, PieceInfo, PoRepConfig, PoRepProofPartitions, ProverId,
        SealPreCommitPhase1Output,
        Ticket, BINARY_ARITY,
    },
};

use super::seal::compute_comm_d;

#[allow(clippy::too_many_arguments)]
pub fn generate_labels_bench<R, Tree: 'static + MerkleTreeTrait>(
    porep_config: PoRepConfig,
    cache_path: R,
    prover_id: ProverId,
    sector_id: SectorId,
    ticket: Ticket,
    piece_infos: &[PieceInfo],
) -> Result<()>
    where
        R: AsRef<Path>,
{
    info!("generate_labels_bench:start: {:?}", sector_id);

    ensure!(
        metadata(cache_path.as_ref())?.is_dir(),
        "cache_path must be a directory"
    );

    let compound_setup_params = compound_proof::SetupParams {
        vanilla_params: setup_params(
            PaddedBytesAmount::from(porep_config),
            usize::from(PoRepProofPartitions::from(porep_config)),
            porep_config.porep_id,
            porep_config.api_version,
        )?,
        partitions: Some(usize::from(PoRepProofPartitions::from(porep_config))),
        priority: false,
    };

    let compound_public_params = <StackedCompound<Tree, DefaultPieceHasher> as CompoundProof<
        StackedDrg<'_, Tree, DefaultPieceHasher>,
        _,
    >>::setup(&compound_setup_params)?;

    let base_tree_size = get_base_tree_size::<DefaultBinaryTree>(porep_config.sector_size)?;
    let base_tree_leafs = get_base_tree_leafs::<DefaultBinaryTree>(base_tree_size)?;
    // MT for original data is always named tree-d, and it will be
    // referenced later in the process as such.
    let config = StoreConfig::new(
        cache_path.as_ref(),
        CacheKey::CommDTree.to_string(),
        default_rows_to_discard(base_tree_leafs, BINARY_ARITY),
    );

    let comm_d = compute_comm_d(porep_config.into(), piece_infos)?;

    let replica_id = generate_replica_id::<Tree::Hasher, _>(
        &prover_id,
        sector_id.into(),
        &ticket,
        comm_d,
        &porep_config.porep_id,
    );

    let _labels = StackedDrg::<Tree, DefaultPieceHasher>::replicate_phase1_bench(
        &compound_public_params.vanilla_params,
        &replica_id,
        config.clone(),
        2,
    )?;

    info!("generate_labels_bench:finish: {:?}", sector_id);
    Ok(())
}