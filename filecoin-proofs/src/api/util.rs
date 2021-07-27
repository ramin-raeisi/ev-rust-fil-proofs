use std::mem::size_of;

use anyhow::{Context, Result};
use bellperson::bls::Fr;
use filecoin_hashers::{Domain, Hasher};
use fr32::{bytes_into_fr, fr_into_bytes};
use merkletree::merkle::{get_merkle_tree_leafs, get_merkle_tree_len};
use storage_proofs_core::merkle::{get_base_tree_count, MerkleTreeTrait};
use typenum::Unsigned;
use log::error;

use crate::types::{Commitment, SectorSize};

pub fn as_safe_commitment<H: Domain, T: AsRef<str>>(
    comm: &[u8; 32],
    commitment_name: T,
) -> Result<H> {
    bytes_into_fr(comm)
        .map(Into::into)
        .with_context(|| format!("Invalid commitment ({})", commitment_name.as_ref(), ))
}

pub fn commitment_from_fr(fr: Fr) -> Commitment {
    let mut commitment = [0; 32];
    for (i, b) in fr_into_bytes(&fr).iter().enumerate() {
        commitment[i] = *b;
    }
    commitment
}

pub fn get_base_tree_size<Tree: MerkleTreeTrait>(sector_size: SectorSize) -> Result<usize> {
    let base_tree_leaves = u64::from(sector_size) as usize
        / size_of::<<Tree::Hasher as Hasher>::Domain>()
        / get_base_tree_count::<Tree>();

    get_merkle_tree_len(base_tree_leaves, Tree::Arity::to_usize())
}

pub fn get_base_tree_leafs<Tree: MerkleTreeTrait>(base_tree_size: usize) -> Result<usize> {
    get_merkle_tree_leafs(base_tree_size, Tree::Arity::to_usize())
}

pub fn bind_p1_tree() -> bool {
    let res: usize = std::env::var("FIL_PROOFS_BIND_P1_TREE")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_PROOFS_BIND_P1_TREE! Defaulting to {:?}", 0);
                Ok(0)
            }
        })
        .unwrap_or(0);
    res != 0
}