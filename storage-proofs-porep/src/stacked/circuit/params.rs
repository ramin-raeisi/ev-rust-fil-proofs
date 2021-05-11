use std::marker::PhantomData;
use log::*;
use std::time::Instant;
use std::sync::{Arc, RwLock};

use bellperson::{
    bls::{Bls12, Fr, Engine},
    gadgets::{boolean::Boolean, num::AllocatedNum, uint32::UInt32},
    ConstraintSystem, SynthesisError, Index, Variable,
};
use filecoin_hashers::{Hasher, PoseidonArity};
use generic_array::typenum::{U0, U2};
use storage_proofs_core::{
    drgraph::Graph,
    gadgets::por::{AuthPath, PoRCircuit},
    gadgets::{encode::encode, uint64::UInt64, variables::Root},
    merkle::{DiskStore, MerkleProofTrait, MerkleTreeTrait, MerkleTreeWrapper},
    util::reverse_bit_numbering,
};
use rayon::prelude::*;

use crate::stacked::{
    circuit::{column_proof::ColumnProof, column::AllocatedColumn, create_label_circuit, hash::hash_single_column},
    vanilla::{
        Proof as VanillaProof, PublicParams, ReplicaColumnProof as VanillaReplicaColumnProof,},
};

type TreeAuthPath<T> = AuthPath<
    <T as MerkleTreeTrait>::Hasher,
    <T as MerkleTreeTrait>::Arity,
    <T as MerkleTreeTrait>::SubTreeArity,
    <T as MerkleTreeTrait>::TopTreeArity,
>;

type TreeColumnProof<T> = ColumnProof<
    <T as MerkleTreeTrait>::Hasher,
    <T as MerkleTreeTrait>::Arity,
    <T as MerkleTreeTrait>::SubTreeArity,
    <T as MerkleTreeTrait>::TopTreeArity,
>;

/// Proof for a single challenge.
#[derive(Debug)]
pub struct Proof<Tree: MerkleTreeTrait, G: Hasher> {
    /// Inclusion path for the challenged data node in tree D.
    pub comm_d_path: AuthPath<G, U2, U0, U0>,
    /// The value of the challenged data node.
    pub data_leaf: Option<Fr>,
    /// The index of the challenged node.
    pub challenge: Option<u64>,
    /// Inclusion path of the challenged replica node in tree R.
    pub comm_r_last_path: TreeAuthPath<Tree>,
    /// Inclusion path of the column hash of the challenged node  in tree C.
    pub comm_c_path: TreeAuthPath<Tree>,
    /// Column proofs for the drg parents.
    pub drg_parents_proofs: Vec<TreeColumnProof<Tree>>,
    /// Column proofs for the expander parents.
    pub exp_parents_proofs: Vec<TreeColumnProof<Tree>>,
    _t: PhantomData<Tree>,
}

// We must manually implement Clone for all types generic over MerkleTreeTrait (instead of using
// #[derive(Clone)]) because derive(Clone) will only expand for MerkleTreeTrait types that also
// implement Clone. Not every MerkleTreeTrait type is Clone-able because not all merkel Store's are
// Clone-able, therefore deriving Clone would impl Clone for less than all possible Tree types.
impl<Tree: MerkleTreeTrait, G: 'static + Hasher> Clone for Proof<Tree, G> {
    fn clone(&self) -> Self {
        Proof {
            comm_d_path: self.comm_d_path.clone(),
            data_leaf: self.data_leaf,
            challenge: self.challenge,
            comm_r_last_path: self.comm_r_last_path.clone(),
            comm_c_path: self.comm_c_path.clone(),
            drg_parents_proofs: self.drg_parents_proofs.clone(),
            exp_parents_proofs: self.exp_parents_proofs.clone(),
            _t: self._t,
        }
    }
}

impl<Tree: MerkleTreeTrait, G: 'static + Hasher> Proof<Tree, G> {
    /// Create an empty proof, used in `blank_circuit`s.
    pub fn empty(params: &PublicParams<Tree>) -> Self {
        Proof {
            comm_d_path: AuthPath::blank(params.graph.size()),
            data_leaf: None,
            challenge: None,
            comm_r_last_path: AuthPath::blank(params.graph.size()),
            comm_c_path: AuthPath::blank(params.graph.size()),
            drg_parents_proofs: vec![
                ColumnProof::empty(params);
                params.graph.base_graph().degree()
            ],
            exp_parents_proofs: vec![ColumnProof::empty(params); params.graph.expansion_degree()],
            _t: PhantomData,
        }
    }
    /// Circuit synthesis.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize<CS: ConstraintSystem<Bls12>>(
        self,
        mut cs: CS,
        layers: usize,
        comm_d: &AllocatedNum<Bls12>,
        comm_c: &AllocatedNum<Bls12>,
        comm_r_last: &AllocatedNum<Bls12>,
        replica_id: &[Boolean],
        shift: usize,
    //) -> (Vec<AllocatedNum<Bls12>>, TreeAuthPath<Tree>, TreeAuthPath<Tree>) {
    ) -> Result<(), SynthesisError> {
        let Proof {
            comm_d_path,
            data_leaf,
            challenge,
            comm_r_last_path,
            comm_c_path,
            drg_parents_proofs,
            exp_parents_proofs,
            ..
        } = self;

        let start = Instant::now();

        assert!(!drg_parents_proofs.is_empty());
        assert!(!exp_parents_proofs.is_empty());

        let data_leaf_num = AllocatedNum::alloc(cs.namespace(|| "data_leaf"), || {
            data_leaf.ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        info!{"data leaf index = {}", cs.get_index(&mut data_leaf_num.get_variable())};

        // enforce inclusion of the data leaf in the tree D
        enforce_inclusion(
            cs.namespace(|| "comm_d_inclusion"),
            comm_d_path,
            comm_d,
            &data_leaf_num,
        ).unwrap();

        let alloc_count;
        if layers == 2 {
            alloc_count = 4609;
        }
        else {
            alloc_count = 5979;
        }

        // -- verify replica column openings

        // Private Inputs for the DRG parent nodes.
        let drg_len = drg_parents_proofs.len();
        let exp_len = exp_parents_proofs.len();
        let parents_proofs = vec![drg_parents_proofs, exp_parents_proofs];
        let mut drg_parents = Vec::new();
        let mut exp_parents = Vec::new();
        let mut drg_cs = cs.make_vector(drg_len).unwrap();
        let mut exp_cs = cs.make_vector(exp_len).unwrap();
        let mut data_parents = vec![&mut drg_parents, &mut exp_parents];
        let mut all_cs = vec![drg_cs, exp_cs];
        let aux_len = cs.get_aux_assigment_len();

        let time = start.elapsed();
        info!{"initial params: {:?}", time};

        let start = Instant::now();
        parents_proofs.into_par_iter().enumerate()
        .zip(data_parents.par_iter_mut()
        .zip(all_cs.par_iter_mut()))
        .for_each( |((k, proofs), (parents, css))| {
            let len = proofs.len();
    
            for i in 0..len {
                parents.push(AllocatedColumn::empty(layers));
            }    

            proofs.into_par_iter().enumerate()
            .zip(css.par_iter_mut()
            .zip(parents.par_iter_mut()))
            .for_each(|((i, parent), (other_cs, node_parent))| {
                let (parent_col, inclusion_path) = 
                parent.alloc(other_cs.namespace(|| format!("drg_parent_{}_num", i))).unwrap();
                assert_eq!(layers, parent_col.len());

                let val = parent_col.hash(other_cs.namespace(|| format!("drg_parent_{}_constraint", i))).unwrap();

                *node_parent = parent_col;

                for j in 1..node_parent.len() + 1 {
                    let mut row = node_parent.get_mut_value(j);
                    let mut v = row.get_mut_variable();
                    other_cs.align_variable(&mut v, 0, aux_len + j - 1 + (i + k*drg_len)*alloc_count);
                    other_cs.print_index(&mut v);
                }

                let comm_c_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_c_{}_num", 0)), 
                || { comm_c.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                let idx = other_cs.get_index(&mut comm_c.get_variable());
                other_cs.align_variable(&mut comm_c_copy.get_variable(), 0, idx);
                enforce_inclusion(
                    other_cs.namespace(|| format!("drg_parent_{}_inclusion", i)),
                    inclusion_path,
                    &comm_c_copy,
                    //comm_c,
                    &val,
                );
                other_cs.deallocate(comm_c_copy.get_variable()).unwrap();
            });
        });
        let time = start.elapsed();
        info!{"parallel computation: {:?}", time};
        for new_cs in all_cs {
            cs.aggregate(new_cs);
        }
       // -- Verify labeling and encoding
       let start = Instant::now();
       // stores the labels of the challenged column
       let mut column_labels = Vec::new();

       // PublicInput: challenge index
       let challenge_num = UInt64::alloc(cs.namespace(|| "challenge"), challenge).unwrap();
       challenge_num.pack_into_input(cs.namespace(|| "challenge input")).unwrap();

       for layer in 1..=layers {
           let layer_num = UInt32::constant(layer as u32);

           let mut cs = cs.namespace(|| format!("labeling_{}", layer));

           // Collect the parents
           let mut parents = Vec::new();

           // all layers have drg parents
           //info!("start : drg_parents");
           for parent_col in &drg_parents {
               let parent_val_num = parent_col.get_value(layer);
               let parent_val_bits =
                   reverse_bit_numbering(parent_val_num.to_bits_le(
                       cs.namespace(|| format!("drg_parent_{}_bits", parents.len())),
                   ).unwrap());
               parents.push(parent_val_bits);
           }
           //info!("start : exp_parents");
           // the first layer does not contain expander parents
           if layer > 1 {
               for parent_col in &exp_parents {
                   // subtract 1 from the layer index, as the exp parents, are shifted by one, as they
                   // do not store a value for the first layer
                   let parent_val_num = parent_col.get_value(layer - 1);
                   let parent_val_bits = reverse_bit_numbering(parent_val_num.to_bits_le(
                       cs.namespace(|| format!("exp_parent_{}_bits", parents.len())),
                   ).unwrap());
                   parents.push(parent_val_bits);
               }
           } 
           // Duplicate parents, according to the hashing algorithm.
           let mut expanded_parents = parents.clone();
           if layer > 1 {
               expanded_parents.extend_from_slice(&parents); // 28
               expanded_parents.extend_from_slice(&parents[..9]); // 37
           } else {
               // layer 1 only has drg parents
               expanded_parents.extend_from_slice(&parents); // 12
               expanded_parents.extend_from_slice(&parents); // 18
               expanded_parents.extend_from_slice(&parents); // 24
               expanded_parents.extend_from_slice(&parents); // 30
               expanded_parents.extend_from_slice(&parents); // 36
               expanded_parents.push(parents[0].clone()); // 37
           };
          // info!("start : create label circuit");
           // Reconstruct the label
           let mut label = create_label_circuit(
               cs.namespace(|| "create_label"),
               replica_id,
               expanded_parents,
               layer_num,
               challenge_num.clone(),
           ).unwrap();
           column_labels.push(label);
       }


        // -- encoding node
        {
            // encode the node
 
            // key is the last label
            let mut key = &column_labels[column_labels.len() - 1];
            info!("key index = {:?}", cs.get_index(&mut key.get_variable()));
            info!("data_leaf_num index = {:?}", cs.get_index(&mut data_leaf_num.get_variable()));
 
 
             info!("start : encode node");
            let mut encoded_node = encode(cs.namespace(|| "encode_node"), key, &data_leaf_num).unwrap();
 
             /*info!{"align encoded node"};
            let mut v = encoded_node.get_mut_variable();
            let idx = cs.get_index(&mut v);
            info!{"index of necode node: {}", idx};
            cs.align_variable(&mut v, 0, idx + shift);*/
            info!{"enforce encoded node"}
 
            // verify inclusion of the encoded node
            enforce_inclusion(
                cs.namespace(|| "comm_r_last_data_inclusion"),
                comm_r_last_path,
                comm_r_last,
                &encoded_node,
            ).unwrap();
            /*info!{"dealign encoded node"};
            let mut v = encoded_node.get_mut_variable();
            cs.align_variable(&mut v, 0, idx); */
        }
      
       info!("start : enforce hash");
       // -- ensure the column hash of the labels is included
       {
           // calculate column_hash
           let column_hash =
               hash_single_column(cs.namespace(|| "c_x_column_hash"), &column_labels)?;

            /*info!("start : align parents");
            for col in &mut column_labels {
                let mut v = col.get_mut_variable();
                let idx = cs.get_index(&mut v);
                info!{"label index = {}", idx};
                cs.align_variable(&mut v, 0, idx + shift);
            };*/

           // enforce inclusion of the column hash in the tree C
           enforce_inclusion(
               cs.namespace(|| "c_x_inclusion"),
               comm_c_path,
               comm_c,
               &column_hash,
           )?;
       }

      // info!{"pack into input"}
       //challenge_num.pack_into_input_with_align(cs.namespace(|| "challenge input"), input)?;
       let time = start.elapsed();
      // info!("finish pack into input");
       info!{"labeling: {:?}", time};
       //(column_labels, comm_c_path, comm_r_last_path)
       Ok(())
   }

   /*pub fn labelling<CS: ConstraintSystem<Bls12>>(
    mut cs: CS,
    layers: usize,
    comm_c_path: TreeAuthPath<Tree>,
    comm_r_last_path: &AllocatedNum<Bls12>,
    replica_id: &[Boolean],
    shift: usize,
    data_leaf_num: &AllocatedNum<Bls12>,
    column_labels:&Vec<AllocatedNum<Bls12>>,
    ) -> Result<(), SynthesisError> {


    // -- Verify labeling and encoding
    let start = Instant::now();
    // stores the labels of the challenged column
    let mut column_labels = Vec::new();

    // PublicInput: challenge index
    let challenge_num = UInt64::alloc(cs.namespace(|| "challenge"), challenge).unwrap();
    challenge_num.pack_into_input(cs.namespace(|| "challenge input")).unwrap();

    for layer in 1..=layers {
        let layer_num = UInt32::constant(layer as u32);

        let mut cs = cs.namespace(|| format!("labeling_{}", layer));

        // Collect the parents
        let mut parents = Vec::new();

        // all layers have drg parents
        //info!("start : drg_parents");
        for parent_col in &drg_parents {
            let parent_val_num = parent_col.get_value(layer);
            let parent_val_bits =
                reverse_bit_numbering(parent_val_num.to_bits_le(
                    cs.namespace(|| format!("drg_parent_{}_bits", parents.len())),
                ).unwrap());
            parents.push(parent_val_bits);
        }
        //info!("start : exp_parents");
        // the first layer does not contain expander parents
        if layer > 1 {
            for parent_col in &exp_parents {
                // subtract 1 from the layer index, as the exp parents, are shifted by one, as they
                // do not store a value for the first layer
                let parent_val_num = parent_col.get_value(layer - 1);
                let parent_val_bits = reverse_bit_numbering(parent_val_num.to_bits_le(
                    cs.namespace(|| format!("exp_parent_{}_bits", parents.len())),
                ).unwrap());
                parents.push(parent_val_bits);
            }
        } 
        // Duplicate parents, according to the hashing algorithm.
        let mut expanded_parents = parents.clone();
        if layer > 1 {
            expanded_parents.extend_from_slice(&parents); // 28
            expanded_parents.extend_from_slice(&parents[..9]); // 37
        } else {
            // layer 1 only has drg parents
            expanded_parents.extend_from_slice(&parents); // 12
            expanded_parents.extend_from_slice(&parents); // 18
            expanded_parents.extend_from_slice(&parents); // 24
            expanded_parents.extend_from_slice(&parents); // 30
            expanded_parents.extend_from_slice(&parents); // 36
            expanded_parents.push(parents[0].clone()); // 37
        };
       // info!("start : create label circuit");
        // Reconstruct the label
        let mut label = create_label_circuit(
            cs.namespace(|| "create_label"),
            replica_id,
            expanded_parents,
            layer_num,
            challenge_num.clone(),
        ).unwrap();
        column_labels.push(label);
    }


     // -- encoding node
     {
         // encode the node

         // key is the last label
         let mut key = &column_labels[column_labels.len() - 1];
         info!("key index = {:?}", cs.get_index(&mut key.get_variable()));
         info!("data_leaf_num index = {:?}", cs.get_index(&mut data_leaf_num.get_variable()));


          info!("start : encode node");
         let mut encoded_node = encode(cs.namespace(|| "encode_node"), key, &data_leaf_num).unwrap();

          /*info!{"align encoded node"};
         let mut v = encoded_node.get_mut_variable();
         let idx = cs.get_index(&mut v);
         info!{"index of necode node: {}", idx};
         cs.align_variable(&mut v, 0, idx + shift);*/
         info!{"enforce encoded node"}

         // verify inclusion of the encoded node
         enforce_inclusion(
             cs.namespace(|| "comm_r_last_data_inclusion"),
             comm_r_last_path,
             comm_r_last,
             &encoded_node,
         ).unwrap();
         /*info!{"dealign encoded node"};
         let mut v = encoded_node.get_mut_variable();
         cs.align_variable(&mut v, 0, idx); */
     }
   
    info!("start : enforce hash");
    // -- ensure the column hash of the labels is included
    {
        // calculate column_hash
        let column_hash =
            hash_single_column(cs.namespace(|| "c_x_column_hash"), &column_labels)?;

         /*info!("start : align parents");
         for col in &mut column_labels {
             let mut v = col.get_mut_variable();
             let idx = cs.get_index(&mut v);
             info!{"label index = {}", idx};
             cs.align_variable(&mut v, 0, idx + shift);
         };*/

        // enforce inclusion of the column hash in the tree C
        enforce_inclusion(
            cs.namespace(|| "c_x_inclusion"),
            comm_c_path,
            comm_c,
            &column_hash,
        )?;
    }

    let time = start.elapsed();
    info!{"labeling: {:?}", time};

        Ok(())
    }

    pub fn make_parents<CS: ConstraintSystem<Bls12>>(
        self,
        mut cs: CS,
        layers: usize,
        comm_d: &AllocatedNum<Bls12>,
        comm_c: &AllocatedNum<Bls12>,
        comm_r_last: &AllocatedNum<Bls12>,
        replica_id: &[Boolean],
        data_leaf_num: &AllocatedNum<Bls12>,
     ) -> (Vec<AllocatedColumn<Bls12>>, Vec<AllocatedColumn<Bls12>>, TreeAuthPath<Tree>, TreeAuthPath<Tree>, Option<u64>) {
        let Proof {
            comm_d_path,
            data_leaf,
            challenge,
            comm_r_last_path,
            comm_c_path,
            drg_parents_proofs,
            exp_parents_proofs,
            ..
        } = self;

        let start = Instant::now();

        assert!(!drg_parents_proofs.is_empty());
        assert!(!exp_parents_proofs.is_empty());


        // enforce inclusion of the data leaf in the tree D
        enforce_inclusion(
            cs.namespace(|| "comm_d_inclusion"),
            comm_d_path,
            comm_d,
            &data_leaf_num,
        ).unwrap();

        let alloc_count;
        if layers == 2 {
            alloc_count = 4609;
        }
        else {
            alloc_count = 5979;
        }

        // -- verify replica column openings

        // Private Inputs for the DRG parent nodes.
        let drg_len = drg_parents_proofs.len();
        let exp_len = exp_parents_proofs.len();
        let parents_proofs = vec![drg_parents_proofs, exp_parents_proofs];
        let mut drg_parents = Vec::new();
        let mut exp_parents = Vec::new();
        let mut drg_cs = cs.make_vector(drg_len).unwrap();
        let mut exp_cs = cs.make_vector(exp_len).unwrap();
        let mut data_parents = vec![&mut drg_parents, &mut exp_parents];
        let mut all_cs = vec![drg_cs, exp_cs];
        let aux_len = cs.get_aux_assigment_len();

        let time = start.elapsed();
        info!{"initial params: {:?}", time};

        let start = Instant::now();
        parents_proofs.into_par_iter().enumerate()
        .zip(data_parents.par_iter_mut()
        .zip(all_cs.par_iter_mut()))
        .for_each( |((k, proofs), (parents, css))| {
            let len = proofs.len();
    
            for i in 0..len {
                parents.push(AllocatedColumn::empty(layers));
            }    

            proofs.into_par_iter().enumerate()
            .zip(css.par_iter_mut()
            .zip(parents.par_iter_mut()))
            .for_each(|((i, parent), (other_cs, node_parent))| {
                let (parent_col, inclusion_path) = 
                parent.alloc(other_cs.namespace(|| format!("drg_parent_{}_num", i))).unwrap();
                assert_eq!(layers, parent_col.len());

                let val = parent_col.hash(other_cs.namespace(|| format!("drg_parent_{}_constraint", i))).unwrap();

                *node_parent = parent_col;

                for j in 1..node_parent.len() + 1 {
                    let mut row = node_parent.get_mut_value(j);
                    let mut v = row.get_mut_variable();
                    other_cs.align_variable(&mut v, 0, aux_len + j - 1 + (i + k*drg_len)*alloc_count);
                }

                let comm_c_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_c_{}_num", 0)), 
                || { comm_c.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                let idx = other_cs.get_index(&mut comm_c.get_variable());
                other_cs.align_variable(&mut comm_c_copy.get_variable(), 0, idx);
                enforce_inclusion(
                    other_cs.namespace(|| format!("drg_parent_{}_inclusion", i)),
                    inclusion_path,
                    &comm_c_copy,
                    &val,
                );
                other_cs.deallocate(comm_c_copy.get_variable()).unwrap();
            });
        });
        let time = start.elapsed();
        info!{"parallel computation: {:?}", time};
        for new_cs in all_cs {
            cs.aggregate(new_cs);
        }
        (drg_parents, exp_parents, comm_c_path, comm_r_last_path, challenge)
    }*/

}

impl<Tree: MerkleTreeTrait, G: Hasher> From<VanillaProof<Tree, G>> for Proof<Tree, G>
    where
        Tree::Hasher: 'static,
{
    fn from(vanilla_proof: VanillaProof<Tree, G>) -> Self {
        let VanillaProof {
            comm_d_proofs,
            comm_r_last_proof,
            replica_column_proofs,
            labeling_proofs,
            ..
        } = vanilla_proof;
        let VanillaReplicaColumnProof {
            c_x,
            drg_parents,
            exp_parents,
        } = replica_column_proofs;

        let data_leaf = Some(comm_d_proofs.leaf().into());

        Proof {
            comm_d_path: comm_d_proofs.as_options().into(),
            data_leaf,
            challenge: Some(labeling_proofs[0].node),
            comm_r_last_path: comm_r_last_proof.as_options().into(),
            comm_c_path: c_x.inclusion_proof.as_options().into(),
            drg_parents_proofs: drg_parents.into_iter().map(|p| p.into()).collect(),
            exp_parents_proofs: exp_parents.into_iter().map(|p| p.into()).collect(),
            _t: PhantomData,
        }
    }
}

/// Enforce the inclusion of the given path, to the given leaf and the root.
pub fn enforce_inclusion<H, U, V, W, CS: ConstraintSystem<Bls12>>(
    cs: CS,
    path: AuthPath<H, U, V, W>,
    root: &AllocatedNum<Bls12>,
    leaf: &AllocatedNum<Bls12>,
) -> Result<(), SynthesisError>
    where
        H: 'static + Hasher,
        U: 'static + PoseidonArity,
        V: 'static + PoseidonArity,
        W: 'static + PoseidonArity,
{
    info!{"get root from allocated"};
    let root = Root::from_allocated::<CS>(root.clone());
    info!{"get leaf from allocated"};
    let leaf = Root::from_allocated::<CS>(leaf.clone());

    info!{"start: synthesize por circuit"};
    PoRCircuit::<MerkleTreeWrapper<H, DiskStore<H::Domain>, U, V, W>>::synthesize(
        cs, leaf, path, root, true,
    )?;

    Ok(())
}
