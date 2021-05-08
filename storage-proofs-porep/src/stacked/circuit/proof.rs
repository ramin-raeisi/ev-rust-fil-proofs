use std::marker::PhantomData;
use log::*;
use anyhow::ensure;
use std::time::Instant;
use bellperson::{
    bls::{Bls12, Fr},
    gadgets::{num::AllocatedNum, uint32::UInt32},
    Circuit, ConstraintSystem, SynthesisError, groth16::prover::ProvingAssignment,
};
use filecoin_hashers::{HashFunction, Hasher};
use fr32::u64_into_fr;
use rayon::prelude::*;
use storage_proofs_core::{
    compound_proof::{CircuitComponent, CompoundProof},
    drgraph::Graph,
    error::Result,
    gadgets::{constraint, encode::encode, por::PoRCompound, uint64::UInt64},
    merkle::{BinaryMerkleTree, MerkleTreeTrait},
    parameter_cache::{CacheableParameters, ParameterSetMetadata},
    por::{self, PoR},
    proof::ProofScheme,
    util::reverse_bit_numbering,
};

use crate::stacked::{circuit::{params::{Proof, enforce_inclusion}, create_label_circuit, hash::hash_single_column, column::AllocatedColumn}, StackedDrg};

/// Stacked DRG based Proof of Replication.
///
/// # Fields
///
/// * `params` - parameters for the curve
///
pub struct StackedCircuit<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> {
    public_params: <StackedDrg<'a, Tree, G> as ProofScheme<'a>>::PublicParams,
    replica_id: Option<<Tree::Hasher as Hasher>::Domain>,
    comm_d: Option<G::Domain>,
    comm_r: Option<<Tree::Hasher as Hasher>::Domain>,
    comm_r_last: Option<<Tree::Hasher as Hasher>::Domain>,
    comm_c: Option<<Tree::Hasher as Hasher>::Domain>,

    // one proof per challenge
    proofs: Vec<Proof<Tree, G>>,
}

// We must manually implement Clone for all types generic over MerkleTreeTrait (instead of using
// #[derive(Clone)]) because derive(Clone) will only expand for MerkleTreeTrait types that also
// implement Clone. Not every MerkleTreeTrait type is Clone-able because not all merkel Store's are
// Clone-able, therefore deriving Clone would impl Clone for less than all possible Tree types.
impl<'a, Tree: MerkleTreeTrait, G: Hasher> Clone for StackedCircuit<'a, Tree, G> {
    fn clone(&self) -> Self {
        StackedCircuit {
            public_params: self.public_params.clone(),
            replica_id: self.replica_id,
            comm_d: self.comm_d,
            comm_r: self.comm_r,
            comm_r_last: self.comm_r_last,
            comm_c: self.comm_c,
            proofs: self.proofs.clone(),
        }
    }
}

impl<'a, Tree: MerkleTreeTrait, G: Hasher> CircuitComponent for StackedCircuit<'a, Tree, G> {
    type ComponentPrivateInputs = ();
}

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher> StackedCircuit<'a, Tree, G> {
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize<CS>(
        mut cs: CS,
        public_params: <StackedDrg<'a, Tree, G> as ProofScheme<'a>>::PublicParams,
        replica_id: Option<<Tree::Hasher as Hasher>::Domain>,
        comm_d: Option<G::Domain>,
        comm_r: Option<<Tree::Hasher as Hasher>::Domain>,
        comm_r_last: Option<<Tree::Hasher as Hasher>::Domain>,
        comm_c: Option<<Tree::Hasher as Hasher>::Domain>,
        proofs: Vec<Proof<Tree, G>>,
    ) -> Result<(), SynthesisError>
        where
            CS: ConstraintSystem<Bls12>,
    {
        let circuit = StackedCircuit::<'a, Tree, G> {
            public_params,
            replica_id,
            comm_d,
            comm_r,
            comm_r_last,
            comm_c,
            proofs,
        };

        circuit.synthesize(&mut cs)
    }
}

impl<'a, Tree: MerkleTreeTrait, G: Hasher> Circuit<Bls12> for StackedCircuit<'a, Tree, G> {
    /*fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let StackedCircuit {
            public_params,
            proofs,
            replica_id,
            comm_r,
            comm_d,
            comm_r_last,
            comm_c,
            ..
        } = self;

        // Allocate replica_id
        let replica_id_num = AllocatedNum::alloc(cs.namespace(|| "replica_id"), || {
            replica_id
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make replica_id a public input
        replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;

        let replica_id_bits =
            reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

        // Allocate comm_d as Fr
        let comm_d_num = AllocatedNum::alloc(cs.namespace(|| "comm_d"), || {
            comm_d
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_d a public input
        comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

        // Allocate comm_r as Fr
        let comm_r_num = AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            comm_r
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_r a public input
        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // Allocate comm_r_last as Fr
        let comm_r_last_num = AllocatedNum::alloc(cs.namespace(|| "comm_r_last"), || {
            comm_r_last
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Allocate comm_c as Fr
        let comm_c_num = AllocatedNum::alloc(cs.namespace(|| "comm_c"), || {
            comm_c
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Verify comm_r = H(comm_c || comm_r_last)
        {
            let hash_num = <Tree::Hasher as Hasher>::Function::hash2_circuit(
                cs.namespace(|| "H_comm_c_comm_r_last"),
                &comm_c_num,
                &comm_r_last_num,
            )?;

            // Check actual equality
            constraint::equal(
                cs,
                || "enforce comm_r = H(comm_c || comm_r_last)",
                &comm_r_num,
                &hash_num,
            );
        }
        let gen_aux = cs.get_aux_assigment_len();
        let gen_input = cs.get_input_assigment_len();
        let len = proofs.len();
        let mut gen_cs = cs.make_vector(len)?;
        let proof_alloc_aux = 7231629;
        let proof_alloc_input = 18;
        let mut challenges = Vec::new();

        
        for ((i, proof), other_cs) in proofs.into_iter().enumerate().zip(gen_cs.iter_mut()) {
            let input_len1 = other_cs.get_input_assigment_len();
            info!{"input_len_1 = {}, i = {}", input_len1, i};
            /*let comm_d_num_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_d_{}_num", 0)), 
            || { comm_d_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
            comm_d_num_copy.inputize(cs.namespace(|| "comm_d_input"))?;
            let idx = other_cs.get_index(&mut comm_d_num.get_variable());
            other_cs.align_variable(&mut comm_d_num_copy.get_variable(), 0, idx);

            let comm_c_num_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_c_{}_num", 0)), 
            || { comm_c_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
            let idx = other_cs.get_index(&mut comm_c_num.get_variable());
            other_cs.align_variable(&mut comm_c_num_copy.get_variable(), 0, idx);

            let comm_r_last_num_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_r_last_{}_num", 0)), 
            || { comm_r_last_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
            let idx = other_cs.get_index(&mut comm_r_last_num.get_variable());
            other_cs.align_variable(&mut comm_r_last_num_copy.get_variable(), 0, idx);

            //let aux_shift = gen_aux + i * proof_alloc_aux - 3*(i+1);
            let aux_shift = 0;
            let mut challenge = proof.synthesize(
                &mut other_cs.namespace(|| format!("challenge_{}", i)),
                public_params.layer_challenges.layers(),
                &comm_d_num_copy,
                &comm_c_num_copy,
                &comm_r_last_num_copy,
                &replica_id_bits,
                aux_shift,
            )?;*/
            let mut aux_shift = 0;
            let mut challenge = proof.synthesize(
                &mut other_cs.namespace(|| format!("challenge_{}", i)),
                public_params.layer_challenges.layers(),
                &comm_d_num,
                &comm_c_num,
                &comm_r_last_num,
                &replica_id_bits,
                aux_shift,
            )?;
            //challenges.push(challenge);
            /*other_cs.deallocate(comm_d_num_copy.get_variable()).unwrap();
            other_cs.deallocate(comm_d_num_copy.get_variable()).unwrap();
            other_cs.deallocate(comm_c_num_copy.get_variable()).unwrap();
            other_cs.deallocate(comm_r_last_num_copy.get_variable()).unwrap();*/
            let input_len2 = other_cs.get_input_assigment_len();
            info!{"input_len_2 = {}, i = {}", input_len2, i};
            aux_shift = gen_aux + i * proof_alloc_aux - 3*(i+1);
            //for (i, aux) in other_cs.aux_assignment {

           // }
        }
        //cs.aggregate_without_inputs(gen_cs);
        cs.aggregate(gen_cs);
        /* for (i,challenge) in challenges.into_iter().enumerate() {
            challenge.pack_into_input(cs.namespace(|| "challenge input"))?;
        }*/

        let gen_aux = cs.get_aux_assigment_len();
        info!{"final aux assigment: {}", gen_aux};

        Ok(())
    }*/
    
    /*fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let StackedCircuit {
            public_params,
            proofs,
            replica_id,
            comm_r,
            comm_d,
            comm_r_last,
            comm_c,
            ..
        } = self;

        // Allocate replica_id
        let replica_id_num = AllocatedNum::alloc(cs.namespace(|| "replica_id"), || {
            replica_id
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make replica_id a public input
        replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;

        let replica_id_bits =
            reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

        // Allocate comm_d as Fr
        let comm_d_num = AllocatedNum::alloc(cs.namespace(|| "comm_d"), || {
            comm_d
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_d a public input
        comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

        // Allocate comm_r as Fr
        let comm_r_num = AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            comm_r
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_r a public input
        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // Allocate comm_r_last as Fr
        let comm_r_last_num = AllocatedNum::alloc(cs.namespace(|| "comm_r_last"), || {
            comm_r_last
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Allocate comm_c as Fr
        let comm_c_num = AllocatedNum::alloc(cs.namespace(|| "comm_c"), || {
            comm_c
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Verify comm_r = H(comm_c || comm_r_last)
        {
            let hash_num = <Tree::Hasher as Hasher>::Function::hash2_circuit(
                cs.namespace(|| "H_comm_c_comm_r_last"),
                &comm_c_num,
                &comm_r_last_num,
            )?;

            // Check actual equality
            constraint::equal(
                cs,
                || "enforce comm_r = H(comm_c || comm_r_last)",
                &comm_r_num,
                &hash_num,
            );
        }
        let mut column_labels_vec = Vec::new();
        let mut comm_c_path_vec = Vec::new();
        //let mut comm_r_last_path_vec = Vec::new();
        let mut data_leaf_num_vec = Vec::new();
        let len = proofs.len();

        for (i, proof) in proofs.into_iter().enumerate() {
            // PrivateInput: data_leaf
            let data_leaf_num = AllocatedNum::alloc(cs.namespace(|| "data_leaf"), || {
                proof.data_leaf.ok_or_else(|| SynthesisError::AssignmentMissing)
            })?;
            //let (column_labels, comm_c_path, comm_r_last_path) = proof.synthesize(
                let (column_labels, comm_c_path) = proof.synthesize(
                &mut cs.namespace(|| format!("challenge_{}", i)),
                public_params.layer_challenges.layers(),
                &comm_d_num,
                &comm_c_num,
                &comm_r_last_num,
                &replica_id_bits,
                &data_leaf_num,
                0,
            );
            column_labels_vec.push(column_labels);
            comm_c_path_vec.push(comm_c_path);
            //comm_r_last_path_vec.push(comm_r_last_path);
            data_leaf_num_vec.push(data_leaf_num);
        }
        let mut gen_cs = cs.make_vector(len)?;

        column_labels_vec.into_iter()
        .zip(comm_c_path_vec.into_iter()
        //.zip(comm_r_last_path_vec.into_iter()
        .zip(gen_cs.iter_mut()
        .zip(data_leaf_num_vec.into_iter())))
        .for_each( |(column_labels, (comm_c_path, (other_cs, data_leaf_num)))| {
            let mut column_labels_copy = Vec::new();
            let mut i = 0;
            info!{"alloc labels and data_leaf_num"};
            for label in &column_labels {
                let label_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("label_{}_num", i)), 
                || { label.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                other_cs.align_variable(&mut label_copy.get_variable(), 0, i);
                column_labels_copy.push(label_copy);
                i = i+1;
            }
            /*let data_leaf_num_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("label_{}_num", i)), 
            || { data_leaf_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
            other_cs.align_variable(&mut data_leaf_num_copy.get_variable(), 0, i);

            let mut key = &column_labels_copy[column_labels_copy.len() - 1];
            info!("start : encode node");
            let mut encoded_node = encode(other_cs.namespace(|| "encode_node"), key, &data_leaf_num).unwrap();

            info!{"enforce encoded node"}
            enforce_inclusion(
                other_cs.namespace(|| "comm_r_last_data_inclusion"),
                comm_r_last_path,
                &comm_r_last_num,
                &encoded_node,
            ).unwrap();*/
        
            info!("start : enforce hash");
            let column_hash =
                hash_single_column(other_cs.namespace(|| "c_x_column_hash"), &column_labels_copy).unwrap();
            enforce_inclusion(
                other_cs.namespace(|| "c_x_inclusion"),
                comm_c_path,
                &comm_c_num,
                &column_hash,
            ).unwrap();
            for label in &column_labels_copy {
                other_cs.deallocate(label.get_variable()).unwrap();
            }
            //other_cs.deallocate(data_leaf_num_copy.get_variable()).unwrap();
        });
        cs.aggregate(gen_cs);
        Ok(())
    }*/

    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let StackedCircuit {
            public_params,
            proofs,
            replica_id,
            comm_r,
            comm_d,
            comm_r_last,
            comm_c,
            ..
        } = self;

        // Allocate replica_id
        let replica_id_num = AllocatedNum::alloc(cs.namespace(|| "replica_id"), || {
            replica_id
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make replica_id a public input
        replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;

        let replica_id_bits =
            reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

        // Allocate comm_d as Fr
        let comm_d_num = AllocatedNum::alloc(cs.namespace(|| "comm_d"), || {
            comm_d
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_d a public input
        comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

        // Allocate comm_r as Fr
        let comm_r_num = AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            comm_r
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_r a public input
        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // Allocate comm_r_last as Fr
        let comm_r_last_num = AllocatedNum::alloc(cs.namespace(|| "comm_r_last"), || {
            comm_r_last
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Allocate comm_c as Fr
        let comm_c_num = AllocatedNum::alloc(cs.namespace(|| "comm_c"), || {
            comm_c
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Verify comm_r = H(comm_c || comm_r_last)
        {
            let hash_num = <Tree::Hasher as Hasher>::Function::hash2_circuit(
                cs.namespace(|| "H_comm_c_comm_r_last"),
                &comm_c_num,
                &comm_r_last_num,
            )?;

            // Check actual equality
            constraint::equal(
                cs,
                || "enforce comm_r = H(comm_c || comm_r_last)",
                &comm_r_num,
                &hash_num,
            );
        }
        let layers = public_params.layer_challenges.layers();

        //let mut drg_parents_vec = Vec::new();
        //let mut exp_parents_vec = Vec::new();
        let proofs_copy = proofs.clone();
        let len = proofs.len();
        info!{"make a cs copy"}
        let mut gen_cs = cs.make_vector_copy(len)?;
        //let mut gen_cs = Vec::new();
        let mut unit = cs.make_copy()?;
        /*for i in 0.. len {
            let mut new = cs.clone();
            gen_cs.push(new);
        }*/
        let mut data_leaf_num_vec = Vec::new();
        let global_aux_len = cs.get_aux_assigment_len();
        let proof_alloc_aux = 7231629;
        let proof_alloc_input = 18;
        info!{"start several gen_cs"};
        for ((p, proof), other_gen_cs) in proofs.into_iter().enumerate()
        .zip(gen_cs.iter_mut()) {
                let Proof {
                    comm_d_path,
                    data_leaf,
                    challenge,
                    comm_r_last_path,
                    comm_c_path,
                    drg_parents_proofs,
                    exp_parents_proofs,
                    ..
                } = proof;

                assert!(!drg_parents_proofs.is_empty());
                assert!(!exp_parents_proofs.is_empty());

                // -- verify initial data layer

                // PrivateInput: data_leaf
                info!{"alloc data_leaf_num, p = {}", p};
                let data_leaf_num = AllocatedNum::alloc(other_gen_cs.namespace(|| "data_leaf"), || {
                    data_leaf.ok_or_else(|| SynthesisError::AssignmentMissing)
                })?;
                /*let aux_len1 = other_gen_cs.get_aux_assigment_len();
                other_gen_cs.align_variable(&mut data_leaf_num.get_variable(), 0, global_aux_len + aux_len1 + proof_alloc_aux*p);

                let comm_d_copy = AllocatedNum::alloc(other_gen_cs.namespace(|| format!("comm_c_{}_num", 0)), 
                || { comm_d_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                comm_d_copy.inputize(other_gen_cs.namespace(|| "comm_d_input"))?;
                let idx = other_gen_cs.get_index(&mut comm_d_num.get_variable());
                other_gen_cs.align_variable(&mut comm_d_copy.get_variable(), idx, 0);*/

                // enforce inclusion of the data leaf in the tree D
                info!{"enforce inclusion of the data leaf in the tree D, p = {}", p};
                enforce_inclusion(
                    other_gen_cs.namespace(|| "comm_d_inclusion"),
                    comm_d_path, // to do
                    &comm_d_num,
                    //&comm_d_copy,
                    &data_leaf_num,
                )?;

                //other_gen_cs.deallocate(comm_d_copy.get_variable()).unwrap();

                /*let alloc_count;
                if layers == 2 {
                    alloc_count = 4609;
                }
                else {
                    alloc_count = 5979;
                }

                let aux_len2 = other_gen_cs.get_aux_assigment_len();
                let initial_shift = aux_len1 - aux_len2; 

                // -- verify replica column openings

                // Private Inputs for the DRG parent nodes.
                let drg_len = drg_parents_proofs.len();
                let exp_len = exp_parents_proofs.len();
                let add_shift = proof_alloc_aux*p + global_aux_len;
                let parents_proofs = vec![drg_parents_proofs, exp_parents_proofs];
                let mut drg_parents = Vec::new();
                let mut exp_parents = Vec::new();
                let mut drg_cs = other_gen_cs.make_vector(drg_len)?;
                let mut exp_cs = other_gen_cs.make_vector(exp_len)?;
                let mut data_parents = vec![&mut drg_parents, &mut exp_parents];
                let mut all_cs = vec![drg_cs, exp_cs];
                let aux_len = other_gen_cs.get_aux_assigment_len();

                /*parents_proofs.into_par_iter().enumerate()
                .zip(data_parents.par_iter_mut()
                .zip(all_cs.par_iter_mut()))*/
                parents_proofs.into_iter().enumerate()
                .zip(data_parents.iter_mut()
                .zip(all_cs.iter_mut()))
                .for_each( |((k, proofs), (parents, css))| {
                    let len = proofs.len();           
                    for i in 0..len {
                        parents.push(AllocatedColumn::empty(layers));
                    }    

                    /*proofs.into_par_iter().enumerate()
                    .zip(css.par_iter_mut()
                    .zip(parents.par_iter_mut()))*/
                    proofs.into_iter().enumerate()
                    .zip(css.iter_mut()
                    .zip(parents.iter_mut()))
                    .for_each(|((i, parent), (other_cs, node_parent))| {
                        let (parent_col, inclusion_path) = 
                        parent.alloc(other_cs.namespace(|| format!("drg_parent_{}_num", i))).unwrap();
                        assert_eq!(layers, parent_col.len());

                        let val = parent_col.hash(other_cs.namespace(|| format!("drg_parent_{}_constraint", i))).unwrap();

                        *node_parent = parent_col;

                        for j in 1..node_parent.len() + 1 {
                            let mut row = node_parent.get_mut_value(j);
                            let mut v = row.get_mut_variable();
                            other_cs.align_variable(&mut v, 0, aux_len + j - 1 + (i + k*drg_len)*alloc_count + add_shift);
                            other_cs.print_index(&mut v);
                        }

                        let comm_c_copy = AllocatedNum::alloc(other_cs.namespace(|| format!("comm_c_{}_num", 0)), 
                        || { comm_c_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                        let idx = other_cs.get_index(&mut comm_c_num.get_variable());
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
                for new_cs in all_cs {
                    other_gen_cs.aggregate(new_cs);
                }
                drg_parents_vec.push(drg_parents);
                exp_parents_vec.push(exp_parents);
                */
                data_leaf_num_vec.push(data_leaf_num);
                
            }
            info!{"start serial"};
            for ((i, proof), (data_leaf_num, agg_cs)) in proofs_copy.into_iter().enumerate()
            //.zip(drg_parents_vec.into_iter()
            //.zip(exp_parents_vec.into_iter()
            .zip(data_leaf_num_vec.into_iter()
            .zip(gen_cs.into_iter())) {
                let Proof {
                    comm_d_path,
                    data_leaf,
                    challenge,
                    comm_r_last_path,
                    comm_c_path,
                    drg_parents_proofs,
                    exp_parents_proofs,
                    ..
                } = proof;
                info!{"start aggregate cs, i = {}", i};
                cs.part_aggregate_element(agg_cs, &unit);
                info!{"start drg\exp parents, i =  ", i};
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
                        || { comm_c_num.get_value().ok_or_else(|| SynthesisError::AssignmentMissing) }).unwrap();
                        let idx = other_cs.get_index(&mut comm_c_num.get_variable());
                        other_cs.align_variable(&mut comm_c_copy.get_variable(), 0, idx);
                        info!{"enforce inclusion of data_parent"};
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
                info!{"start labeling, i = {}", i};
                // stores the labels of the challenged column
                let mut column_labels = Vec::new();

                // PublicInput: challenge index
                let challenge_num = UInt64::alloc(cs.namespace(|| "challenge"), challenge)?;
                challenge_num.pack_into_input(cs.namespace(|| "challenge input"))?;

                for layer in 1..=layers {
                    let layer_num = UInt32::constant(layer as u32);

                    let mut cs = cs.namespace(|| format!("labeling_{}", layer));

                    // Collect the parents
                    let mut parents = Vec::new();

                    // all layers have drg parents
                    
                    for parent_col in &drg_parents {
                        let parent_val_num = parent_col.get_value(layer);
                        let parent_val_bits =
                            reverse_bit_numbering(parent_val_num.to_bits_le(
                                cs.namespace(|| format!("drg_parent_{}_bits", parents.len())),
                            )?);
                        parents.push(parent_val_bits);
                    }

                    // the first layer does not contain expander parents
                    if layer > 1 {
                        for parent_col in &exp_parents {
                            // subtract 1 from the layer index, as the exp parents, are shifted by one, as they
                            // do not store a value for the first layer
                            let parent_val_num = parent_col.get_value(layer - 1);
                            let parent_val_bits = reverse_bit_numbering(parent_val_num.to_bits_le(
                                cs.namespace(|| format!("exp_parent_{}_bits", parents.len())),
                            )?);
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

                    // Reconstruct the label
                    let label = create_label_circuit(
                        cs.namespace(|| "create_label"),
                        &replica_id_bits,
                        expanded_parents,
                        layer_num,
                        challenge_num.clone(),
                    )?;
                    column_labels.push(label);
                }

                // -- encoding node
                {
                    // encode the node

                    // key is the last label
                    let key = &column_labels[column_labels.len() - 1];
                    let encoded_node = encode(cs.namespace(|| "encode_node"), key, &data_leaf_num)?;
                    info!{"enforce inclusion of encoded node, i = {}", i};
                    // verify inclusion of the encoded node
                    enforce_inclusion(
                        cs.namespace(|| "comm_r_last_data_inclusion"),
                        comm_r_last_path,
                        &comm_r_last_num,
                        &encoded_node,
                    )?;
                }

                // -- ensure the column hash of the labels is included
                {
                    // calculate column_hash
                    let column_hash =
                        hash_single_column(cs.namespace(|| "c_x_column_hash"), &column_labels)?;

                    // enforce inclusion of the column hash in the tree C
                    info!{"enforce inclusion of column labels hash, i = {}", i}
                    enforce_inclusion(
                        cs.namespace(|| "c_x_inclusion"),
                        comm_c_path,
                        &comm_c_num,
                        &column_hash,
                    )?;
                }
            }
        Ok(())
    } 

    /* fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let StackedCircuit {
            public_params,
            proofs,
            replica_id,
            comm_r,
            comm_d,
            comm_r_last,
            comm_c,
            ..
        } = self;

        // Allocate replica_id
        let replica_id_num = AllocatedNum::alloc(cs.namespace(|| "replica_id"), || {
            replica_id
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make replica_id a public input
        replica_id_num.inputize(cs.namespace(|| "replica_id_input"))?;

        let replica_id_bits =
            reverse_bit_numbering(replica_id_num.to_bits_le(cs.namespace(|| "replica_id_bits"))?);

        // Allocate comm_d as Fr
        let comm_d_num = AllocatedNum::alloc(cs.namespace(|| "comm_d"), || {
            comm_d
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_d a public input
        comm_d_num.inputize(cs.namespace(|| "comm_d_input"))?;

        // Allocate comm_r as Fr
        let comm_r_num = AllocatedNum::alloc(cs.namespace(|| "comm_r"), || {
            comm_r
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // make comm_r a public input
        comm_r_num.inputize(cs.namespace(|| "comm_r_input"))?;

        // Allocate comm_r_last as Fr
        let comm_r_last_num = AllocatedNum::alloc(cs.namespace(|| "comm_r_last"), || {
            comm_r_last
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Allocate comm_c as Fr
        let comm_c_num = AllocatedNum::alloc(cs.namespace(|| "comm_c"), || {
            comm_c
                .map(Into::into)
                .ok_or_else(|| SynthesisError::AssignmentMissing)
        })?;

        // Verify comm_r = H(comm_c || comm_r_last)
        {
            let hash_num = <Tree::Hasher as Hasher>::Function::hash2_circuit(
                cs.namespace(|| "H_comm_c_comm_r_last"),
                &comm_c_num,
                &comm_r_last_num,
            )?;

            // Check actual equality
            constraint::equal(
                cs,
                || "enforce comm_r = H(comm_c || comm_r_last)",
                &comm_r_num,
                &hash_num,
            );
        }

        for (i, proof) in proofs.into_iter().enumerate() {
            proof.synthesize(
                &mut cs.namespace(|| format!("challenge_{}", i)),
                public_params.layer_challenges.layers(),
                &comm_d_num,
                &comm_c_num,
                &comm_r_last_num,
                &replica_id_bits,
                0,
            )?;
        }

        Ok(())
    }*/
}

#[allow(dead_code)]
pub struct StackedCompound<Tree: MerkleTreeTrait, G: Hasher> {
    partitions: Option<usize>,
    _t: PhantomData<Tree>,
    _g: PhantomData<G>,
}

impl<C: Circuit<Bls12>, P: ParameterSetMetadata, Tree: MerkleTreeTrait, G: Hasher>
CacheableParameters<C, P> for StackedCompound<Tree, G>
{
    fn cache_prefix() -> String {
        format!(
            "stacked-proof-of-replication-{}-{}",
            Tree::display(),
            G::name()
        )
    }
}

impl<'a, Tree: 'static + MerkleTreeTrait, G: 'static + Hasher>
CompoundProof<'a, StackedDrg<'a, Tree, G>, StackedCircuit<'a, Tree, G>>
for StackedCompound<Tree, G>
{
    fn generate_public_inputs(
        pub_in: &<StackedDrg<'_, Tree, G> as ProofScheme<'_>>::PublicInputs,
        pub_params: &<StackedDrg<'_, Tree, G> as ProofScheme<'_>>::PublicParams,
        k: Option<usize>,
    ) -> Result<Vec<Fr>> {
        let graph = &pub_params.graph;

        let mut inputs = Vec::new();

        let replica_id = pub_in.replica_id;
        inputs.push(replica_id.into());

        let comm_d = pub_in.tau.as_ref().expect("missing tau").comm_d;
        inputs.push(comm_d.into());

        let comm_r = pub_in.tau.as_ref().expect("missing tau").comm_r;
        inputs.push(comm_r.into());

        let por_setup_params = por::SetupParams {
            leaves: graph.size(),
            private: true,
        };

        let por_params = PoR::<Tree>::setup(&por_setup_params)?;
        let por_params_d = PoR::<BinaryMerkleTree<G>>::setup(&por_setup_params)?;

        let all_challenges = pub_in.challenges(&pub_params.layer_challenges, graph.size(), k);

        for challenge in all_challenges.into_iter() {
            // comm_d inclusion proof for the data leaf
            inputs.extend(generate_inclusion_inputs::<BinaryMerkleTree<G>>(
                &por_params_d,
                challenge,
                k,
            )?);

            // drg parents
            let mut drg_parents = vec![0; graph.base_graph().degree()];
            graph.base_graph().parents(challenge, &mut drg_parents)?;

            // Inclusion Proofs: drg parent node in comm_c
            for parent in drg_parents.into_iter() {
                inputs.extend(generate_inclusion_inputs::<Tree>(
                    &por_params,
                    parent as usize,
                    k,
                )?);
            }

            // exp parents
            let mut exp_parents = vec![0; graph.expansion_degree()];
            graph.expanded_parents(challenge, &mut exp_parents)?;

            // Inclusion Proofs: expander parent node in comm_c
            for parent in exp_parents.into_iter() {
                inputs.extend(generate_inclusion_inputs::<Tree>(
                    &por_params,
                    parent as usize,
                    k,
                )?);
            }

            inputs.push(u64_into_fr(challenge as u64));

            // Inclusion Proof: encoded node in comm_r_last
            inputs.extend(generate_inclusion_inputs::<Tree>(
                &por_params,
                challenge,
                k,
            )?);

            // Inclusion Proof: column hash of the challenged node in comm_c
            inputs.extend(generate_inclusion_inputs::<Tree>(
                &por_params,
                challenge,
                k,
            )?);
        }

        Ok(inputs)
    }

    fn circuit<'b>(
        public_inputs: &'b <StackedDrg<'_, Tree, G> as ProofScheme<'_>>::PublicInputs,
        _component_private_inputs: <StackedCircuit<'a, Tree, G> as CircuitComponent>::ComponentPrivateInputs,
        vanilla_proof: &'b <StackedDrg<'_, Tree, G> as ProofScheme<'_>>::Proof,
        public_params: &'b <StackedDrg<'_, Tree, G> as ProofScheme<'_>>::PublicParams,
        _partition_k: Option<usize>,
    ) -> Result<StackedCircuit<'a, Tree, G>> {
        ensure!(
            !vanilla_proof.is_empty(),
            "Cannot create a circuit with no vanilla proofs"
        );

        let comm_r_last = vanilla_proof[0].comm_r_last();
        let comm_c = vanilla_proof[0].comm_c();

        // ensure consistency
        ensure!(
            vanilla_proof.iter().all(|p| p.comm_r_last() == comm_r_last),
            "inconsistent comm_r_lasts"
        );
        ensure!(
            vanilla_proof.iter().all(|p| p.comm_c() == comm_c),
            "inconsistent comm_cs"
        );

        Ok(StackedCircuit {
            public_params: public_params.clone(),
            replica_id: Some(public_inputs.replica_id),
            comm_d: public_inputs.tau.as_ref().map(|t| t.comm_d),
            comm_r: public_inputs.tau.as_ref().map(|t| t.comm_r),
            comm_r_last: Some(comm_r_last),
            comm_c: Some(comm_c),
            proofs: vanilla_proof.iter().cloned().map(|p| p.into()).collect(),
        })
    }

    fn blank_circuit(
        public_params: &<StackedDrg<'_, Tree, G> as ProofScheme<'_>>::PublicParams,
    ) -> StackedCircuit<'a, Tree, G> {
        StackedCircuit {
            public_params: public_params.clone(),
            replica_id: None,
            comm_d: None,
            comm_r: None,
            comm_r_last: None,
            comm_c: None,
            proofs: (0..public_params.layer_challenges.challenges_count_all())
                .map(|_challenge_index| Proof::empty(public_params))
                .collect(),
        }
    }
}

/// Helper to generate public inputs for inclusion proofs.
fn generate_inclusion_inputs<Tree: 'static + MerkleTreeTrait>(
    por_params: &por::PublicParams,
    challenge: usize,
    k: Option<usize>,
) -> Result<Vec<Fr>> {
    let pub_inputs = por::PublicInputs::<<Tree::Hasher as Hasher>::Domain> {
        challenge,
        commitment: None,
    };

    PoRCompound::<Tree>::generate_public_inputs(&pub_inputs, por_params, k)
}
