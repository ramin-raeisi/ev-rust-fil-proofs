use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use bellperson::bls::Bls12;
use bellperson::gadgets::boolean::{self, Boolean};
use criterion::{black_box, Criterion, criterion_group, criterion_main, ParameterizedBenchmark};
use rand::{Rng, thread_rng};

use storage_proofs_core::crypto::xor;
use storage_proofs_core::gadgets;

struct XorExample<'a> {
    key: &'a [Option<bool>],
    data: &'a [Option<bool>],
}

impl<'a> Circuit<Bls12> for XorExample<'a> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let key: Vec<Boolean> = self
            .key
            .iter()
            .enumerate()
            .map(|(i, b)| {
                Ok(Boolean::from(boolean::AllocatedBit::alloc(
                    cs.namespace(|| format!("key_bit {}", i)),
                    *b,
                )?))
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;
        let data: Vec<Boolean> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, b)| {
                Ok(Boolean::from(boolean::AllocatedBit::alloc(
                    cs.namespace(|| format!("data_bit {}", i)),
                    *b,
                )?))
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        let mut cs = cs.namespace(|| "xor");
        let _res = gadgets::xor::xor(&mut cs, &key, &data)?;

        Ok(())
    }
}

fn xor_benchmark(c: &mut Criterion) {
    let params = vec![32, 64, 10 * 32];

    c.bench(
        "xor",
        ParameterizedBenchmark::new(
            "non-circuit",
            |b, bytes| {
                let mut rng = thread_rng();
                let key: Vec<u8> = (0..32).map(|_| rng.gen()).collect();
                let data: Vec<u8> = (0..*bytes).map(|_| rng.gen()).collect();

                b.iter(|| black_box(xor::encode(&key, &data)))
            },
            params,
        ),
    );
}

criterion_group!(benches, xor_benchmark);
criterion_main!(benches);
