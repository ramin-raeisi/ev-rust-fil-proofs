use bellperson::bls::Bls12;
use bellperson::gadgets::boolean::{self, Boolean};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use rand::{thread_rng, Rng};

struct Blake2sExample<'a> {
    data: &'a [Option<bool>],
}

impl<'a> Circuit<Bls12> for Blake2sExample<'a> {
    fn synthesize<CS: ConstraintSystem<Bls12>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let data: Vec<Boolean> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, b)| {
                Ok(Boolean::from(boolean::AllocatedBit::alloc(
                    cs.namespace(|| format!("bit {}", i)),
                    *b,
                )?))
            })
            .collect::<Result<Vec<_>, SynthesisError>>()?;

        let cs = cs.namespace(|| "blake2s");
        let personalization = vec![0u8; 8];
        let _res = bellperson::gadgets::blake2s::blake2s(cs, &data, &personalization)?;
        Ok(())
    }
}

fn blake2s_benchmark(c: &mut Criterion) {
    let params = vec![32, 64, 10 * 32];

    c.bench(
        "hash-blake2s",
        ParameterizedBenchmark::new(
            "non-circuit",
            |b, bytes| {
                let mut rng = thread_rng();
                let data: Vec<u8> = (0..*bytes).map(|_| rng.gen()).collect();

                b.iter(|| black_box(blake2s_simd::blake2s(&data)))
            },
            params,
        ),
    );
}

criterion_group!(benches, blake2s_benchmark);
criterion_main!(benches);
