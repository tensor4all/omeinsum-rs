use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use omeinsum::{Cpu, Standard, Tensor};

const MPO_BOND_DIM: usize = 9;
const CHI_VALUES: [usize; 3] = [32, 64, 128];

#[derive(Clone, Copy)]
struct TdvpContraction {
    name: &'static str,
    left_shape: fn(usize) -> Vec<usize>,
    right_shape: fn(usize) -> Vec<usize>,
    left_labels: &'static [usize],
    right_labels: &'static [usize],
    output_labels: &'static [usize],
    output_shape: fn(usize) -> Vec<usize>,
    scalar_fmas: fn(usize) -> u64,
}

struct PreparedContraction {
    left: Tensor<Complex64, Cpu>,
    right: Tensor<Complex64, Cpu>,
    left_labels: &'static [usize],
    right_labels: &'static [usize],
    output_labels: &'static [usize],
}

fn patterned_complex(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let real = ((index.wrapping_mul(17) + seed.wrapping_mul(31)) % 257) as f64;
            let imag = ((index.wrapping_mul(29) + seed.wrapping_mul(13)) % 251) as f64;
            Complex64::new((real - 128.0) / 37.0, (imag - 125.0) / 41.0)
        })
        .collect()
}

fn prepare(case: TdvpContraction, chi: usize) -> PreparedContraction {
    let left_shape = (case.left_shape)(chi);
    let right_shape = (case.right_shape)(chi);
    let left_len = left_shape.iter().product();
    let right_len = right_shape.iter().product();

    PreparedContraction {
        left: Tensor::from_data(&patterned_complex(left_len, 1), &left_shape),
        right: Tensor::from_data(&patterned_complex(right_len, 2), &right_shape),
        left_labels: case.left_labels,
        right_labels: case.right_labels,
        output_labels: case.output_labels,
    }
}

fn run(prepared: &PreparedContraction) -> Tensor<Complex64, Cpu> {
    prepared.left.contract_binary::<Standard<Complex64>>(
        &prepared.right,
        prepared.left_labels,
        prepared.right_labels,
        prepared.output_labels,
    )
}

fn h1_left_shape(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}

fn h1_wavefunction_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi]
}

fn h1_left_output_shape(chi: usize) -> Vec<usize> {
    vec![MPO_BOND_DIM, chi, 2, chi]
}

fn h1_right_intermediate_shape(chi: usize) -> Vec<usize> {
    vec![chi, chi, 2, MPO_BOND_DIM]
}

fn h1_right_output_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi]
}

fn right_environment_shape(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}

fn h2_wavefunction_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, 2, chi]
}

fn h2_left_output_shape(chi: usize) -> Vec<usize> {
    vec![MPO_BOND_DIM, chi, 2, 2, chi]
}

fn h2_right_intermediate_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi, 2, MPO_BOND_DIM]
}

fn h2_right_output_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, 2, chi]
}

const TDVP_CONTRACTIONS: [TdvpContraction; 4] = [
    TdvpContraction {
        name: "h1-left-environment",
        left_shape: h1_left_shape,
        right_shape: h1_wavefunction_shape,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4],
        output_labels: &[1, 2, 3, 4],
        output_shape: h1_left_output_shape,
        scalar_fmas: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h1-right-environment",
        left_shape: h1_right_intermediate_shape,
        right_shape: right_environment_shape,
        left_labels: &[2, 4, 5, 6],
        right_labels: &[4, 6, 7],
        output_labels: &[2, 5, 7],
        output_shape: h1_right_output_shape,
        scalar_fmas: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h2-left-environment",
        left_shape: h1_left_shape,
        right_shape: h2_wavefunction_shape,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4, 5],
        output_labels: &[1, 2, 3, 4, 5],
        output_shape: h2_left_output_shape,
        scalar_fmas: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h2-right-environment",
        left_shape: h2_right_intermediate_shape,
        right_shape: right_environment_shape,
        left_labels: &[2, 6, 5, 8, 9],
        right_labels: &[5, 9, 10],
        output_labels: &[2, 6, 8, 10],
        output_shape: h2_right_output_shape,
        scalar_fmas: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
];

fn bench_tdvp_complex_contractions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tdvp-complex-binary");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    for case in TDVP_CONTRACTIONS {
        for chi in CHI_VALUES {
            let prepared = prepare(case, chi);
            let actual_shape = run(&prepared).shape().to_vec();
            assert_eq!(
                actual_shape,
                (case.output_shape)(chi),
                "benchmark contraction produced the wrong shape"
            );

            group.throughput(Throughput::Elements((case.scalar_fmas)(chi)));
            group.bench_with_input(
                BenchmarkId::new(case.name, format!("chi{chi}-d{MPO_BOND_DIM}")),
                &prepared,
                |bencher, prepared| {
                    bencher.iter(|| {
                        let output = run(black_box(prepared));
                        black_box(output);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_tdvp_complex_contractions);
criterion_main!(benches);
