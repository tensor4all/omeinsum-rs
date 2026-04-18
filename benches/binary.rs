use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[path = "support/binary.rs"]
mod binary_support;

fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary");

    for scenario in binary_support::binary_scenarios() {
        let prepared = binary_support::prepare_binary_case(scenario);
        group.bench_with_input(
            BenchmarkId::new("einsum", scenario.name),
            &prepared,
            |b, prepared| {
                b.iter(|| binary_support::run_binary_case(black_box(prepared)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_binary);
criterion_main!(benches);
