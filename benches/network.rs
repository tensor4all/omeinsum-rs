use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[path = "support/network.rs"]
mod network_support;

fn bench_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("network");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for scenario in network_support::network_scenarios() {
        let prepared = network_support::prepare_network_case(scenario);
        group.bench_with_input(
            BenchmarkId::new("einsum", scenario.name),
            &prepared,
            |b, prepared| {
                b.iter(|| network_support::run_network_case(black_box(prepared)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_network);
criterion_main!(benches);
