use std::collections::HashMap;
use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

use omeco::{EinCode, NestedEinsum};
use omeinsum::backend::Backend;
use omeinsum::{Cpu, Einsum, Standard, Tensor};

#[derive(Clone, Copy)]
enum Scenario {
    All,
    RhsTransposeView,
    BatchMajorBatched,
    RootOutputPermutation,
}

impl Scenario {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "rhs-transpose-view" => Ok(Self::RhsTransposeView),
            "batch-major-batched" => Ok(Self::BatchMajorBatched),
            "root-output-permutation" => Ok(Self::RootOutputPermutation),
            _ => Err(format!(
                "unknown scenario `{value}`; expected one of all, rhs-transpose-view, \
                 batch-major-batched, root-output-permutation"
            )),
        }
    }
}

#[derive(Clone, Copy)]
struct Config {
    scenario: Scenario,
    iterations: usize,
    warmup: usize,
    dim: usize,
    batch: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            scenario: Scenario::All,
            iterations: 40,
            warmup: 5,
            dim: 128,
            batch: 24,
        }
    }
}

fn usage() -> &'static str {
    "Usage: cargo run --release --example cpu_contract_bench -- [options]

Options:
  --scenario <all|rhs-transpose-view|batch-major-batched|root-output-permutation>
  --iterations <count>
  --warmup <count>
  --dim <matrix-dimension>
  --batch <batch-size>
  --help"
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scenario" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--scenario requires a value".to_string())?;
                config.scenario = Scenario::parse(&value)?;
            }
            "--iterations" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--iterations requires a value".to_string())?;
                config.iterations = value
                    .parse()
                    .map_err(|_| format!("invalid iteration count `{value}`"))?;
            }
            "--warmup" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--warmup requires a value".to_string())?;
                config.warmup = value
                    .parse()
                    .map_err(|_| format!("invalid warmup count `{value}`"))?;
            }
            "--dim" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--dim requires a value".to_string())?;
                config.dim = value
                    .parse()
                    .map_err(|_| format!("invalid dimension `{value}`"))?;
            }
            "--batch" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--batch requires a value".to_string())?;
                config.batch = value
                    .parse()
                    .map_err(|_| format!("invalid batch size `{value}`"))?;
            }
            "--help" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    if config.iterations == 0 {
        return Err("--iterations must be greater than 0".to_string());
    }
    if config.dim == 0 {
        return Err("--dim must be greater than 0".to_string());
    }
    if config.batch == 0 {
        return Err("--batch must be greater than 0".to_string());
    }

    Ok(config)
}

fn seeded_data(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let value = ((i.wrapping_mul(17)).wrapping_add(seed.wrapping_mul(31))) % 257;
            (value as f32 - 128.0) / 37.0
        })
        .collect()
}

fn transpose_column_major(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for col in 0..cols {
        for row in 0..rows {
            let src = row + col * rows;
            let dst = col + row * cols;
            out[dst] = data[src];
        }
    }
    out
}

fn run_benchmark<F>(name: &str, config: Config, mut f: F)
where
    F: FnMut() -> f32,
{
    for _ in 0..config.warmup {
        black_box(f());
    }

    let start = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..config.iterations {
        checksum += f() as f64;
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1_000.0 / config.iterations as f64;

    println!(
        "{name}: iterations={} warmup={} dim={} batch={} total_ms={:.3} avg_ms={:.3} checksum={:.6}",
        config.iterations,
        config.warmup,
        config.dim,
        config.batch,
        elapsed.as_secs_f64() * 1_000.0,
        avg_ms,
        checksum,
    );
}

fn bench_rhs_transpose_view(config: Config) {
    let cpu = Cpu;
    let a = seeded_data(config.dim * config.dim, 1);
    let b = seeded_data(config.dim * config.dim, 2);
    let b_transposed = transpose_column_major(&b, config.dim, config.dim);

    let strided = cpu.contract::<Standard<f32>>(
        &a,
        &[config.dim, config.dim],
        &[1, config.dim],
        &[0, 1],
        &b,
        &[config.dim, config.dim],
        &[config.dim, 1],
        &[1, 2],
        &[config.dim, config.dim],
        &[0, 2],
    );
    let contiguous = cpu.contract::<Standard<f32>>(
        &a,
        &[config.dim, config.dim],
        &[1, config.dim],
        &[0, 1],
        &b_transposed,
        &[config.dim, config.dim],
        &[1, config.dim],
        &[1, 2],
        &[config.dim, config.dim],
        &[0, 2],
    );
    assert_eq!(
        strided, contiguous,
        "rhs transpose view benchmark sanity check failed"
    );

    run_benchmark("rhs-transpose-view", config, || {
        let result = cpu.contract::<Standard<f32>>(
            &a,
            &[config.dim, config.dim],
            &[1, config.dim],
            &[0, 1],
            &b,
            &[config.dim, config.dim],
            &[config.dim, 1],
            &[1, 2],
            &[config.dim, config.dim],
            &[0, 2],
        );
        black_box(result[0]) + black_box(*result.last().expect("result must be non-empty"))
    });
}

fn bench_batch_major_batched(config: Config) {
    let cpu = Cpu;
    let numel = config.batch * config.dim * config.dim;
    let a = seeded_data(numel, 3);
    let b = seeded_data(numel, 4);
    let strides = [1, config.batch, config.batch * config.dim];

    run_benchmark("batch-major-batched", config, || {
        let result = cpu.contract::<Standard<f32>>(
            &a,
            &[config.batch, config.dim, config.dim],
            &strides,
            &[0, 1, 2],
            &b,
            &[config.batch, config.dim, config.dim],
            &strides,
            &[0, 2, 3],
            &[config.batch, config.dim, config.dim],
            &[0, 1, 3],
        );
        black_box(result[0]) + black_box(*result.last().expect("result must be non-empty"))
    });
}

fn root_output_tree() -> NestedEinsum<usize> {
    NestedEinsum::node(
        vec![
            NestedEinsum::node(
                vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
                EinCode::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2]),
            ),
            NestedEinsum::leaf(2),
        ],
        EinCode::new(vec![vec![0, 2], vec![2, 3]], vec![3, 0]),
    )
}

fn bench_root_output_permutation(config: Config) {
    let a = Tensor::<f32, Cpu>::from_data(
        &seeded_data(config.dim * config.dim, 5),
        &[config.dim, config.dim],
    );
    let b = Tensor::<f32, Cpu>::from_data(
        &seeded_data(config.dim * config.dim, 6),
        &[config.dim, config.dim],
    );
    let c = Tensor::<f32, Cpu>::from_data(
        &seeded_data(config.dim * config.dim, 7),
        &[config.dim, config.dim],
    );
    let sizes: HashMap<usize, usize> = [
        (0, config.dim),
        (1, config.dim),
        (2, config.dim),
        (3, config.dim),
    ]
    .into();

    let pairwise = Einsum::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        vec![0, 3],
        sizes.clone(),
    );
    let mut rooted = Einsum::new(
        vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        vec![0, 3],
        sizes.clone(),
    );
    rooted.set_contraction_tree(root_output_tree());

    let reference = pairwise.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);
    let optimized = rooted.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);
    assert_eq!(
        optimized.to_vec(),
        reference.to_vec(),
        "root output permutation benchmark sanity check failed"
    );

    run_benchmark("root-output-permutation", config, || {
        let result = rooted.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);
        let values = result.to_vec();
        black_box(values[0]) + black_box(*values.last().expect("result must be non-empty"))
    });
}

fn main() {
    let config = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}\n\n{}", usage());
        process::exit(2);
    });

    match config.scenario {
        Scenario::All => {
            bench_rhs_transpose_view(config);
            bench_batch_major_batched(config);
            bench_root_output_permutation(config);
        }
        Scenario::RhsTransposeView => bench_rhs_transpose_view(config),
        Scenario::BatchMajorBatched => bench_batch_major_batched(config),
        Scenario::RootOutputPermutation => bench_root_output_permutation(config),
    }
}
