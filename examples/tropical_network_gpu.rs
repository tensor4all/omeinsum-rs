//! GPU tropical network contraction runner (benchmark harness for issue #52).
//!
//! Contracts a whole tensor-network fixture through the `cuda-tropical` path
//! (max-plus on the GPU) and reports end-to-end wall-clock, so the device-
//! residency work (context/module caching, device-resident GEMM, device-side
//! permute) can be measured before/after on a real workload.
//!
//! Build & run on a CUDA host (e.g. hkustgz-hpc2):
//!
//! ```bash
//! cargo run --release --features cuda-tropical --example tropical_network_gpu -- \
//!     benches/network_medium.json --warmup 3 --repeats 10
//! ```
//!
//! Reuses the same JSON network fixtures as the CPU `network` bench
//! (`benches/network_{small,medium,large,3reg_150}.json`).

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use omeco::{EinCode, NestedEinsum};
use omeinsum::{Cuda, Einsum, MaxPlus, Tensor};
use serde::Deserialize;

#[derive(Deserialize)]
struct NetworkJson {
    #[serde(rename = "n_vertices")]
    _n_vertices: usize,
    bond_dim: usize,
    edges: Vec<(usize, usize)>,
    tree: TreeNodeJson,
}

#[derive(Deserialize)]
struct TreeNodeJson {
    #[serde(rename = "isleaf")]
    is_leaf: bool,
    tensorindex: Option<usize>,
    eins: Option<EinsJson>,
    args: Option<Vec<TreeNodeJson>>,
}

#[derive(Deserialize)]
struct EinsJson {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
}

fn json_to_nested(node: &TreeNodeJson) -> NestedEinsum<usize> {
    if node.is_leaf {
        NestedEinsum::leaf(node.tensorindex.expect("leaf node missing tensor index"))
    } else {
        let eins = node.eins.as_ref().expect("internal node missing eins");
        let args = node
            .args
            .as_ref()
            .expect("internal node missing args")
            .iter()
            .map(json_to_nested)
            .collect();
        NestedEinsum::node(args, EinCode::new(eins.ixs.clone(), eins.iy.clone()))
    }
}

fn main() {
    // --- args: <fixture.json> [--warmup N] [--repeats N] ---
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "benches/network_medium.json".to_string());
    let mut warmup = 3usize;
    let mut repeats = 10usize;
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag needs a value");
        match flag.as_str() {
            "--warmup" => warmup = val.parse().expect("--warmup expects an integer"),
            "--repeats" => repeats = val.parse().expect("--repeats expects an integer"),
            other => panic!("unknown flag {other}"),
        }
    }

    // --- load the network fixture ---
    let file = File::open(&path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let network: NetworkJson =
        serde_json::from_reader(BufReader::new(file)).expect("failed to parse network json");

    let ixs: Vec<Vec<usize>> = network.edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![];
    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &network.edges {
        size_dict.insert(u, network.bond_dim);
        size_dict.insert(v, network.bond_dim);
    }

    let mut einsum = Einsum::new(ixs, iy, size_dict);
    einsum.set_contraction_tree(json_to_nested(&network.tree));

    // --- build the GPU backend + device-resident input tensors (once) ---
    let cuda = match Cuda::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no CUDA device available: {e}");
            std::process::exit(1);
        }
    };

    // Same fill as the CPU bench so values are comparable.
    let fill: f32 = 0.5_f32.powf(0.4);
    let tensors: Vec<Tensor<f32, Cuda>> = network
        .edges
        .iter()
        .map(|_| {
            let shape = [network.bond_dim, network.bond_dim];
            let data = vec![fill; network.bond_dim * network.bond_dim];
            Tensor::<f32, Cuda>::from_data_with_backend(&data, &shape, cuda.clone())
        })
        .collect();
    let refs: Vec<&Tensor<f32, Cuda>> = tensors.iter().collect();

    let run_once = || {
        let result = einsum.execute::<MaxPlus<f32>, f32, Cuda>(&refs);
        // Force completion (scalar output → tiny download) so timing is honest.
        std::hint::black_box(result.to_vec());
    };

    // --- warmup (also triggers the one-time context/module build) ---
    for _ in 0..warmup {
        run_once();
    }

    // --- timed repeats ---
    let mut times_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        run_once();
        times_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }

    times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let median = times_ms[times_ms.len() / 2];
    let min = times_ms[0];
    let max = *times_ms.last().unwrap();

    println!(
        "network={path} nodes={} bond_dim={} warmup={warmup} repeats={repeats}",
        network.edges.len(),
        network.bond_dim
    );
    println!("wall_clock_ms mean={mean:.3} median={median:.3} min={min:.3} max={max:.3}");
}
