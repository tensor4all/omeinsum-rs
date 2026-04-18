//! Profiling and one-shot timing harness for fixed tensor-network contractions.
//!
//! Generate network JSON:
//!   cargo run --release --example profile_network -- --generate --scenario small
//! One-shot timing:
//!   cargo run --release --example profile_network -- --scenario 3reg_150 --output benchmarks/data/rust_network_timings.json

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use omeinsum::algebra::Standard;
use omeinsum::{Cpu, Einsum, Tensor};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct NetworkJson {
    n_vertices: usize,
    bond_dim: usize,
    edges: Vec<(usize, usize)>,
    tree: TreeNodeJson,
}

#[derive(Serialize, Deserialize)]
struct TreeNodeJson {
    #[serde(rename = "isleaf")]
    is_leaf: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensorindex: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eins: Option<EinsJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Vec<TreeNodeJson>>,
}

#[derive(Serialize, Deserialize)]
struct EinsJson {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
}

struct NetworkScenario {
    name: &'static str,
    path: &'static str,
    n_vertices: usize,
}

const SCENARIOS: [NetworkScenario; 4] = [
    NetworkScenario {
        name: "small",
        path: "benches/network_small.json",
        n_vertices: 10,
    },
    NetworkScenario {
        name: "medium",
        path: "benches/network_medium.json",
        n_vertices: 50,
    },
    NetworkScenario {
        name: "large",
        path: "benches/network_large.json",
        n_vertices: 100,
    },
    NetworkScenario {
        name: "3reg_150",
        path: "benches/network_3reg_150.json",
        n_vertices: 150,
    },
];

fn generate_3_regular_graph(n: usize) -> Vec<(usize, usize)> {
    assert!(n.is_multiple_of(2) && n >= 4, "n must be even and >= 4");

    let mut rng = rand::rng();
    let mut stubs: Vec<usize> = (0..n).flat_map(|v| vec![v; 3]).collect();

    for _ in 0..1000 {
        stubs.shuffle(&mut rng);
        let mut edges = HashSet::new();
        let mut valid = true;

        for chunk in stubs.chunks(2) {
            let (u, v) = (chunk[0], chunk[1]);
            if u == v {
                valid = false;
                break;
            }
            let edge = if u < v { (u, v) } else { (v, u) };
            if edges.contains(&edge) {
                valid = false;
                break;
            }
            edges.insert(edge);
        }

        if valid {
            return edges.into_iter().collect();
        }
    }

    panic!("Failed to generate 3-regular graph");
}

fn nested_to_json(tree: &NestedEinsum<usize>) -> TreeNodeJson {
    match tree {
        NestedEinsum::Leaf { tensor_index } => TreeNodeJson {
            is_leaf: true,
            tensorindex: Some(*tensor_index),
            eins: None,
            args: None,
        },
        NestedEinsum::Node { args, eins } => TreeNodeJson {
            is_leaf: false,
            tensorindex: None,
            eins: Some(EinsJson {
                ixs: eins.ixs.clone(),
                iy: eins.iy.clone(),
            }),
            args: Some(args.iter().map(nested_to_json).collect()),
        },
    }
}

fn json_to_nested(node: &TreeNodeJson) -> NestedEinsum<usize> {
    if node.is_leaf {
        NestedEinsum::leaf(node.tensorindex.expect("Leaf node missing tensor index"))
    } else {
        let eins = node.eins.as_ref().expect("Node missing eins");
        let args = node
            .args
            .as_ref()
            .expect("Node missing args")
            .iter()
            .map(json_to_nested)
            .collect();
        NestedEinsum::node(args, EinCode::new(eins.ixs.clone(), eins.iy.clone()))
    }
}

fn generate_network(scenario: &NetworkScenario, bond_dim: usize) {
    println!(
        "Generating 3-regular graph for scenario {} with {} vertices...",
        scenario.name, scenario.n_vertices
    );
    let edges = generate_3_regular_graph(scenario.n_vertices);

    let ixs: Vec<Vec<usize>> = edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![];
    let code = EinCode::new(ixs.clone(), iy);

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &edges {
        size_dict.insert(u, bond_dim);
        size_dict.insert(v, bond_dim);
    }

    let greedy = GreedyMethod::new(0.0, 0.0);
    let greedy_tree =
        optimize_code(&code, &size_dict, &greedy).expect("Failed to optimize contraction order");
    let greedy_cc = contraction_complexity(&greedy_tree, &size_dict, &ixs);

    let treesa = TreeSA::default().with_ntrials(1).with_niters(100);
    let treesa_tree =
        optimize_code(&code, &size_dict, &treesa).expect("Failed to optimize contraction order");
    let treesa_cc = contraction_complexity(&treesa_tree, &size_dict, &ixs);

    let tree = if treesa_cc.sc <= greedy_cc.sc {
        treesa_tree
    } else {
        greedy_tree
    };

    let network = NetworkJson {
        n_vertices: scenario.n_vertices,
        bond_dim,
        edges,
        tree: nested_to_json(&tree),
    };

    let file = File::create(scenario.path).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &network).expect("Failed to write JSON");
}

fn run_benchmark(scenario: &NetworkScenario, n_iterations: usize) -> u128 {
    let file = File::open(scenario.path).expect("Failed to open network file");
    let reader = BufReader::new(file);
    let network: NetworkJson = serde_json::from_reader(reader).expect("Failed to parse JSON");

    let ixs: Vec<Vec<usize>> = network.edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![];

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &network.edges {
        size_dict.insert(u, network.bond_dim);
        size_dict.insert(v, network.bond_dim);
    }

    let mut einsum = Einsum::new(ixs, iy, size_dict);
    einsum.set_contraction_tree(json_to_nested(&network.tree));

    let fill_value: f32 = 0.5_f32.powf(0.4);
    let tensors: Vec<Tensor<f32, Cpu>> = network
        .edges
        .iter()
        .map(|_| {
            let shape = vec![network.bond_dim, network.bond_dim];
            let size = network.bond_dim * network.bond_dim;
            Tensor::from_data(&vec![fill_value; size], &shape)
        })
        .collect();
    let tensor_refs: Vec<&Tensor<f32, Cpu>> = tensors.iter().collect();

    let start = Instant::now();
    for _ in 0..n_iterations {
        let _ = einsum.execute::<Standard<f32>, f32, Cpu>(&tensor_refs);
    }
    start.elapsed().as_nanos() / n_iterations as u128
}

fn scenario_by_name(name: &str) -> &'static NetworkScenario {
    SCENARIOS
        .iter()
        .find(|scenario| scenario.name == name)
        .unwrap_or_else(|| panic!("Unknown scenario: {name}"))
}

fn maybe_write_output(output: Option<PathBuf>, scenario: &str, ns: u128) {
    if let Some(path) = output {
        let mut data = HashMap::new();
        data.insert(scenario.to_string(), ns);
        let file = File::create(path).expect("Failed to create output file");
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &data).expect("Failed to write timing JSON");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut scenario_name = "3reg_150";
    let mut should_generate = false;
    let mut output: Option<PathBuf> = None;
    let mut iterations = 1usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--generate" => should_generate = true,
            "--scenario" => {
                scenario_name = args.get(i + 1).expect("--scenario requires a value");
                i += 1;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).expect("--output requires a value"),
                ));
                i += 1;
            }
            "--iterations" => {
                iterations = args
                    .get(i + 1)
                    .expect("--iterations requires a value")
                    .parse()
                    .expect("Invalid iteration count");
                i += 1;
            }
            other => panic!("Unknown argument: {other}"),
        }
        i += 1;
    }

    let scenario = scenario_by_name(scenario_name);

    if should_generate {
        generate_network(scenario, 2);
        return;
    }

    if !Path::new(scenario.path).exists() {
        panic!(
            "Network file {} is missing. Run with --generate --scenario {} first.",
            scenario.path, scenario.name
        );
    }

    let ns = run_benchmark(scenario, iterations);
    println!("{}: {} ns", scenario.name, ns);
    maybe_write_output(output, scenario.name, ns);
}
