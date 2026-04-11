//! Profiling benchmark for tensor network contraction
//!
//! Generate network: cargo bench --bench profile_network -- --generate
//! Run benchmark:    cargo bench --bench profile_network
//! Profile:          samply record cargo bench --bench profile_network

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use omeco::{contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, TreeSA};
use omeinsum::algebra::Standard;
use omeinsum::{Cpu, Einsum, Tensor};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

const NETWORK_FILE: &str = "benches/network_3reg_150.json";

/// Generate a random 3-regular graph using configuration model
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

// JSON serialization structures
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
        NestedEinsum::leaf(node.tensorindex.unwrap())
    } else {
        let eins = node.eins.as_ref().unwrap();
        let args: Vec<_> = node
            .args
            .as_ref()
            .unwrap()
            .iter()
            .map(json_to_nested)
            .collect();
        NestedEinsum::node(args, EinCode::new(eins.ixs.clone(), eins.iy.clone()))
    }
}

/// Generate and save optimized network using TreeSA
fn generate_network(n_vertices: usize, bond_dim: usize, output: &Path) {
    println!("Generating 3-regular graph with {} vertices...", n_vertices);
    let edges = generate_3_regular_graph(n_vertices);
    println!("  {} edges (tensors)", edges.len());

    // Build einsum code
    let ixs: Vec<Vec<usize>> = edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![]; // Contract to scalar
    let code = EinCode::new(ixs.clone(), iy);

    // Build size dictionary
    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &edges {
        size_dict.insert(u, bond_dim);
        size_dict.insert(v, bond_dim);
    }

    // Try both optimizers and use the one with lower sc
    println!("Optimizing with GreedyMethod...");
    let greedy = GreedyMethod::new(0.0, 0.0);
    let greedy_tree =
        optimize_code(&code, &size_dict, &greedy).expect("Failed to optimize contraction order");
    let greedy_cc = contraction_complexity(&greedy_tree, &size_dict, &ixs);
    println!("  Greedy: tc={:.2}, sc={:.2}", greedy_cc.tc, greedy_cc.sc);

    println!("Optimizing with TreeSA (ntrials=1, niters=100)...");
    // Match omeco benchmark parameters - no sc_target, use niters
    let treesa = TreeSA::default().with_ntrials(1).with_niters(100);
    let treesa_tree =
        optimize_code(&code, &size_dict, &treesa).expect("Failed to optimize contraction order");
    let treesa_cc = contraction_complexity(&treesa_tree, &size_dict, &ixs);
    println!("  TreeSA: tc={:.2}, sc={:.2}", treesa_cc.tc, treesa_cc.sc);

    // Use the one with lower space complexity
    let (tree, cc) = if treesa_cc.sc <= greedy_cc.sc {
        println!("Using TreeSA result (lower sc)");
        (treesa_tree, treesa_cc)
    } else {
        println!("Using Greedy result (lower sc)");
        (greedy_tree, greedy_cc)
    };
    println!("  Final: tc={:.2}, sc={:.2}", cc.tc, cc.sc);

    // Save to JSON
    let network = NetworkJson {
        n_vertices,
        bond_dim,
        edges,
        tree: nested_to_json(&tree),
    };

    let file = File::create(output).expect("Failed to create output file");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &network).expect("Failed to write JSON");

    println!("Saved to {:?}", output);
}

/// Load network from JSON and run benchmark
fn run_benchmark(input: &Path, n_iterations: usize) {
    println!("Loading network from {:?}...", input);
    let file = File::open(input).expect("Failed to open network file");
    let reader = BufReader::new(file);
    let network: NetworkJson = serde_json::from_reader(reader).expect("Failed to parse JSON");

    println!(
        "  {} vertices, {} edges, bond_dim={}",
        network.n_vertices,
        network.edges.len(),
        network.bond_dim
    );

    // Build Einsum with pre-optimized tree
    let ixs: Vec<Vec<usize>> = network.edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![];

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &network.edges {
        size_dict.insert(u, network.bond_dim);
        size_dict.insert(v, network.bond_dim);
    }

    let tree = json_to_nested(&network.tree);
    let mut einsum = Einsum::new(ixs, iy, size_dict);
    einsum.set_contraction_tree(tree.clone());

    // Create tensors
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

    // Debug: Check tree structure and max intermediate size
    println!("Checking contraction tree...");
    fn tree_depth(tree: &NestedEinsum<usize>) -> usize {
        match tree {
            NestedEinsum::Leaf { .. } => 1,
            NestedEinsum::Node { args, .. } => 1 + args.iter().map(tree_depth).max().unwrap_or(0),
        }
    }
    fn tree_size(tree: &NestedEinsum<usize>) -> usize {
        match tree {
            NestedEinsum::Leaf { .. } => 1,
            NestedEinsum::Node { args, .. } => 1 + args.iter().map(tree_size).sum::<usize>(),
        }
    }
    fn max_intermediate_indices(tree: &NestedEinsum<usize>) -> usize {
        match tree {
            NestedEinsum::Leaf { .. } => 0,
            NestedEinsum::Node { args, eins } => {
                let this_size = eins.iy.len();
                let child_max = args.iter().map(max_intermediate_indices).max().unwrap_or(0);
                this_size.max(child_max)
            }
        }
    }
    println!(
        "  Depth: {}, Nodes: {}, Max intermediate indices: {}",
        tree_depth(&tree),
        tree_size(&tree),
        max_intermediate_indices(&tree)
    );

    // Warmup with timing
    println!("Warming up (1 iteration)...");
    let warmup_start = std::time::Instant::now();
    for _ in 0..1 {
        let _ = einsum.execute::<Standard<f32>, f32, Cpu>(&tensor_refs);
    }
    println!(
        "  Warmup took: {:.3}s",
        warmup_start.elapsed().as_secs_f64()
    );

    // Benchmark
    println!("Running {} iterations...", n_iterations);
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        let _ = einsum.execute::<Standard<f32>, f32, Cpu>(&tensor_refs);
    }
    let elapsed = start.elapsed();

    let result = einsum.execute::<Standard<f32>, f32, Cpu>(&tensor_refs);
    let result_val = result.to_vec()[0];

    println!("Result: {}", result_val);

    println!(
        "Total: {:.3}s, Per iteration: {:.3}ms",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / n_iterations as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let network_path = Path::new(NETWORK_FILE);

    if args.iter().any(|a| a == "--generate") {
        generate_network(150, 2, network_path);
    } else if args.iter().any(|a| a == "--small") {
        // Quick test with tiny graph
        println!("Running small test (10 vertices)...");
        let small_path = Path::new("benches/network_small.json");
        generate_network(10, 2, small_path);
        run_benchmark(small_path, 1);
    } else if args.iter().any(|a| a == "--medium") {
        // Medium test
        println!("Running medium test (50 vertices)...");
        let med_path = Path::new("benches/network_medium.json");
        generate_network(50, 2, med_path);
        run_benchmark(med_path, 1);
    } else if args.iter().any(|a| a == "--large") {
        // Large test
        println!("Running large test (100 vertices)...");
        let large_path = Path::new("benches/network_large.json");
        generate_network(100, 2, large_path);
        run_benchmark(large_path, 1);
    } else if network_path.exists() {
        run_benchmark(network_path, 1);
    } else {
        eprintln!("Network file not found: {:?}", network_path);
        eprintln!("Run with --generate first to create it");
        std::process::exit(1);
    }
}
