use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use omeco::{EinCode, NestedEinsum};
use omeinsum::algebra::Standard;
use omeinsum::{Cpu, Einsum, Tensor};
use serde::Deserialize;

pub struct NetworkScenario {
    pub name: &'static str,
    pub path: &'static str,
}

pub struct PreparedNetworkCase {
    einsum: Einsum<usize>,
    tensors: Vec<Tensor<f32, Cpu>>,
}

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

const NETWORK_SCENARIOS: [NetworkScenario; 3] = [
    NetworkScenario {
        name: "small",
        path: "benches/network_small.json",
    },
    NetworkScenario {
        name: "medium",
        path: "benches/network_medium.json",
    },
    NetworkScenario {
        name: "large",
        path: "benches/network_large.json",
    },
];

pub fn network_scenarios() -> &'static [NetworkScenario] {
    &NETWORK_SCENARIOS
}

pub fn prepare_network_case(scenario: &NetworkScenario) -> PreparedNetworkCase {
    let file = File::open(scenario.path).expect("Failed to open network file");
    let reader = BufReader::new(file);
    let network: NetworkJson = serde_json::from_reader(reader).expect("Failed to parse network");

    let ixs: Vec<Vec<usize>> = network.edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let iy: Vec<usize> = vec![];

    let mut size_dict: HashMap<usize, usize> = HashMap::new();
    for &(u, v) in &network.edges {
        size_dict.insert(u, network.bond_dim);
        size_dict.insert(v, network.bond_dim);
    }

    let tree = json_to_nested(&network.tree);
    let mut einsum = Einsum::new(ixs, iy, size_dict);
    einsum.set_contraction_tree(tree);

    let fill_value: f32 = 0.5_f32.powf(0.4);
    let tensors = network
        .edges
        .iter()
        .map(|_| {
            let shape = vec![network.bond_dim, network.bond_dim];
            let size = network.bond_dim * network.bond_dim;
            Tensor::from_data(&vec![fill_value; size], &shape)
        })
        .collect();

    PreparedNetworkCase { einsum, tensors }
}

pub fn run_network_case(prepared: &PreparedNetworkCase) -> Tensor<f32, Cpu> {
    let tensor_refs: Vec<&Tensor<f32, Cpu>> = prepared.tensors.iter().collect();
    prepared
        .einsum
        .execute::<Standard<f32>, f32, Cpu>(&tensor_refs)
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
