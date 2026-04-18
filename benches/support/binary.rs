use omeinsum::algebra::Standard;
use omeinsum::{einsum, Cpu, Tensor};

pub struct BinaryScenario {
    pub name: &'static str,
    pub rank_a: usize,
    pub rank_b: usize,
    pub num_contracted: usize,
    pub num_batch: usize,
}

impl BinaryScenario {
    pub const fn new(
        name: &'static str,
        rank_a: usize,
        rank_b: usize,
        num_contracted: usize,
        num_batch: usize,
    ) -> Self {
        Self {
            name,
            rank_a,
            rank_b,
            num_contracted,
            num_batch,
        }
    }
}

pub struct PreparedBinaryCase {
    tensor_a: Tensor<f32, Cpu>,
    tensor_b: Tensor<f32, Cpu>,
    ixs_a: Vec<usize>,
    ixs_b: Vec<usize>,
    ixs_c: Vec<usize>,
}

const BINARY_SCENARIOS: [BinaryScenario; 8] = [
    BinaryScenario::new("matmul_10x10", 10, 10, 5, 0),
    BinaryScenario::new("batched_matmul_8x8_batch_4", 8, 8, 4, 4),
    BinaryScenario::new("high_d_12x12_contract_6", 12, 12, 6, 0),
    BinaryScenario::new("high_d_15x15_contract_7", 15, 15, 7, 0),
    BinaryScenario::new("high_d_18x18_contract_8", 18, 18, 8, 0),
    BinaryScenario::new("high_d_20x20_contract_9", 20, 20, 9, 0),
    BinaryScenario::new("high_d_12x12_contract_4_batch_4", 12, 12, 4, 4),
    BinaryScenario::new("high_d_15x15_contract_5_batch_5", 15, 15, 5, 5),
];

pub fn binary_scenarios() -> &'static [BinaryScenario] {
    &BINARY_SCENARIOS
}

pub fn prepare_binary_case(scenario: &BinaryScenario) -> PreparedBinaryCase {
    let num_left_a = scenario.rank_a - scenario.num_contracted - scenario.num_batch;
    let num_right_b = scenario.rank_b - scenario.num_contracted - scenario.num_batch;

    let mut next_idx = 0usize;
    let left_indices: Vec<usize> = (next_idx..next_idx + num_left_a).collect();
    next_idx += num_left_a;

    let contracted_indices: Vec<usize> = (next_idx..next_idx + scenario.num_contracted).collect();
    next_idx += scenario.num_contracted;

    let right_indices: Vec<usize> = (next_idx..next_idx + num_right_b).collect();
    next_idx += num_right_b;

    let batch_indices: Vec<usize> = (next_idx..next_idx + scenario.num_batch).collect();

    let ixs_a: Vec<usize> = left_indices
        .iter()
        .chain(contracted_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    let ixs_b: Vec<usize> = contracted_indices
        .iter()
        .chain(right_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    let ixs_c: Vec<usize> = left_indices
        .iter()
        .chain(right_indices.iter())
        .chain(batch_indices.iter())
        .copied()
        .collect();

    let shape_a: Vec<usize> = vec![2; scenario.rank_a];
    let shape_b: Vec<usize> = vec![2; scenario.rank_b];

    let numel_a: usize = shape_a.iter().product();
    let numel_b: usize = shape_b.iter().product();

    let data_a: Vec<f32> = (0..numel_a).map(|i| i as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..numel_b).map(|i| i as f32 * 0.001).collect();

    PreparedBinaryCase {
        tensor_a: Tensor::<f32, Cpu>::from_data(&data_a, &shape_a),
        tensor_b: Tensor::<f32, Cpu>::from_data(&data_b, &shape_b),
        ixs_a,
        ixs_b,
        ixs_c,
    }
}

pub fn run_binary_case(prepared: &PreparedBinaryCase) -> Tensor<f32, Cpu> {
    einsum::<Standard<f32>, _, _>(
        &[&prepared.tensor_a, &prepared.tensor_b],
        &[prepared.ixs_a.as_slice(), prepared.ixs_b.as_slice()],
        &prepared.ixs_c,
    )
}
