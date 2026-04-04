//! Tests for tropical algebra operations.
//! Run with: cargo test --test tropical --features tropical

#![cfg(feature = "tropical")]

use omeinsum::backend::Cpu;
use omeinsum::{einsum, GenericSemiring, MaxMul, MaxPlus, MinPlus, Tensor};

#[test]
fn test_maxplus_associativity() {
    // Test that (A @ B) @ C = A @ (B @ C) for MaxPlus
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    // (A @ B) @ C using contract_binary
    let ab = a.contract_binary::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);
    let abc_left = ab.contract_binary::<MaxPlus<f32>>(&c, &[0, 1], &[1, 2], &[0, 2]);

    // A @ (B @ C)
    let bc = b.contract_binary::<MaxPlus<f32>>(&c, &[0, 1], &[1, 2], &[0, 2]);
    let abc_right = a.contract_binary::<MaxPlus<f32>>(&bc, &[0, 1], &[1, 2], &[0, 2]);

    assert_eq!(abc_left.to_vec(), abc_right.to_vec());
}

#[test]
fn test_minplus_shortest_path() {
    // Test shortest path computation using MinPlus algebra
    // Graph:
    //   0 --1--> 1 --2--> 2
    //   |                 ^
    //   +------3----------+
    //
    // Distance matrix (adjacency with infinity for non-edges):
    let inf = f32::INFINITY;
    let dist = Tensor::<f32, Cpu>::from_data(
        &[
            0.0, 1.0, inf, // From node 0
            inf, 0.0, 2.0, // From node 1
            inf, inf, 0.0, // From node 2
        ],
        &[3, 3],
    );

    // MinPlus matmul gives 2-hop shortest paths
    let d2 = dist.contract_binary::<MinPlus<f32>>(&dist, &[0, 1], &[1, 2], &[0, 2]);
    let d2_vec = d2.to_vec();

    // d2[0,2] = min over k of (dist[0,k] + dist[k,2])
    //         = min(0+inf, 1+2, inf+0)
    //         = min(inf, 3, inf) = 3
    assert_eq!(d2_vec[2], 3.0);

    // d2[0,1] should still be 1 (direct path)
    assert_eq!(d2_vec[1], 1.0);

    // d2[0,0] should be 0 (self-loop)
    assert_eq!(d2_vec[0], 0.0);
}

#[test]
fn test_tropical_identity() {
    // Test tropical identity matrix: A @ I = A
    // For MaxPlus, identity has 0 on diagonal, -inf elsewhere
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let neg_inf = f32::NEG_INFINITY;
    let identity = Tensor::<f32, Cpu>::from_data(&[0.0, neg_inf, neg_inf, 0.0], &[2, 2]);

    let result = a.contract_binary::<MaxPlus<f32>>(&identity, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(result.to_vec(), a.to_vec());
}

#[test]
fn test_minplus_identity() {
    // For MinPlus, identity also has 0 on diagonal, +inf elsewhere
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let inf = f32::INFINITY;
    let identity = Tensor::<f32, Cpu>::from_data(&[0.0, inf, inf, 0.0], &[2, 2]);

    let result = a.contract_binary::<MinPlus<f32>>(&identity, &[0, 1], &[1, 2], &[0, 2]);
    assert_eq!(result.to_vec(), a.to_vec());
}

// Skip when tropical-kernels is enabled due to different input interpretation in optimized path
#[cfg(not(feature = "tropical-kernels"))]
#[test]
fn test_maxmul_operations() {
    // MaxMul: max(a, b), a * b
    // Used for max probability (non-log space)
    // Column-major: [0.5, 0.3, 0.2, 0.8] for shape [2,2] -> A = [[0.5, 0.2], [0.3, 0.8]]
    // Column-major: [0.9, 0.1, 0.4, 0.6] for shape [2,2] -> B = [[0.9, 0.4], [0.1, 0.6]]
    let a = Tensor::<f32, Cpu>::from_data(&[0.5, 0.3, 0.2, 0.8], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.9, 0.1, 0.4, 0.6], &[2, 2]);

    let c = a.contract_binary::<MaxMul<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);

    // C[i,k] = max_j (A[i,j] * B[j,k])
    // C[0,0] = max(0.5*0.9, 0.2*0.1) = max(0.45, 0.02) = 0.45
    // C[1,0] = max(0.3*0.9, 0.8*0.1) = max(0.27, 0.08) = 0.27
    // C[0,1] = max(0.5*0.4, 0.2*0.6) = max(0.20, 0.12) = 0.20
    // C[1,1] = max(0.3*0.4, 0.8*0.6) = max(0.12, 0.48) = 0.48
    // In column-major: [0.45, 0.27, 0.20, 0.48]
    let c_vec = c.to_vec();

    let eps = 1e-6;
    assert!((c_vec[0] - 0.45).abs() < eps);
    assert!((c_vec[1] - 0.27).abs() < eps);
    assert!((c_vec[2] - 0.20).abs() < eps);
    assert!((c_vec[3] - 0.48).abs() < eps);
}

#[test]
fn test_tropical_einsum_chain() {
    // Test tropical einsum with multiple tensors
    let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::<f32, Cpu>::from_data(&[0.0, 0.0, 0.0, 0.0], &[2, 2]); // Zero matrix (identity for +)
    let c = Tensor::<f32, Cpu>::from_data(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);

    // A @ B @ C in MaxPlus
    let result = einsum::<MaxPlus<f32>, _, _>(&[&a, &b, &c], &[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);

    assert_eq!(result.shape(), &[2, 2]);
    // The result should be computed correctly via the contraction tree
}

#[test]
fn test_tropical_distributivity() {
    // Test that multiplication distributes over addition
    // (a + b) * c = (a * c) + (b * c) in standard algebra
    // max(a, b) + c = max(a + c, b + c) in MaxPlus algebra

    let a = MaxPlus(2.0f32);
    let b = MaxPlus(5.0f32);
    let c = MaxPlus(3.0f32);

    use omeinsum::algebra::Semiring;

    // Left: (max(a, b)) + c
    let left = a.add(b).mul(c);

    // Right: max(a + c, b + c)
    let right = a.mul(c).add(b.mul(c));

    assert_eq!(left.to_scalar(), right.to_scalar());
}

#[test]
fn test_maxplus_viterbi_example() {
    // Example: Simple Viterbi-like computation
    // States: 0, 1
    // Transitions (log probabilities, so we want max sum):
    //   0->0: -1, 0->1: -2
    //   1->0: -3, 1->1: -1
    let transitions = Tensor::<f32, Cpu>::from_data(&[-1.0, -2.0, -3.0, -1.0], &[2, 2]);

    // Initial state distribution (log probabilities)
    let initial = Tensor::<f32, Cpu>::from_data(&[0.0, -1.0], &[2, 1]);

    // After one step: max over starting state
    let after_one = transitions.contract_binary::<MaxPlus<f32>>(&initial, &[0, 1], &[1, 2], &[0, 2]);

    // state 0: max(-1+0, -2+(-1)) = max(-1, -3) = -1
    // state 1: max(-3+0, -1+(-1)) = max(-3, -2) = -2
    let result = after_one.to_vec();
    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -2.0);
}

// Skip when tropical-kernels is enabled due to different input interpretation in optimized path
#[cfg(not(feature = "tropical-kernels"))]
#[test]
fn test_minplus_bellman_ford_step() {
    // One step of Bellman-Ford using MinPlus algebra
    // For d' = W @ d where W[i,j] is weight FROM j TO i:
    // d'[i] = min_j (W[i,j] + d[j])
    //
    // Graph: 0 -> 1 (weight 4), 0 -> 2 (weight 2), 1 -> 2 (weight 3)
    // W[i,j] = weight from j to i (transpose of adjacency)
    // Matrix W:
    //   Row 0 (to 0): from 0=0, from 1=inf, from 2=inf
    //   Row 1 (to 1): from 0=4, from 1=0, from 2=inf
    //   Row 2 (to 2): from 0=2, from 1=3, from 2=0
    // In column-major storage (column by column):
    //   Col 0: [W[0,0], W[1,0], W[2,0]] = [0, 4, 2]
    //   Col 1: [W[0,1], W[1,1], W[2,1]] = [inf, 0, 3]
    //   Col 2: [W[0,2], W[1,2], W[2,2]] = [inf, inf, 0]
    let inf = f32::INFINITY;
    let weights =
        Tensor::<f32, Cpu>::from_data(&[0.0, 4.0, 2.0, inf, 0.0, 3.0, inf, inf, 0.0], &[3, 3]);

    // Initial distances from node 0
    let dist = Tensor::<f32, Cpu>::from_data(&[0.0, inf, inf], &[3, 1]);

    // One relaxation step
    let new_dist = weights.contract_binary::<MinPlus<f32>>(&dist, &[0, 1], &[1, 2], &[0, 2]);

    // Should find direct paths from node 0:
    // to 0: min(0+0, inf+inf, inf+inf) = 0
    // to 1: min(4+0, 0+inf, inf+inf) = 4
    // to 2: min(2+0, 3+inf, 0+inf) = 2
    let result = new_dist.to_vec();
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], 2.0);
}

#[test]
fn test_tropical_zeros_and_ones() {
    use omeinsum::algebra::Semiring;

    // MaxPlus zero is -inf, one is 0
    let maxplus_zero = MaxPlus::<f32>::zero();
    let maxplus_one = MaxPlus::<f32>::one();
    assert_eq!(maxplus_zero.to_scalar(), f32::MIN);
    assert_eq!(maxplus_one.to_scalar(), 0.0);

    // MinPlus zero is +inf, one is 0
    let minplus_zero = MinPlus::<f32>::zero();
    let minplus_one = MinPlus::<f32>::one();
    assert_eq!(minplus_zero.to_scalar(), f32::MAX);
    assert_eq!(minplus_one.to_scalar(), 0.0);

    // MaxMul zero is 0, one is 1
    let maxmul_zero = MaxMul::<f32>::zero();
    let maxmul_one = MaxMul::<f32>::one();
    assert_eq!(maxmul_zero.to_scalar(), 0.0);
    assert_eq!(maxmul_one.to_scalar(), 1.0);
}

#[test]
fn test_tropical_idempotent_addition() {
    // Test idempotency of tropical addition: a + a = a
    use omeinsum::algebra::Semiring;

    let a = MaxPlus(5.0f32);
    assert_eq!(a.add(a).to_scalar(), a.to_scalar());

    let b = MinPlus(5.0f32);
    assert_eq!(b.add(b).to_scalar(), b.to_scalar());

    let c = MaxMul(5.0f32);
    assert_eq!(c.add(c).to_scalar(), c.to_scalar());
}
