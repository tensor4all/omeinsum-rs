//! Einsum execution engine with omeco integration.

use std::collections::{HashMap, HashSet};

use omeco::{optimize_code, EinCode, GreedyMethod, Label, NestedEinsum, TreeSA};

use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar};
use crate::tensor::{BinaryContractOptions, DenseTensor, Tensor};

/// Einsum specification and execution engine.
///
/// Supports contraction order optimization via omeco.
///
/// # Example
///
/// ```rust
/// use omeinsum::{Einsum, Tensor, Cpu};
/// use omeinsum::algebra::Standard;
/// use std::collections::HashMap;
///
/// // A[i,j] × B[j,k] → C[i,k]
/// let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
/// let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
///
/// let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
/// let mut ein = Einsum::new(
///     vec![vec![0, 1], vec![1, 2]],
///     vec![0, 2],
///     sizes,
/// );
///
/// ein.optimize_greedy();
/// let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub struct Einsum<L: Label = usize> {
    /// Input index labels for each tensor
    pub ixs: Vec<Vec<L>>,

    /// Output index labels
    pub iy: Vec<L>,

    /// Dimension sizes for each index
    pub size_dict: HashMap<L, usize>,

    /// Optimized contraction tree (after optimization)
    optimized: Option<NestedEinsum<L>>,
}

impl<L: Label> Einsum<L> {
    /// Create a new einsum specification.
    ///
    /// # Arguments
    ///
    /// * `ixs` - Index labels for each input tensor
    /// * `iy` - Output index labels
    /// * `size_dict` - Mapping from index labels to dimension sizes
    pub fn new(ixs: Vec<Vec<L>>, iy: Vec<L>, size_dict: HashMap<L, usize>) -> Self {
        Self {
            ixs,
            iy,
            size_dict,
            optimized: None,
        }
    }

    /// Get the einsum code specification.
    pub fn code(&self) -> EinCode<L> {
        EinCode::new(self.ixs.clone(), self.iy.clone())
    }

    /// Optimize contraction order using greedy algorithm.
    ///
    /// Fast O(n²) algorithm, good for most cases.
    pub fn optimize_greedy(&mut self) -> &mut Self {
        let code = self.code();
        let optimizer = GreedyMethod::new(0.0, 0.0);
        self.optimized = optimize_code(&code, &self.size_dict, &optimizer);
        self
    }

    /// Optimize contraction order using simulated annealing.
    ///
    /// Slower but finds better orderings for complex networks.
    pub fn optimize_treesa(&mut self) -> &mut Self {
        let code = self.code();
        let optimizer = TreeSA::default();
        self.optimized = optimize_code(&code, &self.size_dict, &optimizer);
        self
    }

    /// Check if optimization has been performed.
    pub fn is_optimized(&self) -> bool {
        self.optimized.is_some()
    }

    /// Get the optimized contraction tree.
    pub fn contraction_tree(&self) -> Option<&NestedEinsum<L>> {
        self.optimized.as_ref()
    }

    /// Set a pre-computed contraction tree.
    ///
    /// This is useful for benchmarking with pre-optimized trees loaded from files.
    pub fn set_contraction_tree(&mut self, tree: NestedEinsum<L>) -> &mut Self {
        self.optimized = Some(tree);
        self
    }
}

struct OrderedTensor<T: Scalar, B: Backend> {
    tensor: Tensor<T, B>,
    indices: Vec<usize>,
}

impl Einsum<usize> {
    /// Execute the einsum contraction.
    ///
    /// # Type Parameters
    ///
    /// * `A` - The algebra to use (e.g., `Standard<f32>`, `MaxPlus<f32>`)
    /// * `T` - The scalar type
    /// * `B` - The backend type
    pub fn execute<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        assert_eq!(
            tensors.len(),
            self.ixs.len(),
            "Number of tensors {} doesn't match number of index specs {}",
            tensors.len(),
            self.ixs.len()
        );

        match &self.optimized {
            Some(tree) => {
                // Handle top-level Leaf (single tensor) specially to apply unary transformations
                if let NestedEinsum::Leaf { tensor_index } = tree {
                    execute_unary_naive::<A, T, B>(
                        tensors[*tensor_index],
                        &self.ixs[*tensor_index],
                        &self.iy,
                        &self.size_dict,
                    )
                } else {
                    let emit_final_root_output =
                        can_emit_final_root_output(optimized_tree_output(tree), &self.iy);
                    let result = self.execute_tree::<A, T, B>(
                        tree,
                        tensors,
                        emit_final_root_output.then_some(self.iy.as_slice()),
                    );
                    if emit_final_root_output {
                        result.tensor
                    } else {
                        finalize_ordered_result::<A, T, B>(result, &self.iy, &self.size_dict)
                    }
                }
            }
            None => self.execute_pairwise::<A, T, B>(tensors),
        }
    }

    /// Execute with argmax tracking for backpropagation.
    ///
    /// Returns `(result, argmax_cache)` where `argmax_cache` contains argmax
    /// tensors for each binary contraction in the execution tree.
    pub fn execute_with_argmax<A, T, B>(
        &self,
        tensors: &[&Tensor<T, B>],
    ) -> (Tensor<T, B>, Vec<Tensor<u32, B>>)
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        assert_eq!(
            tensors.len(),
            self.ixs.len(),
            "Number of tensors {} doesn't match number of index specs {}",
            tensors.len(),
            self.ixs.len()
        );

        let mut argmax_cache = Vec::new();

        let result = match &self.optimized {
            Some(tree) => {
                // Handle top-level Leaf (single tensor) specially to apply unary transformations
                if let NestedEinsum::Leaf { tensor_index } = tree {
                    if A::needs_argmax() {
                        let (result, argmax) = execute_unary_with_argmax::<A, T, B>(
                            tensors[*tensor_index],
                            &self.ixs[*tensor_index],
                            &self.iy,
                            &self.size_dict,
                        );
                        argmax_cache.push(argmax);
                        result
                    } else {
                        execute_unary_naive::<A, T, B>(
                            tensors[*tensor_index],
                            &self.ixs[*tensor_index],
                            &self.iy,
                            &self.size_dict,
                        )
                    }
                } else {
                    let emit_final_root_output =
                        can_emit_final_root_output(optimized_tree_output(tree), &self.iy);
                    let result = self.execute_tree_with_argmax::<A, T, B>(
                        tree,
                        tensors,
                        &mut argmax_cache,
                        emit_final_root_output.then_some(self.iy.as_slice()),
                    );
                    if emit_final_root_output {
                        result
                    } else {
                        finalize_optimized_result_with_argmax::<A, T, B>(
                            result,
                            tree,
                            &self.iy,
                            &self.size_dict,
                            &mut argmax_cache,
                        )
                    }
                }
            }
            None => self.execute_pairwise_with_argmax::<A, T, B>(tensors, &mut argmax_cache),
        };

        (result, argmax_cache)
    }

    /// Execute an optimized contraction tree with argmax tracking.
    #[allow(clippy::only_used_in_recursion)]
    fn execute_tree_with_argmax<A, T, B>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[&Tensor<T, B>],
        argmax_cache: &mut Vec<Tensor<u32, B>>,
        preferred_output_indices: Option<&[usize]>,
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        match tree {
            NestedEinsum::Leaf { tensor_index } => tensors[*tensor_index].clone(),
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "Expected binary contraction tree");

                let left =
                    self.execute_tree_with_argmax::<A, T, B>(&args[0], tensors, argmax_cache, None);
                let right =
                    self.execute_tree_with_argmax::<A, T, B>(&args[1], tensors, argmax_cache, None);

                let ia = &eins.ixs[0];
                let ib = &eins.ixs[1];
                let iy = &eins.iy;
                let (left, ia) =
                    normalize_binary_operand::<A, T, B>(&left, ia, ib, iy, &self.size_dict);
                let (right, ib) = normalize_binary_operand::<A, T, B>(
                    &right,
                    ib,
                    ia.as_slice(),
                    iy,
                    &self.size_dict,
                );

                if A::needs_argmax() {
                    let (result, argmax) =
                        if let Some(preferred_output_indices) = preferred_output_indices {
                            let options = BinaryContractOptions {
                                preferred_output_indices: Some(preferred_output_indices.to_vec()),
                            };
                            left.contract_binary_with_argmax_with_options::<A>(
                                &right, &ia, &ib, iy, &options,
                            )
                        } else {
                            left.contract_binary_with_argmax::<A>(&right, &ia, &ib, iy)
                        };
                    argmax_cache.push(argmax);
                    result
                } else if let Some(preferred_output_indices) = preferred_output_indices {
                    let options = BinaryContractOptions {
                        preferred_output_indices: Some(preferred_output_indices.to_vec()),
                    };
                    left.contract_binary_with_options::<A>(&right, &ia, &ib, iy, &options)
                } else {
                    left.contract_binary::<A>(&right, &ia, &ib, iy)
                }
            }
        }
    }

    /// Execute pairwise contraction with argmax tracking.
    fn execute_pairwise_with_argmax<A, T, B>(
        &self,
        tensors: &[&Tensor<T, B>],
        argmax_cache: &mut Vec<Tensor<u32, B>>,
    ) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        if tensors.is_empty() {
            panic!("Cannot execute einsum with no tensors");
        }

        if tensors.len() == 1 {
            if A::needs_argmax() {
                let (result, argmax) = execute_unary_with_argmax::<A, T, B>(
                    tensors[0],
                    &self.ixs[0],
                    &self.iy,
                    &self.size_dict,
                );
                argmax_cache.push(argmax);
                return result;
            } else {
                return execute_unary_naive::<A, T, B>(
                    tensors[0],
                    &self.ixs[0],
                    &self.iy,
                    &self.size_dict,
                );
            }
        }

        // Contract left to right
        let mut result = tensors[0].clone();
        let mut current_indices = self.ixs[0].clone();

        for i in 1..tensors.len() {
            let other = tensors[i];
            let other_indices = &self.ixs[i];

            let intermediate_output = if i == tensors.len() - 1 {
                self.iy.clone()
            } else {
                compute_intermediate_output(&current_indices, other_indices, &self.iy)
            };

            let (left, left_indices) = normalize_binary_operand::<A, T, B>(
                &result,
                &current_indices,
                other_indices,
                &intermediate_output,
                &self.size_dict,
            );
            let (right, right_indices) = normalize_binary_operand::<A, T, B>(
                other,
                other_indices,
                &left_indices,
                &intermediate_output,
                &self.size_dict,
            );

            if A::needs_argmax() {
                let (new_result, argmax) = left.contract_binary_with_argmax::<A>(
                    &right,
                    &left_indices,
                    &right_indices,
                    &intermediate_output,
                );
                argmax_cache.push(argmax);
                result = new_result;
            } else {
                result = left.contract_binary::<A>(
                    &right,
                    &left_indices,
                    &right_indices,
                    &intermediate_output,
                );
            }
            current_indices = intermediate_output;
        }

        result
    }

    /// Execute an optimized contraction tree.
    #[allow(clippy::only_used_in_recursion)]
    fn execute_tree<A, T, B>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[&Tensor<T, B>],
        preferred_output_indices: Option<&[usize]>,
    ) -> OrderedTensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        match tree {
            NestedEinsum::Leaf { tensor_index } => OrderedTensor {
                tensor: tensors[*tensor_index].clone(),
                indices: self.ixs[*tensor_index].clone(),
            },
            NestedEinsum::Node { args, eins } => {
                assert_eq!(args.len(), 2, "Expected binary contraction tree");

                let left = self.execute_tree::<A, T, B>(&args[0], tensors, None);
                let right = self.execute_tree::<A, T, B>(&args[1], tensors, None);

                let iy = &eins.iy;
                let (left_tensor, ia) = normalize_binary_operand::<A, T, B>(
                    &left.tensor,
                    &left.indices,
                    &right.indices,
                    iy,
                    &self.size_dict,
                );
                let (right, ib) = normalize_binary_operand::<A, T, B>(
                    &right.tensor,
                    &right.indices,
                    ia.as_slice(),
                    iy,
                    &self.size_dict,
                );

                if let Some(preferred_output_indices) = preferred_output_indices {
                    let options = BinaryContractOptions {
                        preferred_output_indices: Some(preferred_output_indices.to_vec()),
                    };
                    OrderedTensor {
                        tensor: left_tensor
                            .contract_binary_with_options::<A>(&right, &ia, &ib, iy, &options),
                        indices: preferred_output_indices.to_vec(),
                    }
                } else {
                    let (tensor, indices) =
                        left_tensor.contract_binary_native_order::<A>(&right, &ia, &ib, iy);
                    OrderedTensor { tensor, indices }
                }
            }
        }
    }

    /// Execute using simple pairwise contraction (no optimization).
    fn execute_pairwise<A, T, B>(&self, tensors: &[&Tensor<T, B>]) -> Tensor<T, B>
    where
        A: Algebra<Scalar = T, Index = u32>,
        T: Scalar + BackendScalar<B>,
        B: Backend,
    {
        if tensors.is_empty() {
            panic!("Cannot execute einsum with no tensors");
        }

        if tensors.len() == 1 {
            // Single tensor: just trace/reduce if needed
            return execute_unary_naive::<A, T, B>(
                tensors[0],
                &self.ixs[0],
                &self.iy,
                &self.size_dict,
            );
        }

        // Contract left to right
        let mut result = tensors[0].clone();
        let mut current_indices = self.ixs[0].clone();

        for i in 1..tensors.len() {
            let other = tensors[i];
            let other_indices = &self.ixs[i];

            // Determine output indices for this contraction
            let intermediate_output = if i == tensors.len() - 1 {
                // Last contraction: use final output
                self.iy.clone()
            } else {
                // Intermediate: keep all non-contracted indices
                compute_intermediate_output(&current_indices, other_indices, &self.iy)
            };

            let (left, left_indices) = normalize_binary_operand::<A, T, B>(
                &result,
                &current_indices,
                other_indices,
                &intermediate_output,
                &self.size_dict,
            );
            let (right, right_indices) = normalize_binary_operand::<A, T, B>(
                other,
                other_indices,
                &left_indices,
                &intermediate_output,
                &self.size_dict,
            );

            result = left.contract_binary::<A>(
                &right,
                &left_indices,
                &right_indices,
                &intermediate_output,
            );
            current_indices = intermediate_output;
        }

        result
    }
}

// =========================================================================
// CloneSemiring support: contraction for non-Copy semiring types.
// =========================================================================

impl Einsum<usize> {
    /// Execute the einsum contraction using a `CloneSemiring` (non-Copy types).
    ///
    /// Tensors are `(data, shape)` pairs. Contraction uses generic loops.
    #[deprecated(note = "Use contract() with DenseTensor instead")]
    pub fn execute_clone<S: crate::algebra::CloneSemiring>(
        &self,
        tensors: &[(Vec<S>, Vec<usize>)],
    ) -> (Vec<S>, Vec<usize>) {
        assert_eq!(tensors.len(), self.ixs.len());
        match &self.optimized {
            Some(tree) => {
                if let NestedEinsum::Leaf { tensor_index } = tree {
                    reduce_clone::<S>(
                        &tensors[*tensor_index].0,
                        &tensors[*tensor_index].1,
                        &self.ixs[*tensor_index],
                        &self.iy,
                    )
                } else {
                    self.execute_tree_clone::<S>(tree, tensors)
                }
            }
            None => self.execute_pairwise_clone::<S>(tensors),
        }
    }

    /// Contract tensors using generic loops. Works for any [`CloneSemiring`] type.
    ///
    /// Takes ownership of the tensor vector so the engine can consume
    /// intermediate results by value during tree execution.
    #[allow(deprecated)]
    pub fn contract<S: crate::algebra::CloneSemiring>(
        &self,
        tensors: Vec<DenseTensor<S>>,
    ) -> DenseTensor<S> {
        let raw: Vec<(Vec<S>, Vec<usize>)> = tensors
            .into_iter()
            .map(|t| t.into_data())
            .collect();
        let (data, shape) = self.execute_clone(&raw);
        DenseTensor::from_data(data, shape)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn execute_tree_clone<S: crate::algebra::CloneSemiring>(
        &self,
        tree: &NestedEinsum<usize>,
        tensors: &[(Vec<S>, Vec<usize>)],
    ) -> (Vec<S>, Vec<usize>) {
        match tree {
            NestedEinsum::Leaf { tensor_index } => {
                let (data, shape) = &tensors[*tensor_index];
                (data.clone(), shape.clone())
            }
            NestedEinsum::Node { args, eins } => {
                let (a_data, a_shape) = self.execute_tree_clone::<S>(&args[0], tensors);
                let (b_data, b_shape) = self.execute_tree_clone::<S>(&args[1], tensors);
                contract_clone::<S>(
                    &a_data, &a_shape, &eins.ixs[0],
                    &b_data, &b_shape, &eins.ixs[1],
                    &eins.iy,
                )
            }
        }
    }

    fn execute_pairwise_clone<S: crate::algebra::CloneSemiring>(
        &self,
        tensors: &[(Vec<S>, Vec<usize>)],
    ) -> (Vec<S>, Vec<usize>) {
        assert!(!tensors.is_empty());
        if tensors.len() == 1 {
            return reduce_clone::<S>(
                &tensors[0].0, &tensors[0].1, &self.ixs[0], &self.iy,
            );
        }
        let (mut data, mut shape) = tensors[0].clone();
        let mut current_ix = self.ixs[0].clone();
        for i in 1..tensors.len() {
            let iy = if i == tensors.len() - 1 {
                self.iy.clone()
            } else {
                compute_intermediate_output(&current_ix, &self.ixs[i], &self.iy)
            };
            let result = contract_clone::<S>(
                &data, &shape, &current_ix,
                &tensors[i].0, &tensors[i].1, &self.ixs[i],
                &iy,
            );
            data = result.0;
            shape = result.1;
            current_ix = iy;
        }
        (data, shape)
    }
}

/// Pairwise contraction for `CloneSemiring` types via generic loops.
fn contract_clone<S: crate::algebra::CloneSemiring>(
    a_data: &[S], a_shape: &[usize], modes_a: &[usize],
    b_data: &[S], b_shape: &[usize], modes_b: &[usize],
    modes_c: &[usize],
) -> (Vec<S>, Vec<usize>) {
    // Collect all distinct modes and their sizes
    let mut all_modes: Vec<usize> = Vec::new();
    let mut mode_size: HashMap<usize, usize> = HashMap::new();
    for (i, &m) in modes_a.iter().enumerate() {
        if !all_modes.contains(&m) {
            all_modes.push(m);
            mode_size.insert(m, a_shape[i]);
        }
    }
    for (i, &m) in modes_b.iter().enumerate() {
        if !all_modes.contains(&m) {
            all_modes.push(m);
            mode_size.insert(m, b_shape[i]);
        }
    }

    let c_shape: Vec<usize> = modes_c.iter().map(|m| mode_size[m]).collect();
    let c_numel = c_shape.iter().product::<usize>().max(1);
    let mut result: Vec<S> = (0..c_numel).map(|_| S::zero()).collect();

    // Precompute strides (column-major)
    let a_strides = col_major_strides(a_shape);
    let b_strides = col_major_strides(b_shape);
    let c_strides = col_major_strides(&c_shape);

    let all_sizes: Vec<usize> = all_modes.iter().map(|m| mode_size[m]).collect();
    let total: usize = all_sizes.iter().product::<usize>().max(1);

    for idx in 0..total {
        // Decode linear index → per-mode assignments (column-major)
        let mut remaining = idx;
        let mut assignment: HashMap<usize, usize> = HashMap::new();
        for (i, &m) in all_modes.iter().enumerate() {
            assignment.insert(m, remaining % all_sizes[i]);
            remaining /= all_sizes[i];
        }

        let a_idx: usize = modes_a.iter().enumerate()
            .map(|(d, &m)| assignment[&m] * a_strides[d]).sum();
        let b_idx: usize = modes_b.iter().enumerate()
            .map(|(d, &m)| assignment[&m] * b_strides[d]).sum();
        let c_idx: usize = modes_c.iter().enumerate()
            .map(|(d, &m)| assignment[&m] * c_strides[d]).sum();

        let product = a_data[a_idx].clone().mul(b_data[b_idx].clone());
        result[c_idx] = result[c_idx].clone().add(product);
    }
    (result, c_shape)
}

/// Unary reduction for `CloneSemiring`: sum (⊕) over modes not in the output.
fn reduce_clone<S: crate::algebra::CloneSemiring>(
    data: &[S], shape: &[usize], modes_in: &[usize], modes_out: &[usize],
) -> (Vec<S>, Vec<usize>) {
    if modes_in == modes_out {
        return (data.to_vec(), shape.to_vec());
    }
    // Build per-mode sizes from input
    let mode_size: HashMap<usize, usize> = modes_in.iter().zip(shape).map(|(&m, &s)| (m, s)).collect();
    let out_shape: Vec<usize> = modes_out.iter().map(|m| mode_size[m]).collect();
    let out_numel = out_shape.iter().product::<usize>().max(1);
    let mut result: Vec<S> = (0..out_numel).map(|_| S::zero()).collect();

    let out_strides = col_major_strides(&out_shape);
    let in_numel: usize = shape.iter().product::<usize>().max(1);

    for idx in 0..in_numel {
        let mut remaining = idx;
        let mut coords = vec![0; shape.len()];
        for d in 0..shape.len() {
            coords[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        let out_idx: usize = modes_out.iter().enumerate()
            .map(|(oi, &m)| {
                let pos = modes_in.iter().position(|&x| x == m).unwrap();
                coords[pos] * out_strides[oi]
            })
            .sum();
        result[out_idx] = result[out_idx].clone().add(data[idx].clone());
    }
    (result, out_shape)
}

/// Column-major strides for a given shape.
fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

/// Compute intermediate output indices for pairwise contraction.
fn compute_intermediate_output(ia: &[usize], ib: &[usize], final_output: &[usize]) -> Vec<usize> {
    let final_set: std::collections::HashSet<_> = final_output.iter().copied().collect();
    let ia_set: std::collections::HashSet<_> = ia.iter().copied().collect();
    let ib_set: std::collections::HashSet<_> = ib.iter().copied().collect();

    // Keep indices that are in the final output OR appear in only one input
    let mut output = Vec::new();

    for &i in ia {
        if (final_set.contains(&i) || !ib_set.contains(&i)) && !output.contains(&i) {
            output.push(i);
        }
    }

    for &i in ib {
        if (final_set.contains(&i) || !ia_set.contains(&i)) && !output.contains(&i) {
            output.push(i);
        }
    }

    output
}

pub(crate) fn compute_normalized_binary_indices(
    ix: &[usize],
    other: &[usize],
    output: &[usize],
) -> Vec<usize> {
    let other_set: HashSet<usize> = other.iter().copied().collect();
    let output_set: HashSet<usize> = output.iter().copied().collect();
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();

    for &idx in ix {
        if (other_set.contains(&idx) || output_set.contains(&idx)) && seen.insert(idx) {
            normalized.push(idx);
        }
    }

    normalized
}

pub(crate) fn normalize_binary_operand<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],
    other: &[usize],
    output: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> (Tensor<T, B>, Vec<usize>)
where
    A: Algebra<Scalar = T>,
    T: Scalar,
    B: Backend,
{
    let normalized_ix = compute_normalized_binary_indices(ix, other, output);
    if normalized_ix == ix {
        (tensor.clone(), normalized_ix)
    } else {
        (
            execute_unary_naive::<A, T, B>(tensor, ix, &normalized_ix, size_dict),
            normalized_ix,
        )
    }
}

fn optimized_tree_output(tree: &NestedEinsum<usize>) -> &[usize] {
    match tree {
        NestedEinsum::Leaf { .. } => {
            unreachable!("top-level leaf should be handled before finalizing optimized output")
        }
        NestedEinsum::Node { eins, .. } => &eins.iy,
    }
}

fn can_emit_final_root_output(tree_output: &[usize], final_output: &[usize]) -> bool {
    if tree_output.len() != final_output.len() {
        return false;
    }

    let tree_set: HashSet<usize> = tree_output.iter().copied().collect();
    let final_set: HashSet<usize> = final_output.iter().copied().collect();
    tree_set.len() == tree_output.len()
        && final_set.len() == final_output.len()
        && tree_set == final_set
}

fn permute_tensor_to_indices<T, B>(
    tensor: &Tensor<T, B>,
    current_indices: &[usize],
    target_indices: &[usize],
) -> Tensor<T, B>
where
    T: Scalar,
    B: Backend,
{
    assert_eq!(
        current_indices.len(),
        target_indices.len(),
        "current and target index orders must have the same length"
    );

    if current_indices == target_indices {
        return tensor.clone();
    }

    let axes: Vec<usize> = target_indices
        .iter()
        .map(|target| {
            current_indices
                .iter()
                .position(|current| current == target)
                .expect("target index must exist in current tensor indices")
        })
        .collect();
    tensor.permute(&axes)
}

fn finalize_ordered_result<A, T, B>(
    result: OrderedTensor<T, B>,
    expected_output: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T>,
    T: Scalar,
    B: Backend,
{
    if result.indices == expected_output {
        result.tensor
    } else if can_emit_final_root_output(&result.indices, expected_output) {
        permute_tensor_to_indices(&result.tensor, &result.indices, expected_output)
    } else {
        execute_unary_naive::<A, T, B>(&result.tensor, &result.indices, expected_output, size_dict)
    }
}

fn finalize_optimized_result_with_argmax<A, T, B>(
    result: Tensor<T, B>,
    tree: &NestedEinsum<usize>,
    expected_output: &[usize],
    size_dict: &HashMap<usize, usize>,
    argmax_cache: &mut Vec<Tensor<u32, B>>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    let tree_output = optimized_tree_output(tree);
    if tree_output == expected_output {
        result
    } else if A::needs_argmax() {
        let (result, argmax) =
            execute_unary_with_argmax::<A, T, B>(&result, tree_output, expected_output, size_dict);
        argmax_cache.push(argmax);
        result
    } else {
        execute_unary_naive::<A, T, B>(&result, tree_output, expected_output, size_dict)
    }
}

/// Convert linear index to multi-dimensional index (column-major).
///
/// Given a flat/linear index and a shape, returns the multi-dimensional
/// coordinates for column-major storage order.
///
/// # Arguments
///
/// * `linear` - The flat index into the tensor
/// * `shape` - The shape of the tensor
///
/// # Returns
///
/// A vector of indices, one per dimension
fn linear_to_multi(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut multi = vec![0; shape.len()];
    for i in 0..shape.len() {
        multi[i] = linear % shape[i];
        linear /= shape[i];
    }
    multi
}

#[cfg(test)]
mod cross_path_tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::tensor::{DenseTensor, Tensor};
    use crate::Cpu;

    #[test]
    fn test_contract_vs_execute_matmul() {
        // 2x3 @ 3x2 matmul
        let data_a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];

        // Generic path
        let dense_a = DenseTensor::from_data(
            data_a.iter().map(|&x| Standard(x)).collect(),
            vec![2, 3],
        );
        let dense_b = DenseTensor::from_data(
            data_b.iter().map(|&x| Standard(x)).collect(),
            vec![3, 2],
        );
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes.clone());
        ein.optimize_greedy();
        let result_generic = ein.contract(vec![dense_a, dense_b]);

        // Backend path
        let tensor_a = Tensor::<f64, Cpu>::from_data(&data_a, &[2, 3]);
        let tensor_b = Tensor::<f64, Cpu>::from_data(&data_b, &[3, 2]);
        let mut ein2 = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);
        ein2.optimize_greedy();
        let result_backend = ein2.execute::<Standard<f64>, _, _>(&[&tensor_a, &tensor_b]);

        // Compare
        assert_eq!(result_generic.shape(), result_backend.shape());
        for i in 0..result_generic.len() {
            let generic_val = result_generic.get(i).0;
            let backend_val = result_backend.get(i);
            assert!(
                (generic_val - backend_val).abs() < 1e-10,
                "Mismatch at index {}: generic={}, backend={}",
                i, generic_val, backend_val,
            );
        }
    }
}

/// Compute input tensor position from index values (column-major).
///
/// Given index labels and their current values, computes the flat position
/// in the input tensor using column-major ordering.
///
/// # Arguments
///
/// * `ix` - The index labels for the input tensor
/// * `idx_values` - Mapping from index label to current value
/// * `shape` - The shape of the input tensor
///
/// # Returns
///
/// The flat position in the tensor
fn compute_input_position(
    ix: &[usize],
    idx_values: &HashMap<usize, usize>,
    shape: &[usize],
) -> usize {
    let mut pos = 0;
    let mut stride = 1;
    for (dim, &idx) in ix.iter().enumerate() {
        pos += idx_values[&idx] * stride;
        stride *= shape[dim];
    }
    pos
}

/// Execute unary einsum operation using naive loop.
/// Handles trace, diagonal, sum, permutation uniformly.
///
/// # Type Parameters
///
/// * `A` - The algebra to use for accumulation
/// * `T` - The scalar type
/// * `B` - The backend type
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `ix` - Input index labels (may contain repeated indices for trace/diagonal)
/// * `iy` - Output index labels
/// * `size_dict` - Mapping from index labels to dimension sizes
///
/// # Key Insight
///
/// For repeated indices like `ix = [0, 1, 1, 2]` (ijjk), positions 1 and 2 both map
/// to index label `1`. This automatically handles diagonal extraction because
/// `compute_input_position` uses `idx_values[&idx]` - when the same index label
/// appears multiple times in `ix`, those positions will use the same value.
#[allow(clippy::needless_range_loop)]
pub(crate) fn execute_unary_naive<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Tensor<T, B>
where
    A: Algebra<Scalar = T>,
    T: Scalar,
    B: Backend,
{
    // 1. Classify indices
    // outer = output indices
    // inner = indices that appear in input but not in output (summed over)
    let outer: &[usize] = iy;
    let outer_set: HashSet<usize> = outer.iter().copied().collect();
    // Collect inner indices deterministically, preserving the order from `ix`
    let mut inner_vec: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for i in ix.iter().copied().filter(|i| !outer_set.contains(i)) {
        if seen.insert(i) {
            inner_vec.push(i);
        }
    }

    // 2. Build output shape
    let out_shape: Vec<usize> = outer.iter().map(|&idx| size_dict[&idx]).collect();
    let out_size = out_shape.iter().product::<usize>().max(1);

    // 3. Build inner ranges (dimensions to sum over)
    let inner_ranges: Vec<usize> = inner_vec.iter().map(|&idx| size_dict[&idx]).collect();
    let inner_size = inner_ranges.iter().product::<usize>().max(1);

    // 4. Allocate output
    let mut out_data = vec![A::zero().to_scalar(); out_size];

    // 5. Loop over output positions
    for out_linear in 0..out_size {
        let out_multi = linear_to_multi(out_linear, &out_shape);

        // Map: outer index label -> value
        // For repeated output indices (like `ii`), check consistency
        let mut idx_values: HashMap<usize, usize> = HashMap::new();
        let mut skip_position = false;

        for (&idx, &val) in outer.iter().zip(out_multi.iter()) {
            if let Some(&existing) = idx_values.get(&idx) {
                // Repeated index label - values must match
                if existing != val {
                    skip_position = true;
                    break;
                }
            } else {
                idx_values.insert(idx, val);
            }
        }

        // Skip non-diagonal positions for repeated output indices
        if skip_position {
            // out_data[out_linear] is already zero
            continue;
        }

        // 6. Accumulate over inner indices
        let mut acc = A::zero();
        for inner_linear in 0..inner_size {
            let inner_multi = linear_to_multi(inner_linear, &inner_ranges);
            for (&idx, &val) in inner_vec.iter().zip(inner_multi.iter()) {
                idx_values.insert(idx, val);
            }

            // 7. Compute input position and accumulate
            let in_pos = compute_input_position(ix, &idx_values, tensor.shape());
            acc = acc.add(A::from_scalar(tensor.get(in_pos)));
        }

        out_data[out_linear] = acc.to_scalar();
    }

    Tensor::from_data_with_backend(&out_data, &out_shape, tensor.backend().clone())
}

/// Execute unary einsum with argmax tracking for tropical algebras.
///
/// Returns both the result tensor and an argmax tensor that tracks which
/// inner index position "won" for each output element.
///
/// The argmax tensor has the same shape as the output. Each element stores
/// the linear index into the input tensor that contributed to that output.
pub(crate) fn execute_unary_with_argmax<A, T, B>(
    tensor: &Tensor<T, B>,
    ix: &[usize],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> (Tensor<T, B>, Tensor<u32, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar,
    B: Backend,
{
    // 1. Classify indices (same as execute_unary_naive)
    let outer: &[usize] = iy;
    let outer_set: HashSet<usize> = outer.iter().copied().collect();
    let mut inner_vec: Vec<usize> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for i in ix.iter().copied().filter(|i| !outer_set.contains(i)) {
        if seen.insert(i) {
            inner_vec.push(i);
        }
    }

    // 2. Build output shape
    let out_shape: Vec<usize> = outer.iter().map(|&idx| size_dict[&idx]).collect();
    let out_size = out_shape.iter().product::<usize>().max(1);

    // 3. Build inner ranges
    let inner_ranges: Vec<usize> = inner_vec.iter().map(|&idx| size_dict[&idx]).collect();
    let inner_size = inner_ranges.iter().product::<usize>().max(1);

    // 4. Allocate output and argmax
    let mut out_data = vec![A::zero().to_scalar(); out_size];
    let mut argmax_data = vec![0u32; out_size];

    // 5. Loop over output positions
    for out_linear in 0..out_size {
        let out_multi = linear_to_multi(out_linear, &out_shape);

        // Map: outer index label -> value
        let mut idx_values: HashMap<usize, usize> = HashMap::new();
        let mut skip_position = false;

        for (&idx, &val) in outer.iter().zip(out_multi.iter()) {
            if let Some(&existing) = idx_values.get(&idx) {
                if existing != val {
                    skip_position = true;
                    break;
                }
            } else {
                idx_values.insert(idx, val);
            }
        }

        if skip_position {
            continue;
        }

        // 6. Find max over inner indices (tropical-style)
        let mut best_val = A::zero();
        let mut best_in_pos = 0usize;

        for inner_linear in 0..inner_size {
            let inner_multi = linear_to_multi(inner_linear, &inner_ranges);
            for (&idx, &val) in inner_vec.iter().zip(inner_multi.iter()) {
                idx_values.insert(idx, val);
            }

            let in_pos = compute_input_position(ix, &idx_values, tensor.shape());
            let val = A::from_scalar(tensor.get(in_pos));

            // For first iteration or if this value is better
            if inner_linear == 0 || A::is_better(&val, &best_val) {
                best_val = val;
                best_in_pos = in_pos;
            }
        }

        out_data[out_linear] = best_val.to_scalar();
        argmax_data[out_linear] = best_in_pos as u32;
    }

    let result = Tensor::from_data_with_backend(&out_data, &out_shape, tensor.backend().clone());
    let argmax = Tensor::from_data_with_backend(&argmax_data, &out_shape, tensor.backend().clone());

    (result, argmax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::backend::Cpu;

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_einsum_matmul() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        // Without optimization
        let c1 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c1.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);

        // With optimization
        ein.optimize_greedy();
        let c2 = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c2.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_tropical() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        let c = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a, &b]);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_einsum_chain() {
        // A[i,j] × B[j,k] × C[k,l] → D[i,l]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], vec![0, 3], sizes);

        ein.optimize_greedy();
        let d = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b, &c]);

        assert_eq!(d.shape(), &[2, 2]);
    }

    #[test]
    fn test_einsum_trace() {
        // Trace: A[i,i] -> scalar (sum of diagonal)
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // trace = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_einsum_diagonal() {
        // Diagonal: A[i,i] -> B[i] (extract diagonal)
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![0], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // diagonal = [1, 4]
        assert_eq!(result.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_einsum_sum_axis() {
        // Reduction: A[i,j] -> B[i] (sum over j)
        // Column-major: data [1,2,3,4] for shape [2,2] represents:
        // [[1, 3],
        //  [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1]], vec![0], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // sum over j: [1+3, 2+4] = [4, 6]
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_einsum_sum_all() {
        // Sum all: A[i,j] -> scalar
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1]], vec![], sizes);

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);
        // sum = 1 + 2 + 3 + 4 = 10
        assert_eq!(result.to_vec()[0], 10.0);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_trace_tropical() {
        // Trace with max-plus algebra: A[i,i] -> scalar
        // Matrix: [[1, 2], [3, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        let result = ein.execute::<MaxPlus<f32>, f32, Cpu>(&[&a]);
        // tropical trace = max(1, 4) = 4
        assert_eq!(result.to_vec()[0], 4.0);
    }

    #[test]
    fn test_einsum_binary_repeated_left_label_pairwise() {
        // ii,ij->j should contract the diagonal of the left input with the right input.
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0], vec![0, 1]], vec![1], sizes);

        let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_einsum_binary_repeated_left_label_optimized() {
        // iib,bc->ic should first extract the diagonal across the repeated i label.
        let a =
            Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0, 1], vec![1, 2]], vec![0, 2], sizes);
        ein.optimize_greedy();

        let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0, 5.0, 8.0]);
    }

    #[test]
    fn test_einsum_binary_repeated_label_reduced_to_scalar_before_contract() {
        // ii,jk->jk should trace the left input to a scalar and scale the right input.
        let a = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f64, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0], vec![1, 2]], vec![1, 2], sizes);

        let result = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![5.0, 10.0, 15.0, 20.0]);

        ein.optimize_greedy();
        let optimized = ein.execute::<Standard<f64>, f64, Cpu>(&[&a, &b]);
        assert_eq!(optimized.shape(), &[2, 2]);
        assert_eq!(optimized.to_vec(), vec![5.0, 10.0, 15.0, 20.0]);
    }

    // Tests for helper functions

    #[test]
    fn test_root_output_plan_uses_final_order_for_pure_permutation() {
        let tree_output = vec![2, 0, 1];
        let final_output = vec![0, 1, 2];

        assert!(can_emit_final_root_output(&tree_output, &final_output));
    }

    #[test]
    fn test_root_output_plan_rejects_non_permutation_finalize_cases() {
        let tree_output = vec![0, 1, 2];
        let final_output = vec![0, 0, 2];

        assert!(!can_emit_final_root_output(&tree_output, &final_output));
    }

    #[test]
    fn test_permute_tensor_to_indices_builds_zero_copy_view() {
        let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let tensor = Tensor::<f32, Cpu>::from_data(&data, &[4, 2, 3]);

        let reordered = permute_tensor_to_indices(&tensor, &[2, 0, 1], &[0, 1, 2]);

        assert_eq!(reordered.shape(), &[2, 3, 4]);
        assert_eq!(reordered.strides(), &[4, 8, 1]);
        assert!(!reordered.is_contiguous());
        assert_eq!(reordered.get(6), data[1]);
    }

    #[test]
    fn test_permute_tensor_to_indices_keeps_identity_layout() {
        let data: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let tensor = Tensor::<f32, Cpu>::from_data(&data, &[2, 3, 4]);

        let reordered = permute_tensor_to_indices(&tensor, &[0, 1, 2], &[0, 1, 2]);

        assert_eq!(reordered.shape(), tensor.shape());
        assert_eq!(reordered.strides(), tensor.strides());
        assert!(reordered.is_contiguous());
        assert_eq!(reordered.to_vec(), tensor.to_vec());
    }

    #[test]
    fn test_execute_tree_tracks_native_binary_output_order() {
        let a =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2), (3, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1, 2], vec![0, 2, 3]], vec![0, 1, 3], sizes);
        let tree = NestedEinsum::node(
            vec![NestedEinsum::leaf(0), NestedEinsum::leaf(1)],
            EinCode::new(vec![vec![0, 1, 2], vec![0, 2, 3]], vec![0, 1, 3]),
        );

        let ordered = ein.execute_tree::<Standard<f32>, f32, Cpu>(&tree, &[&a, &b], None);
        let finalized = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);

        assert_eq!(ordered.indices, vec![1, 3, 0]);
        assert_eq!(
            permute_tensor_to_indices(&ordered.tensor, &ordered.indices, &[0, 1, 3]).to_vec(),
            finalized.to_vec()
        );
    }

    #[test]
    fn test_linear_to_multi_empty_shape() {
        // Empty shape should return empty multi-index
        let result = linear_to_multi(0, &[]);
        assert_eq!(result, Vec::<usize>::new());
    }

    #[test]
    fn test_linear_to_multi_1d() {
        // 1D array: linear index equals multi-index
        assert_eq!(linear_to_multi(0, &[5]), vec![0]);
        assert_eq!(linear_to_multi(3, &[5]), vec![3]);
        assert_eq!(linear_to_multi(4, &[5]), vec![4]);
    }

    #[test]
    fn test_linear_to_multi_2d() {
        // 2D array with shape [2, 3] (column-major)
        // Linear 0 -> (0, 0)
        // Linear 1 -> (1, 0)
        // Linear 2 -> (0, 1)
        // Linear 3 -> (1, 1)
        // Linear 4 -> (0, 2)
        // Linear 5 -> (1, 2)
        assert_eq!(linear_to_multi(0, &[2, 3]), vec![0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3]), vec![1, 0]);
        assert_eq!(linear_to_multi(2, &[2, 3]), vec![0, 1]);
        assert_eq!(linear_to_multi(3, &[2, 3]), vec![1, 1]);
        assert_eq!(linear_to_multi(4, &[2, 3]), vec![0, 2]);
        assert_eq!(linear_to_multi(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn test_linear_to_multi_3d() {
        // 3D array with shape [2, 3, 4] (column-major)
        // Strides: [1, 2, 6]
        // Linear 0 -> (0, 0, 0)
        // Linear 1 -> (1, 0, 0)
        // Linear 2 -> (0, 1, 0)
        // Linear 6 -> (0, 0, 1)
        // Linear 7 -> (1, 0, 1)
        assert_eq!(linear_to_multi(0, &[2, 3, 4]), vec![0, 0, 0]);
        assert_eq!(linear_to_multi(1, &[2, 3, 4]), vec![1, 0, 0]);
        assert_eq!(linear_to_multi(2, &[2, 3, 4]), vec![0, 1, 0]);
        assert_eq!(linear_to_multi(6, &[2, 3, 4]), vec![0, 0, 1]);
        assert_eq!(linear_to_multi(7, &[2, 3, 4]), vec![1, 0, 1]);
        // Last element: linear 23 -> (1, 2, 3)
        assert_eq!(linear_to_multi(23, &[2, 3, 4]), vec![1, 2, 3]);
    }

    #[test]
    fn test_compute_input_position_1d() {
        // 1D tensor with index label 0
        let ix = vec![0];
        let shape = vec![5];

        let mut idx_values = HashMap::new();
        idx_values.insert(0, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        idx_values.insert(0, 3);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 3);
    }

    #[test]
    fn test_compute_input_position_2d() {
        // 2D tensor with shape [2, 3], index labels (0, 1)
        // Column-major: position = i + j * 2
        let ix = vec![0, 1];
        let shape = vec![2, 3];

        let mut idx_values = HashMap::new();

        // (0, 0) -> position 0
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        // (1, 0) -> position 1
        idx_values.insert(0, 1);
        idx_values.insert(1, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 1);

        // (0, 1) -> position 2
        idx_values.insert(0, 0);
        idx_values.insert(1, 1);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 2);

        // (1, 2) -> position 1 + 2*2 = 5
        idx_values.insert(0, 1);
        idx_values.insert(1, 2);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 5);
    }

    #[test]
    fn test_compute_input_position_3d() {
        // 3D tensor with shape [2, 3, 4], index labels (0, 1, 2)
        // Column-major: position = i + j * 2 + k * 6
        let ix = vec![0, 1, 2];
        let shape = vec![2, 3, 4];

        let mut idx_values = HashMap::new();

        // (0, 0, 0) -> position 0
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 0);

        // (1, 0, 0) -> position 1
        idx_values.insert(0, 1);
        idx_values.insert(1, 0);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 1);

        // (0, 1, 0) -> position 2
        idx_values.insert(0, 0);
        idx_values.insert(1, 1);
        idx_values.insert(2, 0);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 2);

        // (0, 0, 1) -> position 6
        idx_values.insert(0, 0);
        idx_values.insert(1, 0);
        idx_values.insert(2, 1);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 6);

        // (1, 2, 3) -> position 1 + 2*2 + 3*6 = 1 + 4 + 18 = 23
        idx_values.insert(0, 1);
        idx_values.insert(1, 2);
        idx_values.insert(2, 3);
        assert_eq!(compute_input_position(&ix, &idx_values, &shape), 23);
    }

    #[test]
    fn test_linear_to_multi_roundtrip() {
        // Verify that linear_to_multi and compute_input_position are consistent
        let shape = vec![2, 3, 4];
        let ix: Vec<usize> = (0..shape.len()).collect();
        let total_size: usize = shape.iter().product();

        for linear in 0..total_size {
            let multi = linear_to_multi(linear, &shape);

            // Build idx_values from multi
            let mut idx_values = HashMap::new();
            for (dim, &val) in multi.iter().enumerate() {
                idx_values.insert(dim, val);
            }

            let computed_pos = compute_input_position(&ix, &idx_values, &shape);
            assert_eq!(
                computed_pos, linear,
                "Roundtrip failed for linear={}, multi={:?}",
                linear, multi
            );
        }
    }

    // ========================================================================
    // Tests for execute_unary_naive
    // ========================================================================

    #[test]
    fn test_unary_naive_transpose() {
        // Transpose: A[i,j] -> B[j,i]
        // Input matrix (column-major): [[1, 3], [2, 4]]
        // data = [1, 2, 3, 4], shape = [2, 2]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![1, 0]; // B[j,i]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // After transpose: [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_unary_naive_trace() {
        // Trace: A[i,i] -> scalar (sum of diagonal)
        // Matrix (column-major): [[1, 3], [2, 4]]
        // data = [1, 2, 3, 4], shape = [2, 2]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i] - repeated index means diagonal
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // trace = A[0,0] + A[1,1] = 1 + 4 = 5
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_unary_naive_diagonal() {
        // Diagonal extraction: A[i,i] -> B[i]
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i] - repeated index
        let iy = vec![0]; // output B[i]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // diagonal = [A[0,0], A[1,1]] = [1, 4]
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_unary_naive_sum_axis() {
        // Sum over axis: A[i,j] -> B[i] (sum over j)
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![0]; // B[i] - j is summed out

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // sum over j: B[i] = sum_j A[i,j]
        // B[0] = A[0,0] + A[0,1] = 1 + 3 = 4
        // B[1] = A[1,0] + A[1,1] = 2 + 4 = 6
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_unary_naive_sum_all() {
        // Sum all: A[i,j] -> scalar
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // sum all = 1 + 2 + 3 + 4 = 10
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.to_vec()[0], 10.0);
    }

    #[test]
    fn test_unary_naive_partial_trace() {
        // Partial trace: A[i,j,i] -> B[j] (trace over i, keeping j)
        // 3D tensor with shape [2, 3, 2]
        // This is like having a batch of 2x2 matrices and taking the trace of each
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let a = Tensor::<f32, Cpu>::from_data(&data, &[2, 3, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let ix = vec![0, 1, 0]; // A[i,j,i] - i is repeated at positions 0 and 2
        let iy = vec![1]; // B[j] - output keeps only j

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // For each j, we sum A[0,j,0] + A[1,j,1]
        // Column-major layout: data[i + j*2 + k*6]
        // j=0: A[0,0,0] + A[1,0,1] = data[0] + data[1+0*2+1*6] = data[0] + data[7] = 1 + 8 = 9
        // j=1: A[0,1,0] + A[1,1,1] = data[0+1*2+0*6] + data[1+1*2+1*6] = data[2] + data[9] = 3 + 10 = 13
        // j=2: A[0,2,0] + A[1,2,1] = data[0+2*2+0*6] + data[1+2*2+1*6] = data[4] + data[11] = 5 + 12 = 17
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.to_vec(), vec![9.0, 13.0, 17.0]);
    }

    #[test]
    fn test_unary_naive_3d_transpose() {
        // 3D permutation: A[i,j,k] -> B[k,i,j]
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let a = Tensor::<f32, Cpu>::from_data(&data, &[2, 2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let ix = vec![0, 1, 2]; // A[i,j,k]
        let iy = vec![2, 0, 1]; // B[k,i,j]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        assert_eq!(result.shape(), &[2, 2, 2]);

        // Verify by checking specific elements
        // B[k,i,j] = A[i,j,k]
        // Build expected output manually
        let mut expected = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    // A[i,j,k] at position i + j*2 + k*4 in column-major
                    let a_pos = i + j * 2 + k * 4;
                    // B[k,i,j] at position k + i*2 + j*4 in column-major
                    let b_pos = k + i * 2 + j * 4;
                    expected[b_pos] = data[a_pos];
                }
            }
        }
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_unary_naive_identity() {
        // Identity: A[i,j] -> B[i,j] (no change)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2), (1, 2)].into();
        let ix = vec![0, 1]; // A[i,j]
        let iy = vec![0, 1]; // B[i,j]

        let result = execute_unary_naive::<Standard<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), a.to_vec());
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_unary_naive_trace_tropical() {
        // Trace with max-plus algebra: A[i,i] -> scalar
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let size_dict: HashMap<usize, usize> = [(0, 2)].into();
        let ix = vec![0, 0]; // A[i,i]
        let iy = vec![]; // scalar output

        let result = execute_unary_naive::<MaxPlus<f32>, f32, Cpu>(&a, &ix, &iy, &size_dict);

        // tropical trace = max(A[0,0], A[1,1]) = max(1, 4) = 4
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.to_vec()[0], 4.0);
    }

    #[test]
    fn test_einsum_trace_optimized() {
        // Test that the optimized path correctly handles unary trace operations
        // Matrix (column-major): [[1, 3], [2, 4]]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        // Optimize and execute
        ein.optimize_greedy();
        assert!(ein.is_optimized());

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);

        // trace = A[0,0] + A[1,1] = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_einsum_unary_with_argmax_optimized() {
        // Test execute_with_argmax for unary operations (optimized path)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        ein.optimize_greedy();
        let (result, argmax_cache) = ein.execute_with_argmax::<Standard<f32>, f32, Cpu>(&[&a]);

        // trace = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
        // No argmax for unary operations
        assert!(argmax_cache.is_empty());
    }

    #[test]
    fn test_einsum_unary_pairwise_path() {
        // Test unary operation through pairwise path (no optimization)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        // Not optimized - uses pairwise path
        assert!(!ein.is_optimized());

        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);

        // trace = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
    }

    #[test]
    fn test_einsum_unary_with_argmax_pairwise() {
        // Test execute_with_argmax for unary operations (pairwise path)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2)].into();
        let ein = Einsum::new(vec![vec![0, 0]], vec![], sizes);

        // Not optimized - uses pairwise path
        let (result, argmax_cache) = ein.execute_with_argmax::<Standard<f32>, f32, Cpu>(&[&a]);

        // trace = 1 + 4 = 5
        assert_eq!(result.to_vec()[0], 5.0);
        // No argmax for unary operations
        assert!(argmax_cache.is_empty());
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_with_argmax_tropical() {
        // Test execute_with_argmax for tropical algebra (needs argmax)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        ein.optimize_greedy();
        let (result, argmax_cache) = ein.execute_with_argmax::<MaxPlus<f32>, f32, Cpu>(&[&a, &b]);

        // MaxPlus matmul: C[i,k] = max_j(A[i,j] + B[j,k])
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);

        // Should have argmax tensors for binary contractions
        assert!(!argmax_cache.is_empty());
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_einsum_with_argmax_tropical_pairwise() {
        // Test execute_with_argmax for tropical algebra (pairwise path)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);

        // Not optimized - uses pairwise path
        let (result, argmax_cache) = ein.execute_with_argmax::<MaxPlus<f32>, f32, Cpu>(&[&a, &b]);

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);

        // Should have argmax tensors
        assert!(!argmax_cache.is_empty());
    }

    #[test]
    fn test_einsum_transpose_optimized() {
        // Test transpose operation through optimized path
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let mut ein = Einsum::new(vec![vec![0, 1]], vec![1, 0], sizes);

        ein.optimize_greedy();
        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a]);

        assert_eq!(result.shape(), &[3, 2]);
        // A (col-major) = [[1,3,5],[2,4,6]]
        // A^T = [[1,2],[3,4],[5,6]]
        // In col-major: [1, 3, 5, 2, 4, 6]
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_intermediate_output_computation() {
        // Test the compute_intermediate_output function
        // ij,jk->ik: j is contracted
        let output = compute_intermediate_output(&[0, 1], &[1, 2], &[0, 2]);
        assert!(output.contains(&0));
        assert!(output.contains(&2));
        assert!(!output.contains(&1));
    }

    #[test]
    fn test_outer_product_pairwise() {
        // Test outer product via pairwise path (no optimization)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0], &[2]);
        let b = Tensor::<f32, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let ein = Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes);

        // NOT optimized - uses pairwise path
        assert!(!ein.is_optimized());
        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);

        assert_eq!(result.shape(), &[2, 3]);
        // Outer product: a ⊗ b = [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]]
        //                      = [[3, 4, 5], [6, 8, 10]]
        // In column-major: [3, 6, 4, 8, 5, 10]
        assert_eq!(result.to_vec(), vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
    }

    #[test]
    fn test_outer_product_optimized() {
        // Test outer product with optimization
        // The optimizer returns Leaf for outer products (no shared indices),
        // but we detect this and fall back to pairwise execution
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0], &[2]);
        let b = Tensor::<f32, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let mut ein = Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes);

        ein.optimize_greedy();
        let result = ein.execute::<Standard<f32>, f32, Cpu>(&[&a, &b]);

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
    }

    #[test]
    fn test_outer_product_with_argmax() {
        // Test outer product through execute_with_argmax path
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0], &[2]);
        let b = Tensor::<f32, Cpu>::from_data(&[3.0, 4.0, 5.0], &[3]);

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 3)].into();
        let mut ein = Einsum::new(vec![vec![0], vec![1]], vec![0, 1], sizes);

        ein.optimize_greedy();
        let (result, _argmax_cache) = ein.execute_with_argmax::<Standard<f32>, f32, Cpu>(&[&a, &b]);

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::algebra::CloneSemiring;
    use crate::tensor::DenseTensor;

    #[derive(Clone, Debug, PartialEq)]
    struct SumSemiring(f64);

    impl CloneSemiring for SumSemiring {
        fn zero() -> Self { SumSemiring(0.0) }
        fn one() -> Self { SumSemiring(1.0) }
        fn add(self, rhs: Self) -> Self { SumSemiring(self.0 + rhs.0) }
        fn mul(self, rhs: Self) -> Self { SumSemiring(self.0 * rhs.0) }
        fn is_zero(&self) -> bool { self.0 == 0.0 }
    }

    #[test]
    fn test_contract_matmul() {
        // A[i,j] * B[j,k] -> C[i,k], 2x2 matmul
        let a = DenseTensor::from_data(
            vec![SumSemiring(1.0), SumSemiring(2.0), SumSemiring(3.0), SumSemiring(4.0)],
            vec![2, 2], // column-major: [[1,3],[2,4]]
        );
        let b = DenseTensor::from_data(
            vec![SumSemiring(5.0), SumSemiring(6.0), SumSemiring(7.0), SumSemiring(8.0)],
            vec![2, 2], // column-major: [[5,7],[6,8]]
        );

        let sizes: HashMap<usize, usize> = [(0, 2), (1, 2), (2, 2)].into();
        let mut ein = Einsum::new(vec![vec![0, 1], vec![1, 2]], vec![0, 2], sizes);
        ein.optimize_greedy();

        let result = ein.contract(vec![a, b]);
        // C = A @ B = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]]
        //           = [[23, 31], [34, 46]]
        // column-major: [23, 34, 31, 46]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get(0).0, 23.0);
        assert_eq!(result.get(1).0, 34.0);
        assert_eq!(result.get(2).0, 31.0);
        assert_eq!(result.get(3).0, 46.0);
    }
}
