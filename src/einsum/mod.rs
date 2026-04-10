//! Einstein summation engine with contraction order optimization.
//!
//! This module provides the [`Einsum`] type for specifying and executing
//! tensor network contractions, with optional optimization via omeco.

mod backward;
mod builder;
mod engine;

pub use backward::cost_and_gradient;
pub use builder::EinBuilder;
pub use engine::Einsum;

use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar};
use crate::tensor::Tensor;

/// One-shot einsum with automatic optimization.
///
/// # Arguments
///
/// * `tensors` - Input tensors
/// * `ixs` - Index labels for each input tensor
/// * `iy` - Output index labels
///
/// # Example
///
/// ```rust
/// use omeinsum::{einsum, Tensor, Cpu};
/// use omeinsum::algebra::Standard;
///
/// let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
/// let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 4]);
///
/// // C[i,k] = Σ_j A[i,j] × B[j,k]
/// let c = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
/// assert_eq!(c.shape(), &[2, 4]);
/// ```
pub fn einsum<A, T, B>(tensors: &[&Tensor<T, B>], ixs: &[&[usize]], iy: &[usize]) -> Tensor<T, B>
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    let size_dict = infer_size_dict(tensors, ixs);
    let ixs_owned: Vec<Vec<usize>> = ixs.iter().map(|ix| ix.to_vec()).collect();

    let mut ein = Einsum::new(ixs_owned, iy.to_vec(), size_dict);
    ein.optimize_greedy();
    ein.execute::<A, T, B>(tensors)
}

/// Einsum with gradient computation.
///
/// Returns `(result, gradient_fn)` where `gradient_fn` can be called
/// with the output gradient to compute input gradients.
///
/// For Standard algebra, gradients are computed via einsum (no argmax tracking needed).
/// For tropical algebras, argmax is tracked during forward pass for gradient routing.
pub fn einsum_with_grad<A, T, B>(
    tensors: &[&Tensor<T, B>],
    ixs: &[&[usize]],
    iy: &[usize],
) -> (Tensor<T, B>, EinsumGradient<T, B>)
where
    A: Algebra<Scalar = T, Index = u32>,
    T: Scalar + BackendScalar<B>,
    B: Backend,
{
    let size_dict = infer_size_dict(tensors, ixs);
    let ixs_owned: Vec<Vec<usize>> = ixs.iter().map(|ix| ix.to_vec()).collect();

    let mut ein = Einsum::new(ixs_owned, iy.to_vec(), size_dict);
    ein.optimize_greedy();

    let tape = backward::build_gradient_tape::<A, T, B>(&ein, tensors);
    let result = tape.result();
    let gradient = EinsumGradient { tape };

    (result, gradient)
}

/// Gradient computation helper for einsum.
pub struct EinsumGradient<T: Scalar, B: Backend> {
    tape: backward::GradientTape<T, B>,
}

impl<T: Scalar + BackendScalar<B>, B: Backend> EinsumGradient<T, B> {
    /// Compute gradients for all inputs given the output gradient.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the einsum output
    /// * `inputs` - Original input tensors (same as passed to forward)
    ///
    /// # Returns
    ///
    /// Vector of gradients, one for each input tensor.
    pub fn backward<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        grad_output: &Tensor<T, B>,
        inputs: &[&Tensor<T, B>],
    ) -> Vec<Tensor<T, B>> {
        assert_eq!(
            inputs.len(),
            self.tape.num_inputs(),
            "Number of inputs {} doesn't match stored indices {}",
            inputs.len(),
            self.tape.num_inputs()
        );

        for (input, expected_shape) in inputs.iter().zip(self.tape.input_shapes()) {
            assert_eq!(
                input.shape(),
                expected_shape.as_slice(),
                "Input shape {:?} doesn't match tape shape {:?}",
                input.shape(),
                expected_shape
            );
        }

        self.tape.backward::<A>(grad_output)
    }
}

/// Infer size dictionary from tensors and their index labels.
fn infer_size_dict<T: Scalar, B: Backend>(
    tensors: &[&Tensor<T, B>],
    ixs: &[&[usize]],
) -> std::collections::HashMap<usize, usize> {
    let mut size_dict = std::collections::HashMap::new();

    for (tensor, ix) in tensors.iter().zip(ixs.iter()) {
        assert_eq!(
            tensor.ndim(),
            ix.len(),
            "Index count {} doesn't match tensor ndim {}",
            ix.len(),
            tensor.ndim()
        );

        for (dim, &label) in ix.iter().enumerate() {
            let size = tensor.shape()[dim];
            if let Some(&existing) = size_dict.get(&label) {
                assert_eq!(
                    existing, size,
                    "Inconsistent size for index {}: {} vs {}",
                    label, existing, size
                );
            } else {
                size_dict.insert(label, size);
            }
        }
    }

    size_dict
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestBackend;
    use crate::Standard;

    #[test]
    fn test_unary_einsum_preserves_explicit_backend() {
        let backend = TestBackend::new(7);
        let a = Tensor::<f32, TestBackend>::from_data_with_backend(
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            backend.clone(),
        );

        let result = einsum::<Standard<f32>, _, _>(&[&a], &[&[0, 0]], &[]);

        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.to_vec(), vec![5.0]);
        assert_eq!(result.backend(), &backend);
    }

    #[test]
    fn test_unary_backward_preserves_explicit_backend() {
        let backend = TestBackend::new(11);
        let a = Tensor::<f32, TestBackend>::from_data_with_backend(
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            backend.clone(),
        );

        let (result, grad_fn) = einsum_with_grad::<Standard<f32>, _, _>(&[&a], &[&[0, 0]], &[]);
        let grad_out =
            Tensor::<f32, TestBackend>::from_data_with_backend(&[1.0], &[], backend.clone());
        let grads = grad_fn.backward::<Standard<f32>>(&grad_out, &[&a]);

        assert_eq!(result.backend(), &backend);
        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].backend(), &backend);
        assert_eq!(grads[0].to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    }
}
