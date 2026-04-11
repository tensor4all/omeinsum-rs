use num_complex::{Complex32, Complex64};
use omeinsum::algebra::Standard;
use omeinsum::{cost_and_gradient, Algebra, BackendScalar, Cpu, Tensor};

use crate::common::{
    build_explicit_einsum, load_complex_result_tensor, load_complex_tensors,
    load_real_result_tensor, load_real_tensors, read_tensors_file, serialize_complex_tensor_data,
    serialize_real_tensor_data, write_json_output,
};
use crate::format::{AutodiffResultFile, Dtype, GradientFile, TensorsFile};

/// Run the autodiff subcommand.
pub fn run(
    tensors_path: &str,
    topology_path: Option<&str>,
    expr: Option<&str>,
    grad_output_path: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let tensors_file = read_tensors_file(tensors_path)?;

    match tensors_file.dtype {
        Dtype::F32 => run_real(
            &tensors_file,
            topology_path,
            expr,
            grad_output_path,
            output,
            pretty,
            |value| value as f32,
            |value| value as f64,
        ),
        Dtype::F64 => run_real(
            &tensors_file,
            topology_path,
            expr,
            grad_output_path,
            output,
            pretty,
            |value| value,
            |value| value,
        ),
        Dtype::C32 => run_complex(
            &tensors_file,
            topology_path,
            expr,
            grad_output_path,
            output,
            pretty,
            |re, im| Complex32::new(re as f32, im as f32),
            |value| (value.re as f64, value.im as f64),
        ),
        Dtype::C64 => run_complex(
            &tensors_file,
            topology_path,
            expr,
            grad_output_path,
            output,
            pretty,
            Complex64::new,
            |value| (value.re, value.im),
        ),
    }
}

fn run_real<T>(
    tensors_file: &TensorsFile,
    topology_path: Option<&str>,
    expr: Option<&str>,
    grad_output_path: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
    from_f64: fn(f64) -> T,
    to_f64: fn(T) -> f64,
) -> Result<(), String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu> + Copy,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    let tensors = load_real_tensors(tensors_file, from_f64)?;
    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let ein = build_explicit_einsum(&tensor_refs, topology_path, expr)?;
    let grad_output = match grad_output_path {
        Some(path) => {
            let grad_output =
                load_real_result_tensor(path, tensors_file.dtype, tensors_file.order, from_f64)?;
            validate_grad_output_shape(&ein, &grad_output)?;
            Some(grad_output)
        }
        None => {
            if !ein.iy.is_empty() {
                return Err("Non-scalar output requires --grad-output".to_string());
            }
            None
        }
    };

    let (result, gradients) =
        cost_and_gradient::<Standard<T>, _, _>(&ein, &tensor_refs, grad_output.as_ref());

    let result_data = serialize_real_tensor_data(&result, tensors_file.order, to_f64);
    let gradients = gradients
        .iter()
        .enumerate()
        .map(|(input_index, gradient)| {
            let payload = serialize_real_tensor_data(gradient, tensors_file.order, to_f64);
            GradientFile {
                input_index,
                shape: payload.shape,
                data: payload.data,
            }
        })
        .collect();

    let result_file = AutodiffResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        result: result_data,
        gradients,
    };
    write_json_output(&result_file, output, pretty)
}

fn run_complex<T>(
    tensors_file: &TensorsFile,
    topology_path: Option<&str>,
    expr: Option<&str>,
    grad_output_path: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
    make_complex: fn(f64, f64) -> T,
    split_complex: fn(T) -> (f64, f64),
) -> Result<(), String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu> + Copy,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    let tensors = load_complex_tensors(tensors_file, make_complex)?;
    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let ein = build_explicit_einsum(&tensor_refs, topology_path, expr)?;
    let grad_output = match grad_output_path {
        Some(path) => {
            let grad_output = load_complex_result_tensor(
                path,
                tensors_file.dtype,
                tensors_file.order,
                make_complex,
            )?;
            validate_grad_output_shape(&ein, &grad_output)?;
            Some(grad_output)
        }
        None => {
            if !ein.iy.is_empty() {
                return Err("Non-scalar output requires --grad-output".to_string());
            }
            None
        }
    };

    let (result, gradients) =
        cost_and_gradient::<Standard<T>, _, _>(&ein, &tensor_refs, grad_output.as_ref());

    let result_data = serialize_complex_tensor_data(&result, tensors_file.order, split_complex);
    let gradients = gradients
        .iter()
        .enumerate()
        .map(|(input_index, gradient)| {
            let payload =
                serialize_complex_tensor_data(gradient, tensors_file.order, split_complex);
            GradientFile {
                input_index,
                shape: payload.shape,
                data: payload.data,
            }
        })
        .collect();

    let result_file = AutodiffResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        result: result_data,
        gradients,
    };
    write_json_output(&result_file, output, pretty)
}

fn validate_grad_output_shape<T>(
    ein: &omeinsum::Einsum<usize>,
    grad_output: &Tensor<T, Cpu>,
) -> Result<(), String>
where
    T: omeinsum::algebra::Scalar,
{
    let expected_shape: Vec<usize> = ein
        .iy
        .iter()
        .map(|label| {
            *ein.size_dict
                .get(label)
                .expect("output label should have a dimension size")
        })
        .collect();

    if grad_output.shape() != expected_shape.as_slice() {
        return Err(format!(
            "grad_output shape {:?} doesn't match output shape {:?}",
            grad_output.shape(),
            expected_shape
        ));
    }

    Ok(())
}
