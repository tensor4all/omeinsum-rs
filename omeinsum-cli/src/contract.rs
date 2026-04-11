use num_complex::{Complex32, Complex64};
use omeinsum::algebra::Standard;
use omeinsum::{Algebra, BackendScalar, Cpu, Tensor};

use crate::common::{
    build_explicit_einsum, load_complex_tensors, load_real_tensors, read_tensors_file,
    serialize_complex_tensor_data, serialize_real_tensor_data, write_json_output,
};
use crate::format::{Dtype, ResultFile, TensorsFile};

/// Run the contract subcommand.
pub fn run(
    tensors_path: &str,
    topology_path: Option<&str>,
    expr: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let tensors_file = read_tensors_file(tensors_path)?;

    match tensors_file.dtype {
        Dtype::F32 => run_real(
            &tensors_file,
            topology_path,
            expr,
            output,
            pretty,
            |value| value as f32,
            |value| value as f64,
        ),
        Dtype::F64 => run_real(
            &tensors_file,
            topology_path,
            expr,
            output,
            pretty,
            |value| value,
            |value| value,
        ),
        Dtype::C32 => run_complex(
            &tensors_file,
            topology_path,
            expr,
            output,
            pretty,
            |re, im| Complex32::new(re as f32, im as f32),
            |value| (value.re as f64, value.im as f64),
        ),
        Dtype::C64 => run_complex(
            &tensors_file,
            topology_path,
            expr,
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
    output: Option<&str>,
    pretty: Option<bool>,
    from_f64: fn(f64) -> T,
    to_f64: fn(T) -> f64,
) -> Result<(), String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu>,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    let tensors = load_real_tensors(tensors_file, from_f64)?;
    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let ein = build_explicit_einsum(&tensor_refs, topology_path, expr)?;
    let result = ein.execute::<Standard<T>, T, Cpu>(&tensor_refs);
    let result_data = serialize_real_tensor_data(&result, tensors_file.order, to_f64);

    let result_file = ResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        shape: result_data.shape,
        data: result_data.data,
    };
    write_json_output(&result_file, output, pretty)
}

fn run_complex<T>(
    tensors_file: &TensorsFile,
    topology_path: Option<&str>,
    expr: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
    make_complex: fn(f64, f64) -> T,
    split_complex: fn(T) -> (f64, f64),
) -> Result<(), String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu>,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    let tensors = load_complex_tensors(tensors_file, make_complex)?;
    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let ein = build_explicit_einsum(&tensor_refs, topology_path, expr)?;
    let result = ein.execute::<Standard<T>, T, Cpu>(&tensor_refs);
    let result_data = serialize_complex_tensor_data(&result, tensors_file.order, split_complex);

    let result_file = ResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        shape: result_data.shape,
        data: result_data.data,
    };
    write_json_output(&result_file, output, pretty)
}
