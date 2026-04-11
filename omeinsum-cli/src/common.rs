use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use omeco::NestedEinsum;
use omeinsum::algebra::Scalar;
use omeinsum::{BackendScalar, Cpu, Einsum, Tensor};
use serde::Serialize;

use crate::format::{
    col_to_row_major, row_to_col_major, Dtype, ResultFile, StorageOrder, TensorDataFile,
    TensorsFile, TopologyFile,
};
use crate::parse::{parse_flat, parse_parenthesized};

pub(crate) fn validate_execution_source(
    topology_path: Option<&str>,
    expr: Option<&str>,
) -> Result<(), String> {
    match (topology_path, expr) {
        (Some(_), Some(_)) => Err("Cannot specify both -t and --expr".to_string()),
        (None, None) => Err("Must specify either -t or --expr".to_string()),
        _ => Ok(()),
    }
}

pub(crate) fn read_tensors_file(path: &str) -> Result<TensorsFile, String> {
    let json =
        std::fs::read_to_string(path).map_err(|err| format!("Failed to read '{path}': {err}"))?;
    serde_json::from_str(&json).map_err(|err| format!("Failed to parse tensors JSON: {err}"))
}

pub(crate) fn load_real_tensors<T>(
    tensors_file: &TensorsFile,
    from_f64: fn(f64) -> T,
) -> Result<Vec<Tensor<T, Cpu>>, String>
where
    T: Scalar + BackendScalar<Cpu>,
{
    tensors_file
        .tensors
        .iter()
        .map(|entry| {
            validate_shape(&entry.data, &entry.shape, false)?;
            let col_major_data = match tensors_file.order {
                StorageOrder::RowMajor => row_to_col_major(&entry.data, &entry.shape),
                StorageOrder::ColMajor => entry.data.clone(),
            };
            let data: Vec<T> = col_major_data.into_iter().map(from_f64).collect();
            Ok(Tensor::<T, Cpu>::from_data(&data, &entry.shape))
        })
        .collect()
}

pub(crate) fn load_complex_tensors<T>(
    tensors_file: &TensorsFile,
    make_complex: fn(f64, f64) -> T,
) -> Result<Vec<Tensor<T, Cpu>>, String>
where
    T: Scalar + BackendScalar<Cpu>,
{
    tensors_file
        .tensors
        .iter()
        .map(|entry| {
            validate_shape(&entry.data, &entry.shape, true)?;
            let col_major_data = match tensors_file.order {
                StorageOrder::RowMajor => row_to_col_major_interleaved(&entry.data, &entry.shape),
                StorageOrder::ColMajor => entry.data.clone(),
            };
            let data = interleaved_to_complex(&col_major_data, make_complex);
            Ok(Tensor::<T, Cpu>::from_data(&data, &entry.shape))
        })
        .collect()
}

pub(crate) fn load_real_result_tensor<T>(
    path: &str,
    expected_dtype: Dtype,
    expected_order: StorageOrder,
    from_f64: fn(f64) -> T,
) -> Result<Tensor<T, Cpu>, String>
where
    T: Scalar + BackendScalar<Cpu>,
{
    let result_file = read_result_file(path)?;
    validate_result_metadata(&result_file, path, expected_dtype, expected_order)?;
    validate_shape(&result_file.data, &result_file.shape, false)?;

    let col_major_data = match result_file.order {
        StorageOrder::RowMajor => row_to_col_major(&result_file.data, &result_file.shape),
        StorageOrder::ColMajor => result_file.data,
    };
    let data: Vec<T> = col_major_data.into_iter().map(from_f64).collect();
    Ok(Tensor::<T, Cpu>::from_data(&data, &result_file.shape))
}

pub(crate) fn load_complex_result_tensor<T>(
    path: &str,
    expected_dtype: Dtype,
    expected_order: StorageOrder,
    make_complex: fn(f64, f64) -> T,
) -> Result<Tensor<T, Cpu>, String>
where
    T: Scalar + BackendScalar<Cpu>,
{
    let result_file = read_result_file(path)?;
    validate_result_metadata(&result_file, path, expected_dtype, expected_order)?;
    validate_shape(&result_file.data, &result_file.shape, true)?;

    let col_major_data = match result_file.order {
        StorageOrder::RowMajor => {
            row_to_col_major_interleaved(&result_file.data, &result_file.shape)
        }
        StorageOrder::ColMajor => result_file.data,
    };
    let data = interleaved_to_complex(&col_major_data, make_complex);
    Ok(Tensor::<T, Cpu>::from_data(&data, &result_file.shape))
}

pub(crate) fn serialize_real_tensor_data<T>(
    tensor: &Tensor<T, Cpu>,
    order: StorageOrder,
    to_f64: fn(T) -> f64,
) -> TensorDataFile
where
    T: Scalar + BackendScalar<Cpu> + Copy,
{
    let mut data: Vec<f64> = tensor.to_vec().into_iter().map(to_f64).collect();
    if order == StorageOrder::RowMajor {
        data = col_to_row_major(&data, tensor.shape());
    }

    TensorDataFile {
        shape: tensor.shape().to_vec(),
        data,
    }
}

pub(crate) fn serialize_complex_tensor_data<T>(
    tensor: &Tensor<T, Cpu>,
    order: StorageOrder,
    split_complex: fn(T) -> (f64, f64),
) -> TensorDataFile
where
    T: Scalar + BackendScalar<Cpu> + Copy,
{
    let mut data = complex_to_interleaved(&tensor.to_vec(), split_complex);
    if order == StorageOrder::RowMajor {
        data = col_to_row_major_interleaved(&data, tensor.shape());
    }

    TensorDataFile {
        shape: tensor.shape().to_vec(),
        data,
    }
}

pub(crate) fn build_explicit_einsum<T>(
    tensors: &[&Tensor<T, Cpu>],
    topology_path: Option<&str>,
    expr: Option<&str>,
) -> Result<Einsum<usize>, String>
where
    T: Scalar,
{
    validate_execution_source(topology_path, expr)?;

    match topology_path {
        Some(path) => load_einsum_from_topology(tensors, path),
        None => load_einsum_from_expr(tensors, expr.expect("validated expression source")),
    }
}

pub(crate) fn write_json_output<S: Serialize>(
    value: &S,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let use_pretty = pretty.unwrap_or_else(|| io::stdout().is_terminal());
    let json = if use_pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .map_err(|err| format!("JSON serialization failed: {err}"))?;

    match output {
        Some(path) => {
            std::fs::write(path, &json).map_err(|err| format!("Failed to write '{path}': {err}"))?
        }
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(handle, "{json}").map_err(|err| format!("Failed to write stdout: {err}"))?;
        }
    }

    Ok(())
}

fn read_result_file(path: &str) -> Result<ResultFile, String> {
    let json =
        std::fs::read_to_string(path).map_err(|err| format!("Failed to read '{path}': {err}"))?;
    serde_json::from_str(&json).map_err(|err| format!("Failed to parse result JSON: {err}"))
}

fn validate_result_metadata(
    result_file: &ResultFile,
    path: &str,
    expected_dtype: Dtype,
    expected_order: StorageOrder,
) -> Result<(), String> {
    if result_file.dtype != expected_dtype {
        return Err(format!(
            "grad_output dtype {:?} in '{path}' doesn't match tensors dtype {:?}",
            result_file.dtype, expected_dtype
        ));
    }
    if result_file.order != expected_order {
        return Err(format!(
            "grad_output order {:?} in '{path}' doesn't match tensors order {:?}",
            result_file.order, expected_order
        ));
    }
    Ok(())
}

fn load_einsum_from_topology<T>(
    tensors: &[&Tensor<T, Cpu>],
    topology_path: &str,
) -> Result<Einsum<usize>, String>
where
    T: Scalar,
{
    let topology_json = std::fs::read_to_string(topology_path)
        .map_err(|err| format!("Failed to read '{topology_path}': {err}"))?;
    let topology: TopologyFile = serde_json::from_str(&topology_json)
        .map_err(|err| format!("Failed to parse topology JSON: {err}"))?;

    if topology.schema_version != 1 {
        return Err(format!(
            "Unsupported schema_version {}, expected 1",
            topology.schema_version
        ));
    }

    let size_dict: HashMap<usize, usize> = topology
        .size_dict
        .iter()
        .map(|(key, value)| {
            key.parse::<usize>()
                .map(|idx| (idx, *value))
                .map_err(|_| format!("Invalid size_dict key '{key}'"))
        })
        .collect::<Result<_, _>>()?;
    validate_topology_tree(&topology.tree, tensors.len(), &size_dict)?;

    let parsed = parse_flat(&topology.expression)?;
    if tensors.len() != parsed.ixs.len() {
        return Err(format!(
            "Expression expects {} tensors but received {}",
            parsed.ixs.len(),
            tensors.len()
        ));
    }
    validate_size_dict_labels(&parsed.ixs, &parsed.iy, &size_dict)?;

    let mut ein = Einsum::new(parsed.ixs, parsed.iy, size_dict);
    ein.set_contraction_tree(topology.tree);
    Ok(ein)
}

fn load_einsum_from_expr<T>(
    tensors: &[&Tensor<T, Cpu>],
    expr: &str,
) -> Result<Einsum<usize>, String>
where
    T: Scalar,
{
    let parsed = parse_parenthesized(expr)?;
    if tensors.len() != parsed.ixs.len() {
        return Err(format!(
            "Expression expects {} tensors but received {}",
            parsed.ixs.len(),
            tensors.len()
        ));
    }

    let mut size_dict = HashMap::new();
    for (tensor, labels) in tensors.iter().zip(parsed.ixs.iter()) {
        if tensor.ndim() != labels.len() {
            return Err(format!(
                "Tensor has {} dims but expression specifies {} labels",
                tensor.ndim(),
                labels.len()
            ));
        }

        for (dim, label) in labels.iter().copied().enumerate() {
            let size = tensor.shape()[dim];
            if let Some(existing) = size_dict.insert(label, size) {
                if existing != size {
                    return Err(format!(
                        "Inconsistent size for label index {}: {} vs {}",
                        label, existing, size
                    ));
                }
            }
        }
    }

    let mut ein = Einsum::new(parsed.ixs, parsed.iy, size_dict);
    ein.set_contraction_tree(parsed.tree);
    Ok(ein)
}

fn validate_shape(data: &[f64], shape: &[usize], is_complex: bool) -> Result<(), String> {
    let numel: usize = shape.iter().product();
    let expected_len = if is_complex { numel * 2 } else { numel };
    if data.len() != expected_len {
        return Err(format!(
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        ));
    }
    Ok(())
}

fn validate_size_dict_labels(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &HashMap<usize, usize>,
) -> Result<(), String> {
    for &label in ixs.iter().flatten().chain(iy.iter()) {
        if !size_dict.contains_key(&label) {
            return Err(format!(
                "Missing size for label index {} referenced by topology expression",
                label
            ));
        }
    }

    Ok(())
}

fn validate_topology_tree(
    tree: &NestedEinsum<usize>,
    num_tensors: usize,
    size_dict: &HashMap<usize, usize>,
) -> Result<(), String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            if *tensor_index >= num_tensors {
                return Err(format!(
                    "Topology leaf tensor_index {} out of range for {} tensors",
                    tensor_index, num_tensors
                ));
            }
            Ok(())
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return Err(format!(
                    "Topology tree must be binary, found node with {} children",
                    args.len()
                ));
            }

            for label in eins.ixs.iter().flatten().chain(eins.iy.iter()) {
                if !size_dict.contains_key(label) {
                    return Err(format!("Topology references unknown label index {}", label));
                }
            }

            for arg in args {
                validate_topology_tree(arg, num_tensors, size_dict)?;
            }
            Ok(())
        }
    }
}

fn interleaved_to_complex<T>(data: &[f64], make_complex: fn(f64, f64) -> T) -> Vec<T> {
    data.chunks_exact(2)
        .map(|pair| make_complex(pair[0], pair[1]))
        .collect()
}

fn complex_to_interleaved<T>(data: &[T], split_complex: fn(T) -> (f64, f64)) -> Vec<f64>
where
    T: Copy,
{
    let mut result = Vec::with_capacity(data.len() * 2);
    for value in data.iter().copied() {
        let (re, im) = split_complex(value);
        result.push(re);
        result.push(im);
    }
    result
}

fn row_to_col_major_interleaved(data: &[f64], shape: &[usize]) -> Vec<f64> {
    if shape.len() <= 1 {
        return data.to_vec();
    }

    let indices = compute_row_to_col_indices(shape);
    let mut result = vec![0.0; data.len()];
    for (row_idx, col_idx) in indices.into_iter().enumerate() {
        result[col_idx * 2] = data[row_idx * 2];
        result[col_idx * 2 + 1] = data[row_idx * 2 + 1];
    }
    result
}

fn col_to_row_major_interleaved(data: &[f64], shape: &[usize]) -> Vec<f64> {
    if shape.len() <= 1 {
        return data.to_vec();
    }

    let indices = compute_row_to_col_indices(shape);
    let mut result = vec![0.0; data.len()];
    for (row_idx, col_idx) in indices.into_iter().enumerate() {
        result[row_idx * 2] = data[col_idx * 2];
        result[row_idx * 2 + 1] = data[col_idx * 2 + 1];
    }
    result
}

fn compute_row_to_col_indices(shape: &[usize]) -> Vec<usize> {
    let numel: usize = shape.iter().product();
    let col_strides = col_major_strides(shape);

    (0..numel)
        .map(|row_idx| {
            let coords = row_major_coords(row_idx, shape);
            coords
                .into_iter()
                .enumerate()
                .map(|(dim, coord)| coord * col_strides[dim])
                .sum()
        })
        .collect()
}

fn row_major_coords(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for dim in (0..shape.len()).rev() {
        coords[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
    coords
}

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dim in 1..shape.len() {
        strides[dim] = strides[dim - 1] * shape[dim - 1];
    }
    strides
}
