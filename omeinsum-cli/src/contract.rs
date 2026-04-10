use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use num_complex::{Complex32, Complex64};
use omeco::NestedEinsum;
use omeinsum::algebra::Standard;
use omeinsum::{Algebra, BackendScalar, Cpu, Einsum, Tensor};

use crate::format::{
    col_to_row_major, row_to_col_major, Dtype, ResultFile, StorageOrder, TensorsFile,
    TopologyFile,
};
use crate::parse::{parse_flat, parse_parenthesized};

/// Run the contract subcommand.
pub fn run(
    tensors_path: &str,
    topology_path: Option<&str>,
    expr: Option<&str>,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    match (topology_path, expr) {
        (Some(_), Some(_)) => return Err("Cannot specify both -t and --expr".to_string()),
        (None, None) => return Err("Must specify either -t or --expr".to_string()),
        _ => {}
    }

    let tensors_json = std::fs::read_to_string(tensors_path)
        .map_err(|err| format!("Failed to read '{tensors_path}': {err}"))?;
    let tensors_file: TensorsFile = serde_json::from_str(&tensors_json)
        .map_err(|err| format!("Failed to parse tensors JSON: {err}"))?;

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
    let tensors: Vec<Tensor<T, Cpu>> = tensors_file
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
        .collect::<Result<_, String>>()?;

    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let result = match topology_path {
        Some(path) => execute_with_topology(&tensor_refs, path)?,
        None => execute_with_expr(&tensor_refs, expr.unwrap())?,
    };

    let mut data: Vec<f64> = result.to_vec().into_iter().map(to_f64).collect();
    if tensors_file.order == StorageOrder::RowMajor {
        data = col_to_row_major(&data, result.shape());
    }

    let result_file = ResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        shape: result.shape().to_vec(),
        data,
    };
    write_output(&result_file, output, pretty)
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
    let tensors: Vec<Tensor<T, Cpu>> = tensors_file
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
        .collect::<Result<_, String>>()?;

    let tensor_refs: Vec<&Tensor<T, Cpu>> = tensors.iter().collect();
    let result = match topology_path {
        Some(path) => execute_with_topology(&tensor_refs, path)?,
        None => execute_with_expr(&tensor_refs, expr.unwrap())?,
    };

    let mut data = complex_to_interleaved(&result.to_vec(), split_complex);
    if tensors_file.order == StorageOrder::RowMajor {
        data = col_to_row_major_interleaved(&data, result.shape());
    }

    let result_file = ResultFile {
        dtype: tensors_file.dtype,
        order: tensors_file.order,
        shape: result.shape().to_vec(),
        data,
    };
    write_output(&result_file, output, pretty)
}

fn execute_with_topology<T>(
    tensors: &[&Tensor<T, Cpu>],
    topology_path: &str,
) -> Result<Tensor<T, Cpu>, String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu>,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
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

    let mut ein = Einsum::new(parsed.ixs, parsed.iy, size_dict);
    ein.set_contraction_tree(topology.tree);
    Ok(ein.execute::<Standard<T>, T, Cpu>(tensors))
}

fn execute_with_expr<T>(tensors: &[&Tensor<T, Cpu>], expr: &str) -> Result<Tensor<T, Cpu>, String>
where
    T: omeinsum::algebra::Scalar + BackendScalar<Cpu>,
    Standard<T>: Algebra<Scalar = T, Index = u32>,
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
    Ok(ein.execute::<Standard<T>, T, Cpu>(tensors))
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

fn write_output(
    result: &ResultFile,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let use_pretty = pretty.unwrap_or_else(|| io::stdout().is_terminal());
    let json = if use_pretty {
        serde_json::to_string_pretty(result)
    } else {
        serde_json::to_string(result)
    }
    .map_err(|err| format!("JSON serialization failed: {err}"))?;

    match output {
        Some(path) => std::fs::write(path, &json)
            .map_err(|err| format!("Failed to write '{path}': {err}"))?,
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(handle, "{json}")
                .map_err(|err| format!("Failed to write stdout: {err}"))?;
        }
    }

    Ok(())
}
