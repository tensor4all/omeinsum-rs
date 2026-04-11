use std::collections::HashMap;

use omeco::NestedEinsum;
use serde::{Deserialize, Serialize};

/// Storage order for tensor data in JSON.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageOrder {
    #[default]
    ColMajor,
    RowMajor,
}

/// Scalar data type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Dtype {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f64")]
    F64,
    #[serde(rename = "c32")]
    C32,
    #[serde(rename = "c64")]
    C64,
}

/// A single tensor entry in the tensors JSON file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// The tensors JSON file schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorsFile {
    pub dtype: Dtype,
    #[serde(default)]
    pub order: StorageOrder,
    pub tensors: Vec<TensorEntry>,
}

/// The topology JSON file schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyFile {
    pub schema_version: u32,
    pub expression: String,
    pub label_map: HashMap<String, usize>,
    pub size_dict: HashMap<String, usize>,
    pub method: String,
    pub tree: NestedEinsum<usize>,
}

/// The result JSON file schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFile {
    pub dtype: Dtype,
    pub order: StorageOrder,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// Tensor payload used inside higher-level result bundles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDataFile {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// A gradient tensor paired with its input position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFile {
    pub input_index: usize,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// The autodiff JSON output schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutodiffResultFile {
    pub dtype: Dtype,
    pub order: StorageOrder,
    pub result: TensorDataFile,
    pub gradients: Vec<GradientFile>,
}

/// Convert row-major flat data to column-major for a given shape.
pub fn row_to_col_major(data: &[f64], shape: &[usize]) -> Vec<f64> {
    if shape.len() <= 1 {
        return data.to_vec();
    }

    let numel: usize = shape.iter().product();
    let mut result = vec![0.0; numel];
    let col_strides = col_major_strides(shape);

    for (row_idx, value) in data.iter().copied().enumerate() {
        let coords = row_major_coords(row_idx, shape);
        let mut col_idx = 0usize;
        for (dim, coord) in coords.into_iter().enumerate() {
            col_idx += coord * col_strides[dim];
        }
        result[col_idx] = value;
    }

    result
}

/// Convert column-major flat data to row-major for a given shape.
pub fn col_to_row_major(data: &[f64], shape: &[usize]) -> Vec<f64> {
    if shape.len() <= 1 {
        return data.to_vec();
    }

    let numel: usize = shape.iter().product();
    let mut result = vec![0.0; numel];
    let row_strides = row_major_strides(shape);

    for (col_idx, value) in data.iter().copied().enumerate() {
        let coords = col_major_coords(col_idx, shape);
        let mut row_idx = 0usize;
        for (dim, coord) in coords.into_iter().enumerate() {
            row_idx += coord * row_strides[dim];
        }
        result[row_idx] = value;
    }

    result
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

fn row_major_coords(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for dim in (0..shape.len()).rev() {
        coords[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
    coords
}

fn col_major_coords(mut linear_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for dim in 0..shape.len() {
        coords[dim] = linear_idx % shape[dim];
        linear_idx /= shape[dim];
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensors_file_roundtrip() {
        let tf = TensorsFile {
            dtype: Dtype::F64,
            order: StorageOrder::ColMajor,
            tensors: vec![TensorEntry {
                shape: vec![2, 3],
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
        };
        let json = serde_json::to_string(&tf).unwrap();
        let parsed: TensorsFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.dtype, Dtype::F64);
        assert_eq!(parsed.order, StorageOrder::ColMajor);
        assert_eq!(parsed.tensors[0].shape, vec![2, 3]);
    }

    #[test]
    fn test_tensors_file_default_order() {
        let json = r#"{"dtype":"f64","tensors":[{"shape":[2],"data":[1.0,2.0]}]}"#;
        let parsed: TensorsFile = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.order, StorageOrder::ColMajor);
    }

    #[test]
    fn test_result_file_roundtrip() {
        let rf = ResultFile {
            dtype: Dtype::F32,
            order: StorageOrder::RowMajor,
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let json = serde_json::to_string(&rf).unwrap();
        let parsed: ResultFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.dtype, Dtype::F32);
        assert_eq!(parsed.order, StorageOrder::RowMajor);
    }

    #[test]
    fn test_row_to_col_major_2x3() {
        let row_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_data = row_to_col_major(&row_data, &[2, 3]);
        assert_eq!(col_data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_col_to_row_major_2x3() {
        let col_data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let row_data = col_to_row_major(&col_data, &[2, 3]);
        assert_eq!(row_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_row_col_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [2, 2, 2];
        let col = row_to_col_major(&original, &shape);
        let back = col_to_row_major(&col, &shape);
        assert_eq!(back, original);
    }

    #[test]
    fn test_scalar_no_conversion() {
        let data = vec![42.0];
        assert_eq!(row_to_col_major(&data, &[]), data);
        assert_eq!(col_to_row_major(&data, &[]), data);
    }

    #[test]
    fn test_1d_no_conversion() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(row_to_col_major(&data, &[3]), data);
        assert_eq!(col_to_row_major(&data, &[3]), data);
    }
}
