use std::io::Write;

use assert_cmd::Command;
use predicates::str::contains;
use tempfile::NamedTempFile;

fn cmd() -> Command {
    Command::cargo_bin("omeinsum").unwrap()
}

fn write_temp_json(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn test_optimize_matmul() {
    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "--pretty",
            "true",
        ])
        .assert()
        .success()
        .stdout(contains("schema_version"))
        .stdout(contains("\"expression\": \"ij,jk->ik\""));
}

#[test]
fn test_optimize_to_file() {
    let out = NamedTempFile::new().unwrap();
    let out_path = out.path().to_str().unwrap().to_string();

    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "-o",
            &out_path,
        ])
        .assert()
        .success();

    let content = std::fs::read_to_string(&out_path).unwrap();
    let topology: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(topology["schema_version"], 1);
}

#[test]
fn test_contract_with_expr_f64() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            {"shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ij,jk->ik",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "f64");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_contract_with_expr_f32_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f32",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
            {"shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ij,jk->ik",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "f32");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_contract_trace_c32() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c32",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ii->",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c32");
    assert_eq!(result["shape"], serde_json::json!([]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![3.0, 0.0]);
}

#[test]
fn test_contract_transpose_c64_row_major() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "c64",
        "order": "row_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 1.0, 2.0, -1.0, 3.0, 0.0, 4.0, 2.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ij->ji",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["dtype"], "c64");
    assert_eq!(result["order"], "row_major");
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![1.0, 1.0, 3.0, 0.0, 2.0, -1.0, 4.0, 2.0]);
}

#[test]
fn test_optimize_then_contract_pipeline() {
    let topo_file = NamedTempFile::new().unwrap();
    let topo_path = topo_file.path().to_str().unwrap().to_string();

    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=2,k=2",
            "-o",
            &topo_path,
        ])
        .assert()
        .success();

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]},
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "-t",
            &topo_path,
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_contract_scalar_output() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}
        ]
    }"#,
    );

    let output = cmd()
        .args([
            "contract",
            "--expr",
            "ii->",
            tensors.path().to_str().unwrap(),
            "--pretty",
            "false",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["shape"], serde_json::json!([]));
    let data: Vec<f64> = serde_json::from_value(result["data"].clone()).unwrap();
    assert_eq!(data, vec![2.0]);
}

#[test]
fn test_contract_both_flags_error() {
    let tensors = write_temp_json(r#"{"dtype":"f64","tensors":[]}"#);
    cmd()
        .args([
            "contract",
            "-t",
            "topo.json",
            "--expr",
            "ij->ij",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("Cannot specify both"));
}

#[test]
fn test_contract_no_flags_error() {
    let tensors = write_temp_json(r#"{"dtype":"f64","tensors":[]}"#);
    cmd()
        .args(["contract", tensors.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(contains("Must specify either"));
}

#[test]
fn test_optimize_invalid_method_error() {
    cmd()
        .args([
            "optimize",
            "ij,jk->ik",
            "--sizes",
            "i=2,j=3,k=4",
            "--method",
            "badmethod",
        ])
        .assert()
        .failure()
        .stderr(contains("Unknown method"));
}

#[test]
fn test_contract_invalid_schema_version() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 99,
        "expression": "ij,jk->ik",
        "label_map": {"i": 0, "j": 1, "k": 2},
        "size_dict": {"0": 2, "1": 2, "2": 2},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 0}}
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2, 2], "data": [1.0, 0.0, 0.0, 1.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("schema_version"));
}

#[test]
fn test_contract_shape_mismatch() {
    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2, 3], "data": [1.0, 2.0, 3.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "--expr",
            "ij->ij",
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("doesn't match shape"));
}

#[test]
fn test_contract_invalid_leaf_index_in_topology() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i->i",
        "label_map": {"i": 0},
        "size_dict": {"0": 2},
        "method": "greedy",
        "tree": {"Leaf": {"tensor_index": 9}}
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [{"shape": [2], "data": [1.0, 2.0]}]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("out of range"));
}

#[test]
fn test_contract_non_binary_topology_error() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i,j,k->i",
        "label_map": {"i": 0, "j": 1, "k": 2},
        "size_dict": {"0": 2, "1": 2, "2": 2},
        "method": "greedy",
        "tree": {
            "Node": {
                "args": [
                    {"Leaf": {"tensor_index": 0}},
                    {"Leaf": {"tensor_index": 1}},
                    {"Leaf": {"tensor_index": 2}}
                ],
                "eins": {"ixs": [[0], [1], [2]], "iy": [0]}
            }
        }
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0]},
            {"shape": [2], "data": [3.0, 4.0]},
            {"shape": [2], "data": [5.0, 6.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("binary"));
}

#[test]
fn test_contract_unknown_label_in_topology_error() {
    let topology = write_temp_json(
        r#"{
        "schema_version": 1,
        "expression": "i,j->ij",
        "label_map": {"i": 0, "j": 1},
        "size_dict": {"0": 2, "1": 2},
        "method": "greedy",
        "tree": {
            "Node": {
                "args": [
                    {"Leaf": {"tensor_index": 0}},
                    {"Leaf": {"tensor_index": 1}}
                ],
                "eins": {"ixs": [[0], [9]], "iy": [0, 9]}
            }
        }
    }"#,
    );

    let tensors = write_temp_json(
        r#"{
        "dtype": "f64",
        "order": "col_major",
        "tensors": [
            {"shape": [2], "data": [1.0, 2.0]},
            {"shape": [2], "data": [3.0, 4.0]}
        ]
    }"#,
    );

    cmd()
        .args([
            "contract",
            "-t",
            topology.path().to_str().unwrap(),
            tensors.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(contains("unknown label index"));
}
