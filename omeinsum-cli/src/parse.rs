use std::collections::HashMap;

/// Result of parsing a flat NumPy-style einsum expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEinsum {
    /// Index labels for each input tensor (as usize indices)
    pub ixs: Vec<Vec<usize>>,
    /// Output labels (as usize indices)
    pub iy: Vec<usize>,
    /// Maps single-char labels to usize indices
    pub label_map: HashMap<char, usize>,
}

/// Parse a flat NumPy-style einsum expression like "ij,jk->ik".
///
/// Labels are single lowercase ASCII letters [a-z]. The `->` and output
/// are required (no implicit output). Whitespace is tolerated around
/// commas and `->`.
pub fn parse_flat(expr: &str) -> Result<ParsedEinsum, String> {
    let (input_side, output_side) = expr
        .split_once("->")
        .ok_or_else(|| "Missing '->' in expression".to_string())?;

    let mut label_map = HashMap::new();
    let mut next_idx = 0usize;

    let mut get_or_insert = |ch: char| -> Result<usize, String> {
        if !ch.is_ascii_lowercase() {
            return Err(format!("Invalid label '{ch}': must be a-z"));
        }

        if let Some(&idx) = label_map.get(&ch) {
            return Ok(idx);
        }

        let idx = next_idx;
        label_map.insert(ch, idx);
        next_idx += 1;
        Ok(idx)
    };

    let mut ixs = Vec::new();
    for tensor_str in input_side.split(',') {
        let tensor_str = tensor_str.trim();
        if tensor_str.is_empty() {
            return Err("Empty tensor label group".to_string());
        }

        let mut labels = Vec::new();
        for ch in tensor_str.chars() {
            labels.push(get_or_insert(ch)?);
        }
        ixs.push(labels);
    }

    let input_label_set: std::collections::HashSet<usize> =
        ixs.iter().flat_map(|labels| labels.iter().copied()).collect();

    let mut iy = Vec::new();
    for ch in output_side.trim().chars() {
        let idx = get_or_insert(ch)?;
        if !input_label_set.contains(&idx) {
            return Err(format!(
                "Output label '{ch}' not found in any input tensor"
            ));
        }
        iy.push(idx);
    }

    Ok(ParsedEinsum { ixs, iy, label_map })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_flat_matmul() {
        let result = parse_flat("ij,jk->ik").unwrap();
        assert_eq!(result.ixs.len(), 2);
        assert_eq!(result.ixs[0], vec![0, 1]);
        assert_eq!(result.ixs[1], vec![1, 2]);
        assert_eq!(result.iy, vec![0, 2]);
        assert_eq!(result.label_map[&'i'], 0);
        assert_eq!(result.label_map[&'j'], 1);
        assert_eq!(result.label_map[&'k'], 2);
    }

    #[test]
    fn test_parse_flat_three_tensors() {
        let result = parse_flat("ij,jk,kl->il").unwrap();
        assert_eq!(result.ixs.len(), 3);
        assert_eq!(result.ixs[0], vec![0, 1]);
        assert_eq!(result.ixs[1], vec![1, 2]);
        assert_eq!(result.ixs[2], vec![2, 3]);
        assert_eq!(result.iy, vec![0, 3]);
    }

    #[test]
    fn test_parse_flat_trace() {
        let result = parse_flat("ii->").unwrap();
        assert_eq!(result.ixs, vec![vec![0, 0]]);
        assert_eq!(result.iy, Vec::<usize>::new());
    }

    #[test]
    fn test_parse_flat_scalar_output() {
        let result = parse_flat("ij,ji->").unwrap();
        assert_eq!(result.ixs[0], vec![0, 1]);
        assert_eq!(result.ixs[1], vec![1, 0]);
        assert_eq!(result.iy, Vec::<usize>::new());
    }

    #[test]
    fn test_parse_flat_whitespace_tolerance() {
        let result = parse_flat("ij , jk -> ik").unwrap();
        assert_eq!(result.ixs[0], vec![0, 1]);
        assert_eq!(result.ixs[1], vec![1, 2]);
        assert_eq!(result.iy, vec![0, 2]);
    }

    #[test]
    fn test_parse_flat_no_arrow_error() {
        assert!(parse_flat("ij,jk").is_err());
    }

    #[test]
    fn test_parse_flat_invalid_char_error() {
        assert!(parse_flat("iJ,jk->ik").is_err());
    }

    #[test]
    fn test_parse_flat_output_not_subset_error() {
        assert!(parse_flat("ij,jk->iz").is_err());
    }
}
