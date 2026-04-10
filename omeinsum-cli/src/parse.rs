use std::collections::HashMap;

use omeco::{EinCode, NestedEinsum};

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

    let input_label_set: std::collections::HashSet<usize> = ixs
        .iter()
        .flat_map(|labels| labels.iter().copied())
        .collect();

    let mut iy = Vec::new();
    for ch in output_side.trim().chars() {
        let idx = get_or_insert(ch)?;
        if !input_label_set.contains(&idx) {
            return Err(format!("Output label '{ch}' not found in any input tensor"));
        }
        iy.push(idx);
    }

    Ok(ParsedEinsum { ixs, iy, label_map })
}

/// Result of parsing a parenthesized einsum expression.
#[derive(Debug, Clone)]
pub struct ParsedTree {
    /// The contraction tree.
    pub tree: NestedEinsum<usize>,
    /// Full einsum specification for leaf tensors.
    pub ixs: Vec<Vec<usize>>,
    /// Output labels.
    pub iy: Vec<usize>,
    /// Maps single-char labels to usize indices.
    #[cfg_attr(not(test), allow(dead_code))]
    pub label_map: HashMap<char, usize>,
}

/// Parse a parenthesized einsum expression like "(ij,jk),kl->il".
///
/// Parentheses encode contraction order. Each pair of parentheses
/// produces a `Node` in the `NestedEinsum` tree. Leaf tensors are numbered
/// left-to-right in order of appearance.
pub fn parse_parenthesized(expr: &str) -> Result<ParsedTree, String> {
    let (input_side, output_side) = expr
        .split_once("->")
        .ok_or_else(|| "Missing '->' in expression".to_string())?;

    let input_side = input_side.trim();
    let output_side = output_side.trim();

    let mut label_map = HashMap::new();
    let mut next_idx = 0usize;
    for ch in input_side.chars().chain(output_side.chars()) {
        if ch.is_ascii_lowercase() {
            label_map.entry(ch).or_insert_with(|| {
                let idx = next_idx;
                next_idx += 1;
                idx
            });
        } else if ch != '(' && ch != ')' && ch != ',' && ch != ' ' {
            return Err(format!("Invalid character '{ch}' in expression"));
        }
    }

    let mut iy = Vec::new();
    for ch in output_side.chars() {
        if !ch.is_ascii_lowercase() {
            return Err(format!("Invalid character '{ch}' in expression"));
        }
        iy.push(
            label_map
                .get(&ch)
                .copied()
                .ok_or_else(|| format!("Output label '{ch}' not in inputs"))?,
        );
    }

    let chars: Vec<char> = input_side.chars().collect();
    let mut pos = 0usize;
    let mut leaf_counter = 0usize;
    let mut all_ixs = Vec::new();
    let raw = parse_expr(
        &chars,
        &mut pos,
        &label_map,
        &mut leaf_counter,
        &mut all_ixs,
    )?;
    skip_spaces(&chars, &mut pos);
    if pos != chars.len() {
        return Err(format!(
            "Unexpected character '{}' at position {}",
            chars[pos], pos
        ));
    }

    let input_label_set: std::collections::HashSet<usize> = all_ixs
        .iter()
        .flat_map(|labels| labels.iter().copied())
        .collect();
    for &idx in &iy {
        if !input_label_set.contains(&idx) {
            let ch = label_map
                .iter()
                .find_map(|(&ch, &mapped)| (mapped == idx).then_some(ch))
                .unwrap();
            return Err(format!("Output label '{ch}' not in any input tensor"));
        }
    }

    let (tree, _, _) = assign_subtree_iy(raw, &all_ixs, &iy)?;

    Ok(ParsedTree {
        tree,
        ixs: all_ixs,
        iy,
        label_map,
    })
}

enum RawTree {
    Leaf(usize),
    Node(Vec<RawTree>),
}

type BuiltTree = (NestedEinsum<usize>, Vec<usize>, Vec<usize>);

fn parse_expr(
    chars: &[char],
    pos: &mut usize,
    label_map: &HashMap<char, usize>,
    leaf_counter: &mut usize,
    all_ixs: &mut Vec<Vec<usize>>,
) -> Result<RawTree, String> {
    let mut groups = Vec::new();
    let mut expect_group = true;

    loop {
        skip_spaces(chars, pos);

        if *pos >= chars.len() || chars[*pos] == ')' {
            break;
        }

        if !expect_group {
            if chars[*pos] == ',' {
                *pos += 1;
                expect_group = true;
                continue;
            }

            return Err(format!(
                "Unexpected character '{}' at position {}",
                chars[*pos], *pos
            ));
        }

        match chars[*pos] {
            '(' => {
                *pos += 1;
                let subtree = parse_expr(chars, pos, label_map, leaf_counter, all_ixs)?;
                skip_spaces(chars, pos);
                if *pos >= chars.len() || chars[*pos] != ')' {
                    return Err("Unbalanced parentheses: missing ')'".to_string());
                }
                *pos += 1;
                groups.push(subtree);
                expect_group = false;
            }
            ',' => return Err("Empty tensor label group".to_string()),
            ch if ch.is_ascii_lowercase() => {
                let mut labels = Vec::new();
                while *pos < chars.len() && chars[*pos].is_ascii_lowercase() {
                    let ch = chars[*pos];
                    labels.push(
                        label_map
                            .get(&ch)
                            .copied()
                            .ok_or_else(|| format!("Unknown label '{ch}'"))?,
                    );
                    *pos += 1;
                }

                let tensor_idx = *leaf_counter;
                *leaf_counter += 1;
                all_ixs.push(labels);
                groups.push(RawTree::Leaf(tensor_idx));
                expect_group = false;
            }
            ch => {
                return Err(format!(
                    "Unexpected character '{}' at position {}",
                    ch, *pos
                ));
            }
        }
    }

    if groups.is_empty() || expect_group {
        return Err("Empty expression group".to_string());
    }

    if groups.len() == 1 {
        Ok(groups.into_iter().next().unwrap())
    } else {
        Ok(RawTree::Node(groups))
    }
}

fn skip_spaces(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos] == ' ' {
        *pos += 1;
    }
}

fn assign_subtree_iy(
    raw: RawTree,
    all_ixs: &[Vec<usize>],
    global_iy: &[usize],
) -> Result<BuiltTree, String> {
    match raw {
        RawTree::Leaf(idx) => Ok((NestedEinsum::leaf(idx), all_ixs[idx].clone(), vec![idx])),
        RawTree::Node(children) => {
            if children.len() != 2 {
                return Err(format!(
                    "Node must have exactly 2 children, got {}",
                    children.len()
                ));
            }

            let mut built_children = Vec::with_capacity(2);
            let mut child_outputs = Vec::with_capacity(2);
            let mut leaf_ids = Vec::new();

            for child in children {
                let (tree, output_labels, child_leaf_ids) =
                    assign_subtree_iy(child, all_ixs, global_iy)?;
                built_children.push(tree);
                child_outputs.push(output_labels);
                leaf_ids.extend(child_leaf_ids);
            }

            let leaf_id_set: std::collections::HashSet<usize> = leaf_ids.iter().copied().collect();
            let external_labels: std::collections::HashSet<usize> = all_ixs
                .iter()
                .enumerate()
                .filter(|(idx, _)| !leaf_id_set.contains(idx))
                .flat_map(|(_, labels)| labels.iter().copied())
                .collect();
            let keep: std::collections::HashSet<usize> = global_iy
                .iter()
                .copied()
                .chain(external_labels.iter().copied())
                .collect();

            let mut node_iy = Vec::new();
            for label in child_outputs.iter().flatten().copied() {
                if keep.contains(&label) && !node_iy.contains(&label) {
                    node_iy.push(label);
                }
            }

            let eins = EinCode::new(child_outputs, node_iy.clone());
            Ok((NestedEinsum::node(built_children, eins), node_iy, leaf_ids))
        }
    }
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

    #[test]
    fn test_parse_paren_simple() {
        let result = parse_parenthesized("(ij,jk),kl->il").unwrap();
        assert_eq!(result.ixs.len(), 3);
        assert_eq!(
            result.iy,
            vec![result.label_map[&'i'], result.label_map[&'l']]
        );

        match &result.tree {
            NestedEinsum::Node { args, .. } => {
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], NestedEinsum::Node { .. }));
                assert!(matches!(&args[1], NestedEinsum::Leaf { tensor_index: 2 }));
            }
            _ => panic!("Expected Node at root"),
        }
    }

    #[test]
    fn test_parse_paren_flat_pair() {
        let result = parse_parenthesized("ij,jk->ik").unwrap();
        match &result.tree {
            NestedEinsum::Node { args, .. } => {
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], NestedEinsum::Leaf { tensor_index: 0 }));
                assert!(matches!(&args[1], NestedEinsum::Leaf { tensor_index: 1 }));
            }
            _ => panic!("Expected Node at root"),
        }
    }

    #[test]
    fn test_parse_paren_subtree_iy() {
        let result = parse_parenthesized("(ij,jk),kl->il").unwrap();
        let i = result.label_map[&'i'];
        let k = result.label_map[&'k'];

        match &result.tree {
            NestedEinsum::Node { args, .. } => match &args[0] {
                NestedEinsum::Node { eins, .. } => {
                    assert!(eins.iy.contains(&i));
                    assert!(eins.iy.contains(&k));
                    let j = result.label_map[&'j'];
                    assert!(!eins.iy.contains(&j));
                }
                _ => panic!("Expected inner Node"),
            },
            _ => panic!("Expected outer Node"),
        }
    }

    #[test]
    fn test_parse_paren_single_tensor() {
        let result = parse_parenthesized("ij->ji").unwrap();
        match &result.tree {
            NestedEinsum::Leaf { tensor_index: 0 } => {}
            _ => panic!("Expected Leaf for single tensor"),
        }
    }

    #[test]
    fn test_parse_paren_unbalanced_error() {
        assert!(parse_parenthesized("(ij,jk->ik").is_err());
    }

    #[test]
    fn test_parse_paren_empty_group_error() {
        assert!(parse_parenthesized("(,jk),kl->il").is_err());
    }
}
