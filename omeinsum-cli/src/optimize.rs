use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use crate::format::TopologyFile;
use crate::parse::parse_flat;

/// Run the optimize subcommand.
pub fn run(
    expr: &str,
    sizes_str: &str,
    method: &str,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let parsed = parse_flat(expr)?;
    let size_dict = parse_sizes(sizes_str, &parsed.label_map)?;

    let mut ein = omeinsum::Einsum::new(parsed.ixs.clone(), parsed.iy.clone(), size_dict.clone());
    match method {
        "greedy" => {
            ein.optimize_greedy();
        }
        "treesa" => {
            ein.optimize_treesa();
        }
        _ => {
            return Err(format!(
                "Unknown method '{method}': expected 'greedy' or 'treesa'"
            ));
        }
    }

    let tree = ein
        .contraction_tree()
        .ok_or_else(|| "Optimization produced no contraction tree".to_string())?
        .clone();

    let label_map = parsed
        .label_map
        .iter()
        .map(|(label, idx)| (label.to_string(), *idx))
        .collect();
    let size_dict = size_dict
        .iter()
        .map(|(idx, size)| (idx.to_string(), *size))
        .collect();

    let topology = TopologyFile {
        schema_version: 1,
        expression: expr.to_string(),
        label_map,
        size_dict,
        method: method.to_string(),
        tree,
    };

    let use_pretty = pretty.unwrap_or_else(|| io::stdout().is_terminal());
    let json = if use_pretty {
        serde_json::to_string_pretty(&topology)
    } else {
        serde_json::to_string(&topology)
    }
    .map_err(|err| format!("JSON serialization failed: {err}"))?;

    match output {
        Some(path) => {
            std::fs::write(path, &json).map_err(|err| format!("Failed to write '{path}': {err}"))?
        }
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            writeln!(handle, "{json}")
                .map_err(|err| format!("Failed to write to stdout: {err}"))?;
        }
    }

    Ok(())
}

/// Parse "--sizes" string like "i=2,j=3,k=4" into a size_dict.
fn parse_sizes(
    sizes_str: &str,
    label_map: &HashMap<char, usize>,
) -> Result<HashMap<usize, usize>, String> {
    let mut size_dict = HashMap::new();

    for pair in sizes_str.split(',') {
        let pair = pair.trim();
        let (label_str, size_str) = pair
            .split_once('=')
            .ok_or_else(|| format!("Invalid size spec '{pair}': expected 'label=size'"))?;

        let label_str = label_str.trim();
        let size_str = size_str.trim();
        if label_str.len() != 1 {
            return Err(format!(
                "Label must be a single character, got '{label_str}'"
            ));
        }

        let ch = label_str.chars().next().unwrap();
        let idx = label_map
            .get(&ch)
            .ok_or_else(|| format!("Label '{ch}' not found in expression"))?;
        let size = size_str
            .parse::<usize>()
            .map_err(|_| format!("Invalid size '{size_str}' for label '{ch}'"))?;
        if size == 0 {
            return Err(format!("Size for label '{ch}' must be >= 1"));
        }

        size_dict.insert(*idx, size);
    }

    Ok(size_dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sizes() {
        let label_map: HashMap<char, usize> = [('i', 0), ('j', 1), ('k', 2)].into_iter().collect();
        let sizes = parse_sizes("i=2,j=3,k=4", &label_map).unwrap();
        assert_eq!(sizes[&0], 2);
        assert_eq!(sizes[&1], 3);
        assert_eq!(sizes[&2], 4);
    }

    #[test]
    fn test_parse_sizes_unknown_label() {
        let label_map: HashMap<char, usize> = [('i', 0)].into_iter().collect();
        assert!(parse_sizes("z=5", &label_map).is_err());
    }

    #[test]
    fn test_parse_sizes_zero_error() {
        let label_map: HashMap<char, usize> = [('i', 0)].into_iter().collect();
        assert!(parse_sizes("i=0", &label_map).is_err());
    }
}
