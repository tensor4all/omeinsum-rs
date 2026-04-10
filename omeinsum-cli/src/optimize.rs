use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};

use omeco::{optimize_code, EinCode, GreedyMethod, TreeSA};

use crate::format::TopologyFile;
use crate::parse::parse_flat;

/// Method-specific hyperparameters passed from the CLI.
#[derive(Default)]
pub struct OptimizeParams {
    // greedy
    pub alpha: Option<f64>,
    pub temperature: Option<f64>,
    // treesa
    pub ntrials: Option<usize>,
    pub niters: Option<usize>,
    pub sc_target: Option<f64>,
    pub betas: Option<String>,
    pub tc_weight: Option<f64>,
    pub sc_weight: Option<f64>,
    pub rw_weight: Option<f64>,
}

/// Parse a "start:step:stop" string into a Vec of betas.
fn parse_betas(s: &str) -> Result<Vec<f64>, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(format!(
            "Invalid betas '{s}': expected 'start:step:stop' (e.g., '0.01:0.05:15.0')"
        ));
    }
    let start: f64 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid betas start '{}'", parts[0]))?;
    let step: f64 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid betas step '{}'", parts[1]))?;
    let stop: f64 = parts[2]
        .parse()
        .map_err(|_| format!("Invalid betas stop '{}'", parts[2]))?;
    if step <= 0.0 {
        return Err("Betas step must be positive".to_string());
    }
    let mut betas = Vec::new();
    let mut v = start;
    while v <= stop {
        betas.push(v);
        v += step;
    }
    if betas.is_empty() {
        return Err(format!(
            "Betas '{s}' produced an empty schedule (start > stop?)"
        ));
    }
    Ok(betas)
}

/// Run the optimize subcommand.
pub fn run(
    expr: &str,
    sizes_str: &str,
    method: &str,
    params: OptimizeParams,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    let parsed = parse_flat(expr)?;
    let size_dict = parse_sizes(sizes_str, &parsed.label_map)?;

    let code = EinCode::new(parsed.ixs.clone(), parsed.iy.clone());
    let tree = match method {
        "greedy" => {
            let alpha = params.alpha.unwrap_or(0.0);
            let temperature = params.temperature.unwrap_or(0.0);
            let optimizer = GreedyMethod::new(alpha, temperature);
            optimize_code(&code, &size_dict, &optimizer)
        }
        "treesa" => {
            let mut optimizer = TreeSA::default();
            if let Some(ntrials) = params.ntrials {
                optimizer.ntrials = ntrials;
            }
            if let Some(niters) = params.niters {
                optimizer.niters = niters;
            }
            if let Some(betas_str) = params.betas {
                optimizer.betas = parse_betas(&betas_str)?;
            }
            let mut score = optimizer.score;
            if let Some(v) = params.sc_target {
                score.sc_target = v;
            }
            if let Some(v) = params.tc_weight {
                score.tc_weight = v;
            }
            if let Some(v) = params.sc_weight {
                score.sc_weight = v;
            }
            if let Some(v) = params.rw_weight {
                score.rw_weight = v;
            }
            optimizer.score = score;
            optimize_code(&code, &size_dict, &optimizer)
        }
        _ => {
            return Err(format!(
                "Unknown method '{method}': expected 'greedy' or 'treesa'"
            ));
        }
    };

    let tree =
        tree.ok_or_else(|| "Optimization produced no contraction tree".to_string())?;

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
