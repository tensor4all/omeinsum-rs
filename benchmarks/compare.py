#!/usr/bin/env python3
"""Compare Julia and Rust benchmark timings for omeinsum-rs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "benchmarks" / "data"
CRITERION_DIR = ROOT / "target" / "criterion"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_criterion_estimate(group: str, benchmark: str, scenario: str):
    path = CRITERION_DIR / group / benchmark / scenario / "new" / "estimates.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("median", {}).get("point_estimate")


def load_julia_binary():
    path = DATA_DIR / "binary_timings.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `make bench-julia` first.")
        return None
    return load_json(path)


def load_julia_network():
    path = DATA_DIR / "network_timings.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run `make bench-julia` first.")
        return None
    return load_json(path)


def load_rust_heavy_network():
    path = DATA_DIR / "rust_network_timings.json"
    if not path.exists():
        print(
            f"ERROR: {path} not found. Run `make bench-network` first so the 3reg_150 one-shot timing is generated."
        )
        return None
    return load_json(path)


def fmt_ns(value):
    return f"{value:>12.0f}" if value is not None else "         N/A"


def fmt_ratio(julia_ns, rust_ns):
    if not julia_ns or not rust_ns:
        return "     N/A"
    return f"{julia_ns / rust_ns:>8.3f}"


def fmt_drop(julia_ns, rust_ns):
    if not julia_ns or not rust_ns:
        return "    N/A"
    drop = max((rust_ns - julia_ns) / julia_ns, 0.0)
    return f"{drop * 100:>7.2f}%"


def main():
    julia_binary = load_julia_binary()
    julia_network = load_julia_network()
    rust_heavy = load_rust_heavy_network()

    if julia_binary is None or julia_network is None or rust_heavy is None:
        return

    rows = []

    for scenario, julia_ns in sorted(julia_binary.items()):
        rust_ns = load_criterion_estimate("binary", "einsum", scenario)
        rows.append(("binary", scenario, julia_ns, rust_ns))

    for scenario, julia_ns in sorted(julia_network.items()):
        if scenario == "3reg_150":
            rust_ns = rust_heavy.get(scenario)
        else:
            rust_ns = load_criterion_estimate("network", "einsum", scenario)
        rows.append(("network", scenario, julia_ns, rust_ns))

    print(
        f"| {'Benchmark':<10} | {'Scenario':<32} | {'Julia (ns)':>12} | {'Rust (ns)':>12} | {'Julia/Rust':>10} | {'Drop vs Julia':>13} |"
    )
    print(
        f"|{'-' * 12}|{'-' * 34}|{'-' * 14}|{'-' * 14}|{'-' * 12}|{'-' * 15}|"
    )

    for benchmark, scenario, julia_ns, rust_ns in rows:
        print(
            f"| {benchmark:<10} | {scenario:<32} | {fmt_ns(julia_ns)} | {fmt_ns(rust_ns)} | {fmt_ratio(julia_ns, rust_ns)} | {fmt_drop(julia_ns, rust_ns)} |"
        )


if __name__ == "__main__":
    main()
