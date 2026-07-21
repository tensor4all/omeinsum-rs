# CLAUDE.md

## Project Overview

Rust library for Einstein summation over standard and tropical algebras, with contraction-order optimization and backward support.

## Philosophy

- **Correctness before convenience.** Preserve exact einsum semantics, especially around repeated labels, scalar shapes, and tropical argmax routing.
- **Simple logic, maximum reuse.** Prefer shared normalization and execution paths over public ad hoc special cases.
- **Root-cause fixes over patches.** If a bug comes from lowering, topology, or backend ownership, fix it at the source.
- **Topology and tensor data are separate concerns.** Keep contraction structure independent from concrete tensor storage and backend instances.
- **Tests should prove values, not just shapes.** Shape-only assertions are rarely enough for contraction code.

## Commands

```bash
make help
make cargo-check
make check
make test
make cli            # Build and install the omeinsum CLI to ~/.cargo/bin
cargo test --test main
cargo test --features tropical
```

`make check` is the canonical pre-PR gate. It runs formatting, clippy, and the non-GPU test suite.

## Releasing

Releases are cut with the `make release` target and published to crates.io by CI:

```bash
make release V=x.y.z
```

This bumps `version` in `Cargo.toml` + `omeinsum-cli/Cargo.toml`, runs `cargo check`,
commits `release: vx.y.z`, tags `vx.y.z`, pushes `main` + the tag, and creates a
GitHub release with generated notes. Guards: must run from `main`, requires a clean
worktree, and aborts if the tag already exists. (`Cargo.lock` is gitignored, so it is
only added when tracked.)

Publishing is automated: the GitHub *release published* event triggers
`.github/workflows/release.yml`, which runs `cargo publish` with the
`CARGO_REGISTRY_TOKEN` secret. `workflow_dispatch` supports a `dry_run` input
(`cargo publish --dry-run`). The `release` repo-local skill wraps this flow with a
pre-release hygiene gate.

## CLI (`omeinsum`)

Install with `make cli`. Three subcommands:

- **`omeinsum optimize`** — Optimize contraction order for an einsum expression.
  `omeinsum optimize "ij,jk->ik" --sizes "i=2,j=3,k=4"` (methods: `greedy`, `treesa`).
  Greedy accepts `--alpha`, `--temperature`. TreeSA accepts `--ntrials`, `--niters`,
  `--betas`, `--sc-target`, `--tc-weight`, `--sc-weight`, `--rw-weight`.
- **`omeinsum contract`** — Execute a tensor contraction from a tensors JSON file.
  Requires either `--topology <file>` (from optimize) or `--expr "(ij,jk),kl->il"`.
  Supported dtypes: f32, f64, c32, c64.
- **`omeinsum autodiff`** — Execute a contraction and emit the forward result plus
  gradients for each input tensor. Uses the same `--topology` or `--expr` contract
  selection as `contract`. `--grad-output <file>` is optional for scalar outputs and
  required for non-scalar outputs. Supported dtypes: f32, f64, c32, c64.

## Testing Conventions

- Unit tests belong next to the code in `src/` under `#[cfg(test)]`.
- Integration tests are consolidated through `tests/main.rs`, with suites in `tests/suites/`.
- Do not add new top-level `tests/*.rs` integration crates unless there is a strong reason. Prefer wiring new suites into `tests/main.rs`.
- CUDA tests remain feature-gated and are not part of the default non-GPU verification path.
- For contraction and backward changes, prefer regressions that check concrete values, gradients, and backend/device preservation.

## Review Conventions

- Review for correctness first, then DRY/KISS/HC-LC.
- Look for backend preservation bugs, repeated-label lowering mistakes, scalar shape mismatches, and tropical winner-routing issues.
- When fixing bugs, add the regression before or alongside the implementation.

## Repo-Local Skills

Repo-local skills live under `.claude/skills/*/SKILL.md`.

- `issue-to-pr` - convert a GitHub issue into a scoped implementation plan, branch, and PR
- `review-quality` - read-only review checklist for DRY, KISS, correctness, and test quality
- `fix-pr` - workflow for addressing PR comments, CI failures, and missing coverage
- `release` - prepare, verify, tag, and publish a crate release

## Experiments (runscribe)

Experiments in this repo are logged with [runscribe](https://github.com/isPANN/runscribe), organized as **goal → hypothesis → run** (a goal is a single letter `A`; a hypothesis is `A1`, `A2`; runs are timestamped under their hypothesis).

- **Never run an experiment bare.** Wrap every experiment:
  `runscribe run --hyp <code> [--tag <tag>] -- <command>` (any language) or
  `with runscribe.run(hyp="<code>", tag="<tag>") as r: ...` (in-process Python; `r` is how you call `r.log_metric(...)`).
- **Declare goal, open hypothesis, then bind.** `runscribe goal new <handle> -m "..."` → letter `A`; `runscribe hyp new A --from <parent> -m "..."` → `A1`; then bind runs with `--hyp A1`. `--from` is required — the goal (`A`) for a fresh line or a prior hypothesis (`A1`) it builds on; this records the exploration map. `--hyp` must reference an **already-existing** hypothesis; an unknown code is an error, so declare it before binding.
- **Two gates keep the human in control.** Before running, the agent argues the path in the hypothesis's `## Why this path` section (research-paper style: motivation → the specific unknown → the proposed path) and waits for a go-ahead. After a run, the agent records an *observation* plus its own attempt to refute it and the strongest surviving alternative — never a bare verdict — and leaves the judgment to the human.
- **Name slugs deliberately** (tag, goal handle, hypothesis title all become directory names). Use short ASCII kebab-case. The `--tag` should encode the one variable that distinguishes a run within its hypothesis (`n200`, `sor`, `int8`) so the `runs/` listing reads like a results table; the goal handle is a 2–4 word domain noun (`kv-cache-quant`); the hypothesis title is one short testable claim, not a topic.
- **Report the run directory** (`[runscribe] recorded → …`) alongside any result. No bare metrics.
- **Performance:** runscribe records measurements; you do the timing yourself. Call `log_metric` with aggregated results *outside* timed regions; `log_tensor` stays metadata-only. Keep runscribe calls out of hot loops.
- **Read & log helpers.** `runscribe hyp list [<goal>]` prints the goal→hypothesis tree (lineage + latest result) — use it to pick a `--from` parent. `runscribe show <hyp>` tables every run under a hypothesis for trend comparison. `runscribe note <hyp> -m "..."` appends a timestamped bullet to the hypothesis's `## Log` (handy for thoughts in flight; leaves the gated sections alone).
- **Rebuild tables** with `runscribe index`. Never hand-edit `runscribe/**/INDEX.md`.
