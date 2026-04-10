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
cargo test --test main
cargo test --features tropical
```

`make check` is the canonical pre-PR gate. It runs formatting, clippy, and the non-GPU test suite.

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

- `review-quality` - read-only review checklist for DRY, KISS, correctness, and test quality
- `fix-pr` - workflow for addressing PR comments, CI failures, and missing coverage
