# Gradient Tape Unification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `einsum_with_grad()` and `cost_and_gradient()` onto one executed-tape/cache-driven gradient engine, then consolidate integration tests behind a single `tests/main.rs` harness.

**Architecture:** Introduce a shared gradient tape that represents the executed einsum program, including unary normalization/finalization steps and binary contractions. Both public gradient entrypoints will build the same tape, execute it once, cache intermediate tensors/argmax state, and walk the tape backward using the existing elementary rules in `src/einsum/backward.rs`.

**Tech Stack:** Rust, Cargo test harness, omeco contraction trees

---

### Task 1: Lock In The Public Regression

**Files:**
- Modify: `tests/backward.rs`
- Modify: `tests/coverage.rs`

- [ ] Add a failing integration test proving `einsum_with_grad(...).backward()` must support a 3-tensor standard contraction.
- [ ] Run the targeted test and confirm it fails because the current public driver panics for `inputs.len() > 2`.
- [ ] Update the old coverage test that documents the panic so it matches the new expected behavior after the refactor.

### Task 2: Introduce Shared Gradient Tape State

**Files:**
- Modify: `src/einsum/backward.rs`
- Modify: `src/einsum/mod.rs`

- [ ] Add internal tape/topology state that can represent executed unary and binary steps plus leaf ownership.
- [ ] Add a shared forward builder/executor that produces `(result, tape/cache)` for both public entrypoints.
- [ ] Keep elementary unary/binary backward rules in `src/einsum/backward.rs`; only orchestration should move.

### Task 3: Switch Public APIs To The Shared Engine

**Files:**
- Modify: `src/einsum/mod.rs`
- Modify: `src/einsum/backward.rs`

- [ ] Replace the current arity-based `EinsumGradient::backward()` driver with tape replay.
- [ ] Make `einsum_with_grad()` return tape-backed gradient state.
- [ ] Make `cost_and_gradient()` reuse the same tape execution path instead of maintaining a separate orchestration model.

### Task 4: Consolidate Integration Tests

**Files:**
- Create: `tests/main.rs`
- Create: `tests/suites/*`
- Modify: existing `tests/*.rs`

- [ ] Introduce a single integration-test crate entrypoint at `tests/main.rs`.
- [ ] Move existing standalone integration files under `tests/suites/` and wire them in via `#[path]`.
- [ ] Preserve existing test names and assertions while reducing Cargo test binary fan-out.

### Task 5: Verify

**Files:**
- Test: `tests/backward.rs`
- Test: `tests/coverage.rs`
- Test: `tests/main.rs`

- [ ] Run targeted gradient regression tests.
- [ ] Run `cargo test --lib`.
- [ ] Run the consolidated integration target.
