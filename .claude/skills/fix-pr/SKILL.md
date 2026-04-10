---
name: fix-pr
description: Resolve PR review comments, local CI failures, and coverage gaps for omeinsum-rs
---

# Fix PR

Workflow for cleaning up the current PR after review or CI feedback.

## Step 1: Gather PR State

Confirm there is an open PR for the current branch:

```bash
BRANCH=$(git branch --show-current)
gh pr view --json number,title,url,baseRefName,headRefName
gh pr view --comments
gh pr checks
```

If there is no PR, stop and report that first.

## Step 2: Triage Findings

Prioritize in this order:

1. failing checks
2. correctness review comments
3. missing regression coverage
4. style / cleanup comments

Do not bundle unrelated cleanups into the same fix pass.

## Step 3: Reproduce Locally

Run the canonical local gate:

```bash
make check
```

If the PR touches tropical paths, also run:

```bash
cargo test --features tropical
```

If the PR touches CUDA code, run the relevant CUDA tests only when the environment supports them.

## Step 4: Fix Review Comments

For each valid comment:

- read the referenced code in full
- add or tighten a regression when behavior changes
- implement the minimal fix
- rerun the narrowest relevant tests first, then the broader gate

If a comment is technically wrong, do not silently apply it. Record the reasoning for why it should be declined.

## Step 5: Close Coverage Gaps

Prefer targeted regressions over broad smoke tests.

For `omeinsum-rs`, coverage work should usually target:

- repeated-label contraction paths
- backward paths across unary, binary, and multi-tensor topologies
- backend/device-preservation behavior
- exact output and gradient values for small tensors

## Step 6: Verify and Commit

Before committing, run:

```bash
make check
```

If tropical paths changed, also rerun:

```bash
cargo test --features tropical
```

Commit with a message that states what class of PR feedback was addressed.
