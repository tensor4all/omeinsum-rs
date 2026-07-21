---
name: fix-pr
description: Use when an omeinsum-rs pull request has review comments, local CI failures, or coverage gaps to address
---

# Fix PR

Workflow for cleaning up the current PR after review or CI feedback.

## Invocation

- `/fix-pr` - fix feedback on the current branch's PR

For Codex, open this `SKILL.md` directly and treat slash-command forms as aliases.

## Step 1: Gather PR State

Confirm there is an open PR for the current branch:

```bash
BRANCH=$(git branch --show-current)
gh pr view --json number,title,url,baseRefName,headRefName,headRefOid
gh pr view --comments
gh pr checks
```

If there is no PR, stop and report that first.

Read inline review comments, PR conversation comments, linked issue comments referenced by reviewers, CI failures, and Codecov comments. Do not assume `gh pr view --comments` is the only feedback surface if the user points to another source.

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

If a comment is technically wrong, do not silently apply it. Record the reasoning for why it should be declined, and do not rewrite unrelated code while addressing it.

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

Commit with a message that states what class of PR feedback was addressed. If no commit is requested, leave the worktree changes in place and report them clearly.
