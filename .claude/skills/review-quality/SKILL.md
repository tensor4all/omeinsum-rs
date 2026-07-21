---
name: review-quality
description: Use when reviewing omeinsum-rs code changes for correctness, DRY, KISS, cohesion/coupling, CLI/HCI quality, and test quality
---

# Quality Review

Read-only review workflow for `omeinsum-rs`.

This skill reports issues. It does not edit files, commit, or push.

## Invocation

- `/review-quality` - auto-detect the review range from git
- `/review-quality <base>..<head>` - review a specific diff range

For Codex, open this `SKILL.md` directly and treat slash-command forms as aliases.

## Step 1: Get Context

Determine the review range:

```bash
BASE=$(git merge-base HEAD main 2>/dev/null || git rev-parse HEAD~1)
echo "$BASE"
git diff --stat "$BASE"..HEAD
git diff "$BASE"..HEAD
```

If the worktree is dirty, also inspect the uncommitted delta:

```bash
git diff --stat
git diff
```

Read every changed file in full before making judgments.

## Step 2: Evaluate Correctness

Prioritize behavioral risks over style.

Check for:

- **Repeated-label semantics**: repeated labels on a single operand must be normalized correctly before binary contraction.
- **Backend preservation**: new tensors, gradients, and reductions must stay on the caller's backend/device.
- **Scalar shape correctness**: scalar outputs should use `[]`, not `[1]`, unless the API explicitly says otherwise.
- **Tropical argmax routing**: winner-tracking and backward routing must remain consistent across unary and binary paths.
- **Public/API parity**: public helpers should not quietly support a narrower set of cases than the forward path.

## Step 3: Evaluate Design Principles

### DRY

Look for duplicated orchestration logic, especially duplicated forward/backward drivers, repeated normalization code, or copy-pasted backend allocation paths.

### KISS

Flag:

- public special cases that should use the shared engine
- multi-branch control flow where one normalized path would do
- abstractions that add indirection without removing real complexity

### Cohesion / Coupling

Flag:

- files mixing tensor semantics, backend allocation, and public API glue without clear boundaries
- modules that reach into each other's internals when a helper boundary would suffice
- long functions doing both lowering and execution when those responsibilities can be separated cleanly

## Step 4: Evaluate CLI / HCI Quality

Only check these if the diff touches `omeinsum-cli/` or CLI docs:

- Error messages should explain the invalid input and the expected format.
- JSON output should stay stable and documented.
- `optimize`, `contract`, and `autodiff` behavior should be consistent between expression and topology inputs.
- Non-scalar autodiff should require an explicit gradient seed with a clear error.

## Step 5: Evaluate Test Quality

Flag tests that:

- only check shapes or lengths without checking values
- mirror the implementation too closely
- miss adversarial repeated-label cases
- miss backend/device-preservation assertions when backend ownership changed
- avoid checking gradients numerically or by exact small examples

Prefer small exact tensors with hand-checkable expected values.

## Output Format

```markdown
## Quality Review

### Correctness
- OK / ISSUE — description with file:line

### Design Principles
- DRY: OK / ISSUE — description with file:line
- KISS: OK / ISSUE — description with file:line
- HC/LC: OK / ISSUE — description with file:line

### CLI / HCI
- OK / ISSUE — description with file:line, or N/A

### Test Quality
- OK / ISSUE — description with test file references

### Issues

#### Critical
[correctness bugs and regressions]

#### Important
[design problems and missing regressions]

#### Minor
[cleanup opportunities]
```
