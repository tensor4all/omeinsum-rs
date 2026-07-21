---
name: issue-to-pr
description: Use when turning a GitHub issue for omeinsum-rs into a scoped implementation plan, branch, and pull request
---

# Issue to PR

Convert a GitHub issue into a focused `omeinsum-rs` PR. The default output is a plan-only PR that can be executed with `make run-plan`; only implement immediately if the user explicitly asks.

## Invocation

- `/issue-to-pr 42` - plan issue #42, create or resume a PR
- `/issue-to-pr https://github.com/TensorBFS/omeinsum-rs/issues/42` - same, from URL

For Codex, open this `SKILL.md` directly and treat slash-command forms as aliases. This repo does not have a `[action]` PR trigger; use `make run-plan PLAN_FILE=<path>` when execution should be automated.

## Workflow

```
Receive issue reference
  -> fetch issue and comments
  -> inspect repo context
  -> discuss ambiguities with the user
  -> write a concrete plan
  -> create or resume a branch and PR
  -> report how to execute the plan
```

## Step 1: Parse Input

Extract issue number and repository:
- `123` -> issue #123 in the current repo
- full GitHub issue URL -> issue number and repo from URL
- `owner/repo#123` -> issue #123 in that repo

Default repo:
```bash
gh repo view --json nameWithOwner --jq .nameWithOwner
```

## Step 2: Fetch Issue Context

```bash
gh issue view <number> --repo <owner/repo> --json number,title,body,labels,state,url,comments
```

Stop if the issue is closed unless the user explicitly wants a follow-up PR.

Read the body and all comments. Comments may contain maintainer decisions that refine or override the original issue. Summarize the issue back to the user before planning.

## Step 3: Inspect Repo Context

Check the current branch and worktree:
```bash
git status --short
git branch --show-current
git merge-base HEAD main
```

If the worktree is dirty, do not overwrite unrelated changes. Either work around them or ask the user before proceeding if they affect the plan.

Inspect relevant files before choosing an approach. For common change areas:
- Einsum semantics: `src/einsum/`, `tests/suites/binary_rules.rs`, `tests/suites/unary_ops.rs`
- Backward/autodiff: `src/einsum/backward.rs`, `docs/src/backpropagation.md`, `omeinsum-cli/tests/cli.rs`
- CLI behavior: `omeinsum-cli/src/`, `omeinsum-cli/tests/cli.rs`, `docs/src/cli.md`
- Backend/device behavior: `src/backend/`, `tests/suites/backend_contract.rs`
- Optimization/order work: `src/optimizer/`, `docs/src/optimization.md`

## Step 4: Clarify and Choose Approach

For ambiguous issues, discuss with the user before writing the plan. Present 2-3 viable approaches only when there is a meaningful trade-off.

Use these repository priorities when shaping the plan:
- Preserve exact einsum semantics, especially repeated labels and scalar `[]` outputs.
- Keep topology, tensor data, and backend ownership separate.
- Prefer shared normalization/execution paths over public ad hoc special cases.
- For tropical paths, preserve deterministic winner tracking and gradient routing.
- Tests should prove concrete values, gradients, and device/backend preservation, not just shapes.

## Step 5: Write the Plan

Write the plan to `docs/plans/YYYY-MM-DD-<issue-slug>.md`.

Use this structure:
```markdown
# <Issue title or concise plan title>

Issue: #<number> (<url>)

## Context
<What is broken or missing, including relevant issue comments.>

## Approach
<Chosen design and why it fits existing omeinsum-rs patterns.>

## Tasks
1. Add or adjust the focused regression tests first.
2. Implement the minimal source changes.
3. Update CLI/docs/examples if the public surface changes.
4. Run narrow verification, then the canonical gate.

## Test Plan
- <narrow cargo test command(s)>
- `make check`
- `cargo test --features tropical` if tropical paths changed
- CUDA tests only when CUDA code changed and the environment supports them

## Acceptance Criteria
- <specific observable behavior>
- <value/gradient/backend assertions expected>
```

Plan details should be concrete enough for `make run-plan` to execute without rediscovering the issue.

## Step 6: Create or Resume PR

If an open PR for this issue already exists, switch to its branch and update the plan there. Otherwise create a branch from `main`:

```bash
git switch main
git pull --ff-only
git switch -c issue-<number>-<slug>
git add docs/plans/<plan-file>.md
git commit -m "Add plan for #<number>: <title>"
git push -u origin issue-<number>-<slug>
gh pr create --title "Plan #<number>: <title>" --body "## Summary
Adds an implementation plan for #<number>.

## Execution
Run:
\`\`\`bash
make run-plan PLAN_FILE=docs/plans/<plan-file>.md
\`\`\`

Refs #<number>"
```

If the user asked for immediate implementation, run the plan after the plan commit:
```bash
make run-plan PLAN_FILE=docs/plans/<plan-file>.md
```

Then inspect the resulting changes, run verification, and update the PR body if needed. Do not leave the user guessing whether the PR is plan-only or implemented.

## Verification Expectations

Use the narrowest command that exercises the changed behavior first, then run:
```bash
make check
```

Add extra verification when relevant:
- `cargo test --features tropical` for tropical algebra or argmax-routing changes.
- `cargo test -p omeinsum-cli` for CLI JSON, topology, or autodiff changes.
- CUDA-specific tests only when CUDA paths changed and the local environment supports them.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `[action]` in the PR body | This repo has no `[action]` trigger; use `make run-plan` |
| Writing a generic plan | Name exact files, tests, semantics, and acceptance criteria |
| Skipping issue comments | Read comments before planning; they may contain maintainer decisions |
| Testing only shapes | Assert exact values, gradients, scalar shapes, and backend preservation |
| Bundling unrelated cleanup | Keep the PR scoped to the issue |
| Running CUDA tests unconditionally | Only run them when relevant and supported locally |
