---
name: release
description: Use when preparing, versioning, tagging, or publishing an omeinsum-rs crate release
---

# Release

Prepare and publish an `omeinsum-rs` release. Releases are made from `main`, tagged as `vX.Y.Z`, and published through the GitHub Release workflow.

## Invocation

- `/release` - determine the next version, verify, then release
- `/release 0.2.0` - release an explicit version

For Codex, open this `SKILL.md` directly and treat slash-command forms as aliases.

## Step 1: Preflight

Confirm the local branch and remote state:
```bash
git status --short
git branch --show-current
git log --oneline -10
gh api repos/TensorBFS/omeinsum-rs/tags --jq '.[].name'
gh api repos/TensorBFS/omeinsum-rs/releases --jq '.[].tag_name'
```

Stop if:
- the branch is not `main`
- the worktree is dirty
- the chosen tag already exists locally or remotely
- the user has not provided a version and the version bump is ambiguous

## Step 2: Choose Version

Read `Cargo.toml` and `omeinsum-cli/Cargo.toml`.

For `0.x` releases:
- Patch (`0.x.Y`) for bug fixes, docs, CI, and compatibility-only changes.
- Minor (`0.X.0`) for new public API, CLI features, performance features, or broad behavior changes.
- Initial release uses the existing package version if there are no prior tags and crates.io has no published version.

If the crate already exists on crates.io, do not reuse a published version.

## Step 3: Verify

Run the canonical local gate before publishing:
```bash
make check
cargo publish --dry-run
```

If the release is expected to include CLI packaging behavior, also run:
```bash
cargo test -p omeinsum-cli
```

Do not proceed if verification fails.

## Step 4: Publish

Use the Makefile target:
```bash
make release V=x.y.z
```

The target:
- updates `Cargo.toml` and `omeinsum-cli/Cargo.toml`
- stages `Cargo.lock` only if this repo starts tracking it
- commits version changes only when needed
- creates and pushes `vX.Y.Z`
- creates a GitHub Release, which triggers the release workflow

## Step 5: Confirm

Check the release and workflow:
```bash
gh release view vX.Y.Z
gh run list --workflow release.yml --limit 3
```

Report:
- version and tag
- GitHub release URL
- whether the release workflow started or completed
- any follow-up needed for crates.io publication

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Publishing from a dirty tree | Commit or remove unrelated changes first |
| Reusing an existing tag or crate version | Pick the next semver version |
| Only pushing a tag | Create a GitHub Release so the workflow publishes |
| Skipping `cargo publish --dry-run` | Dry-run before `make release` |
| Assuming CUDA tests are required | Run CUDA tests only when release changes CUDA paths and local support exists |
