# Makefile for omeinsum-rs
# Automates environment setup, testing, documentation, and examples

NON_GPU_FEATURES := tropical parallel
BENCH_SCENARIO ?= all
BENCH_ITERATIONS ?= 40
BENCH_WARMUP ?= 5
BENCH_DIM ?= 128
BENCH_BATCH ?= 24

.PHONY: all build build-debug cargo-check check test test-gpu test-release bench bench-binary bench-network bench-cpu-contract bench-julia bench-compare docs clean help
.PHONY: setup setup-rust
.PHONY: docs-build docs-serve docs-book docs-book-serve
.PHONY: fmt fmt-check clippy lint coverage
.PHONY: example-basic example-tropical
.PHONY: cli
.PHONY: release copilot-review run-plan

# Cross-platform sed in-place: macOS needs -i '', Linux needs -i
SED_I := sed -i$(shell if [ "$$(uname)" = "Darwin" ]; then echo " ''"; fi)

# Default target
all: build test

#==============================================================================
# Help
#==============================================================================

help:
	@echo "omeinsum-rs Makefile"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup          - Setup complete development environment"
	@echo "  setup-rust     - Install Rust toolchain and components"
	@echo ""
	@echo "Build targets:"
	@echo "  build          - Build in release mode"
	@echo "  build-debug    - Build in debug mode"
	@echo "  cargo-check    - Fast compile-only check for the non-GPU feature set"
	@echo "  check          - Canonical pre-PR gate (fmt-check + clippy + test)"
	@echo ""
	@echo "Test targets:"
	@echo "  test           - Run tests with non-GPU features (tropical, parallel)"
	@echo "  test-gpu       - Run tests with all features including CUDA"
	@echo "  test-release   - Run tests in release mode (non-GPU features)"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  bench          - Run all Rust benchmarks"
	@echo "  bench-binary   - Run binary contraction benchmarks"
	@echo "  bench-network  - Run tensor-network benchmarks"
	@echo "  bench-cpu-contract - Run the CPU contraction benchmark example"
	@echo "                   Override BENCH_SCENARIO, BENCH_ITERATIONS, BENCH_WARMUP,"
	@echo "                   BENCH_DIM, or BENCH_BATCH to tune the run"
	@echo "  bench-julia    - Run Julia benchmarks in benchmarks/julia/"
	@echo "  bench-compare  - Compare Rust vs Julia benchmark timings"
	@echo ""
	@echo "Example targets:"
	@echo "  example-basic     - Run basic einsum example"
	@echo "  example-tropical  - Run tropical network example"
	@echo ""
	@echo "Documentation targets:"
	@echo "  docs           - Build all documentation (API + user guide)"
	@echo "  docs-build     - Build Rust API documentation"
	@echo "  docs-book      - Build mdBook user guide"
	@echo "  docs-book-serve - Serve mdBook locally (port 3000)"
	@echo "  docs-serve     - Serve API docs locally (port 8000)"
	@echo ""
	@echo "Code quality targets:"
	@echo "  fmt            - Format code"
	@echo "  clippy         - Run clippy lints"
	@echo "  lint           - Run all lints (fmt check + clippy)"
	@echo "  coverage       - Generate test coverage report"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean          - Clean build artifacts"
	@echo ""
	@echo "CLI targets:"
	@echo "  cli            - Build and install the omeinsum CLI to ~/.cargo/bin"
	@echo ""
	@echo "Release targets:"
	@echo "  release V=x.y.z - Tag and push a new release (triggers CI publish)"
	@echo "  copilot-review   - Request Copilot code review on current PR"
	@echo ""
	@echo "Agent targets:"
	@echo "  run-plan         - Execute a plan with Claude or Codex (latest plan in docs/plans/)"
	@echo "                     Set RUNNER=claude to use Claude (default: codex)"
	@echo "                     Override CLAUDE_MODEL or CODEX_MODEL to pick a different model"

#==============================================================================
# Environment Setup
#==============================================================================

setup: setup-rust
	@echo "Development environment setup complete!"

setup-rust:
	@echo "Setting up Rust toolchain..."
	rustup update stable
	rustup component add rustfmt clippy
	cargo install mdbook --no-default-features --features search
	cargo install cargo-tarpaulin
	@echo "Rust setup complete."

#==============================================================================
# Build
#==============================================================================

build:
	cargo build --release

build-debug:
	cargo build

cargo-check:
	cargo check --features "$(NON_GPU_FEATURES)"

#==============================================================================
# Testing
#==============================================================================

test:
	@echo "Running tests with non-GPU features (tropical, parallel)..."
	cargo test --features "$(NON_GPU_FEATURES)"
	@echo "Tests complete."

test-gpu:
	@echo "Running tests with all features (including CUDA)..."
	cargo test --features "tropical parallel cuda"
	@echo "Tests complete."

test-release:
	@echo "Running tests in release mode (non-GPU features)..."
	cargo test --release --features "tropical parallel"
	@echo "Tests complete."

#==============================================================================
# Benchmarks
#==============================================================================

bench:
	@echo "Running all Rust benchmarks..."
	$(MAKE) bench-binary
	$(MAKE) bench-network
	@echo "All Rust benchmarks complete. Results in target/criterion/"

bench-binary:
	cargo bench --bench binary

bench-network:
	cargo bench --bench network
	cargo run --release --example profile_network -- --scenario 3reg_150 --iterations 1 --output benchmarks/data/rust_network_timings.json

bench-cpu-contract:
	@echo "Running CPU contraction benchmarks..."
	cargo run --release --example cpu_contract_bench -- --scenario "$(BENCH_SCENARIO)" --iterations "$(BENCH_ITERATIONS)" --warmup "$(BENCH_WARMUP)" --dim "$(BENCH_DIM)" --batch "$(BENCH_BATCH)"
	@echo "CPU contraction benchmarks complete."

bench-julia:
	cd benchmarks/julia && julia --project=. -e 'using Pkg; omeinsum_path=get(ENV, "OMEINSUM_JL_PATH", expanduser("~/.julia/dev/OMEinsum")); orders_path=get(ENV, "OMEINSUM_CONTRACTION_ORDERS_JL_PATH", expanduser("~/.julia/dev/OMEinsumContractionOrders")); Pkg.develop(path=orders_path); Pkg.develop(path=omeinsum_path); Pkg.instantiate()' && julia --project=. generate_timings.jl

bench-compare:
	python3 benchmarks/compare.py

#==============================================================================
# Examples
#==============================================================================

example-basic:
	@echo "Running basic einsum example..."
	cargo run --release --example basic_einsum --no-default-features
	@echo "Example complete."

example-tropical:
	@echo "Running tropical network example..."
	cargo run --release --example tropical_network --no-default-features
	@echo "Example complete."

#==============================================================================
# Documentation
#==============================================================================

docs: docs-build docs-book

docs-build:
	@echo "Building Rust API documentation..."
	cargo doc --no-deps
	@echo "API documentation built at target/doc/"

docs-book:
	@echo "Building mdBook user guide..."
	@which mdbook > /dev/null 2>&1 || (echo "Install mdbook: cargo install mdbook" && exit 1)
	mdbook build docs/
	@echo "User guide built at docs/book/"

docs-book-serve:
	@echo "Serving mdBook at http://localhost:3000"
	@which mdbook > /dev/null 2>&1 || (echo "Install mdbook: cargo install mdbook" && exit 1)
	mdbook serve docs/

docs-serve: docs-build
	@echo "Serving API documentation at http://localhost:8000"
	@cd target/doc && python -m http.server 8000

#==============================================================================
# Code Quality
#==============================================================================

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --features "$(NON_GPU_FEATURES)" -- -D warnings

lint: fmt-check clippy

check: fmt-check clippy test
	@echo "All checks passed."

coverage:
	@echo "Generating test coverage..."
	cargo tarpaulin --workspace --out Html --output-dir coverage/
	@echo "Coverage report at coverage/tarpaulin-report.html"

#==============================================================================
# Cleanup
#==============================================================================

clean:
	cargo clean
	rm -rf coverage/
	rm -rf docs/book/
	@echo "Clean complete."

#==============================================================================
# CLI
#==============================================================================

cli:
	cargo install --path omeinsum-cli

#==============================================================================
# Release
#==============================================================================

# Release a new version: make release V=0.2.0
release:
ifndef V
	$(error Usage: make release V=x.y.z)
endif
	@echo "Releasing v$(V)..."
	$(SED_I) 's/^version = ".*"/version = "$(V)"/' Cargo.toml
	cargo check
	git add Cargo.toml
	git commit -m "release: v$(V)"
	git tag -a "v$(V)" -m "Release v$(V)"
	git push origin main --tags
	@echo "v$(V) pushed — CI will publish to crates.io"

# Request Copilot code review on the current PR
# Requires: gh extension install ChrisCarini/gh-copilot-review
copilot-review:
	@PR=$$(gh pr view --json number --jq .number 2>/dev/null) || { echo "No PR found for current branch"; exit 1; }; \
	echo "Requesting Copilot review on PR #$$PR..."; \
	gh copilot-review $$PR

#==============================================================================
# Agent-driven plan execution
#==============================================================================

RUNNER ?= codex
CLAUDE_MODEL ?= opus
CODEX_MODEL ?= gpt-5.4

# Run a plan with Codex or Claude
# Usage: make run-plan [INSTRUCTIONS="..."] [OUTPUT=output.log] [RUNNER=codex]
# PLAN_FILE defaults to the most recently modified file in docs/plans/
INSTRUCTIONS ?=
OUTPUT ?= run-plan-output.log
PLAN_FILE ?= $(shell ls -t docs/plans/*.md 2>/dev/null | head -1)

run-plan:
	@. scripts/make_helpers.sh; \
	NL=$$'\n'; \
	BRANCH=$$(git branch --show-current); \
	PLAN_FILE="$(PLAN_FILE)"; \
	if [ -z "$$PLAN_FILE" ]; then echo "No plan files found in docs/plans/"; exit 1; fi; \
	if [ "$(RUNNER)" = "claude" ]; then \
		PROCESS="1. Read the plan file$${NL}2. Execute the plan — it specifies which skill(s) to use$${NL}3. Push: git push origin $$BRANCH$${NL}4. If a PR already exists for this branch, skip. Otherwise create one."; \
	else \
		PROCESS="1. Read the plan file$${NL}2. If the plan references repo-local workflow docs under .claude/skills/*/SKILL.md, open and follow them directly. Treat slash-command names as aliases for those files.$${NL}3. Execute the tasks step by step. For each task, implement and test before moving on.$${NL}4. Push: git push origin $$BRANCH$${NL}5. If a PR already exists for this branch, skip. Otherwise create one."; \
	fi; \
	PROMPT="Execute the plan in '$$PLAN_FILE'."; \
	if [ "$(RUNNER)" != "claude" ]; then \
		PROMPT="$${PROMPT}$${NL}$${NL}Repo-local skills live in .claude/skills/*/SKILL.md. Treat any slash-command references in the plan as aliases for those skill files."; \
	fi; \
	if [ -n "$(INSTRUCTIONS)" ]; then \
		PROMPT="$${PROMPT}$${NL}$${NL}## Additional Instructions$${NL}$(INSTRUCTIONS)"; \
	fi; \
	PROMPT="$${PROMPT}$${NL}$${NL}## Process$${NL}$${PROCESS}$${NL}$${NL}## Rules$${NL}- Tests should be strong enough to catch regressions.$${NL}- Do not modify tests to make them pass.$${NL}- Test failure must be reported."; \
	echo "=== Prompt ===" && echo "$$PROMPT" && echo "===" ; \
	RUNNER="$(RUNNER)" run_agent "$(OUTPUT)" "$$PROMPT"
