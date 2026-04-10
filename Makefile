# Makefile for omeinsum-rs
# Automates environment setup, testing, documentation, and examples

NON_GPU_FEATURES := tropical parallel

.PHONY: all build build-debug cargo-check check test test-gpu test-release bench docs clean help
.PHONY: setup setup-rust
.PHONY: docs-build docs-serve docs-book docs-book-serve
.PHONY: fmt fmt-check clippy lint coverage
.PHONY: example-basic example-tropical

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
	@echo "  bench          - Run benchmarks"
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
	@echo "Running benchmarks..."
	cargo bench
	@echo "Benchmarks complete."

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
