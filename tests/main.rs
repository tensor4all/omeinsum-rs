#[path = "suites/backend_contract.rs"]
mod backend_contract;
#[path = "suites/backward.rs"]
mod backward;
#[path = "suites/binary_rules.rs"]
mod binary_rules;
#[path = "suites/coverage.rs"]
mod coverage;
#[cfg(feature = "cuda")]
#[path = "suites/cuda.rs"]
mod cuda;
#[path = "suites/einsum_core.rs"]
mod einsum_core;
#[path = "suites/integration.rs"]
mod integration;
#[path = "suites/omeinsum_compat.rs"]
mod omeinsum_compat;
#[path = "suites/optimizer.rs"]
mod optimizer;
#[path = "suites/showcase.rs"]
mod showcase;
#[cfg(feature = "tropical")]
#[path = "suites/tropical.rs"]
mod tropical;
#[path = "suites/unary_ops.rs"]
mod unary_ops;
