use clap::{Parser, Subcommand};

mod contract;
mod format;
mod optimize;
mod parse;

#[derive(Parser)]
#[command(name = "omeinsum", about = "Einstein summation CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimize contraction order for an einsum expression
    Optimize {
        /// Einsum expression in NumPy notation (e.g., "ij,jk->ik")
        expression: String,

        /// Dimension sizes as "label=size" pairs (e.g., "i=2,j=3,k=4")
        #[arg(long)]
        sizes: String,

        /// Optimization method
        #[arg(long, default_value = "greedy")]
        method: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Execute a tensor contraction
    Contract {
        /// Tensors JSON file
        tensors: String,

        /// Topology JSON file
        #[arg(short = 't', long)]
        topology: Option<String>,

        /// Parenthesized einsum expression with explicit contraction order
        #[arg(long)]
        expr: Option<String>,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Optimize {
            expression,
            sizes,
            method,
            output,
            pretty,
        } => optimize::run(&expression, &sizes, &method, output.as_deref(), pretty),
        Commands::Contract {
            tensors,
            topology,
            expr,
            output,
            pretty,
        } => contract::run(
            &tensors,
            topology.as_deref(),
            expr.as_deref(),
            output.as_deref(),
            pretty,
        ),
    };

    if let Err(err) = result {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
