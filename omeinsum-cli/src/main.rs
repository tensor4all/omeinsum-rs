use clap::{Parser, Subcommand};

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
    Optimize,
    /// Execute a tensor contraction
    Contract,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Optimize => {
            eprintln!("optimize: not yet implemented");
            std::process::exit(1);
        }
        Commands::Contract => {
            eprintln!("contract: not yet implemented");
            std::process::exit(1);
        }
    }
}
