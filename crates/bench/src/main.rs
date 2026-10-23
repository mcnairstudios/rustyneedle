mod engine;
mod kernel;
mod speech;
mod timing;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Latency and throughput measurements for the engine and its kernels")]
struct Arguments {
    #[arg(long, default_value = "weights")]
    weights: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Decode {
        #[arg(long, default_value_t = 600)]
        context: usize,
        #[arg(long, default_value_t = 400)]
        steps: usize,
    },
    Prefill {
        #[arg(long, default_value_t = 4000)]
        tokens: usize,
    },
    Feed {
        #[arg(long, default_value_t = 1100)]
        prefix: usize,
        #[arg(long, default_value_t = 25)]
        turn: usize,
        #[arg(long, default_value_t = 30)]
        rounds: usize,
    },
    Reset {
        #[arg(long, default_value_t = 1100)]
        prefix: usize,
        #[arg(long, default_value_t = 50)]
        rounds: usize,
    },
    Request {
        #[arg(long, default_value = "examples/tools/home_assistant.json")]
        tools: PathBuf,
        #[arg(long, default_value_t = 12)]
        rounds: usize,
        #[arg(long)]
        no_date: bool,
    },
    Whistle {
        #[arg(long, default_value = "examples/clips/kitchen_light.wav")]
        clip: PathBuf,
        #[arg(long, default_value_t = 30)]
        rounds: usize,
    },
    Primitives,
    Attention {
        #[arg(long, default_value_t = 1024)]
        positions: usize,
    },
}

fn main() {
    let arguments = Arguments::parse();
    match arguments.command {
        Command::Decode { context, steps } => engine::decode(&arguments.weights, context, steps),
        Command::Prefill { tokens } => engine::prefill(&arguments.weights, tokens),
        Command::Feed { prefix, turn, rounds } => engine::feed(&arguments.weights, prefix, turn, rounds),
        Command::Reset { prefix, rounds } => engine::reset(&arguments.weights, prefix, rounds),
        Command::Request { tools, rounds, no_date } => engine::request(&arguments.weights, &tools, rounds, !no_date),
        Command::Whistle { clip, rounds } => speech::transcribe(&arguments.weights, &clip, rounds),
        Command::Primitives => kernel::primitives(),
        Command::Attention { positions } => kernel::attention(positions),
    }
}
