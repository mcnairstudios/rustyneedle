use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rustyneedle", version, about = "On-device tool calling, extraction, embeddings and speech")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub direct: DirectArgs,
}

#[derive(Args)]
pub struct DirectArgs {
    #[arg(long = "model", global = false)]
    pub models: Vec<PathBuf>,
    #[arg(long)]
    pub tools: Option<String>,
    #[arg(long)]
    pub audio: Option<PathBuf>,
    #[arg(long)]
    pub query: Option<String>,
    #[arg(long)]
    pub system: Option<String>,
    #[arg(long)]
    pub depth: Option<usize>,
    #[arg(long)]
    pub audio_depth: Option<usize>,
    #[arg(long)]
    pub pretty: bool,
    #[arg(long)]
    pub no_date: bool,
}

#[derive(Args, Clone)]
pub struct NeedleModel {
    #[arg(long = "model")]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub depth: Option<usize>,
    #[arg(long, default_value_t = false)]
    pub pretty: bool,
    #[arg(long, default_value_t = false)]
    pub no_date: bool,
}

#[derive(Args, Clone)]
pub struct SpeechOptions {
    #[arg(long = "whistle")]
    pub whistle: Option<PathBuf>,
    #[arg(long)]
    pub audio_depth: Option<usize>,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long, value_delimiter = ',')]
    pub keywords: Vec<String>,
    #[arg(long)]
    pub keywords_file: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Command {
    Complete {
        #[command(flatten)]
        model: NeedleModel,
        #[arg(long)]
        tools: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
        query: String,
    },
    Extract {
        #[command(flatten)]
        model: NeedleModel,
        #[arg(long)]
        schema: String,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        no_strict: bool,
        text: String,
    },
    Embed {
        #[command(flatten)]
        model: NeedleModel,
        text: String,
    },
    Transcribe {
        #[command(flatten)]
        speech: SpeechOptions,
        #[arg(long)]
        word_timestamps: bool,
        #[arg(long)]
        pretty: bool,
        audio: PathBuf,
    },
    Stream {
        #[command(flatten)]
        speech: SpeechOptions,
        #[arg(long, default_value_t = 1.0)]
        chunk_seconds: f64,
        audio: PathBuf,
    },
    Intent {
        #[command(flatten)]
        model: NeedleModel,
        #[command(flatten)]
        speech: SpeechOptions,
        #[arg(long)]
        tools: String,
        #[arg(long)]
        system: Option<String>,
        audio: PathBuf,
    },
    Download {
        spec: String,
        #[arg(long, default_value = ".")]
        out: PathBuf,
    },
    Playground {
        #[arg(long = "model")]
        model: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 7860)]
        port: u16,
        #[arg(long)]
        depth: Option<usize>,
    },
}
