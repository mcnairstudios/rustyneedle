mod args;
mod commands;
mod resources;

use clap::Parser;

use args::{Cli, Command, DirectArgs};
use commands::AgentSettings;
use resources::{carries_audio, CliError};

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("rustyneedle: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Some(command) => dispatch(command),
        None => direct(cli.direct),
    }
}

fn dispatch(command: Command) -> Result<(), CliError> {
    match command {
        Command::Complete { model, tools, system, max_new_tokens, query } => {
            println!(
                "{}",
                commands::complete(
                    model.model.as_deref(),
                    model.depth,
                    &tools,
                    AgentSettings { system: system.as_deref(), auto_date: !model.no_date },
                    max_new_tokens,
                    &query,
                    model.pretty
                )?
            )
        }
        Command::Extract { model, schema, system, no_strict, text } => {
            println!("{}", commands::extraction(model.model.as_deref(), model.depth, &schema, system.as_deref(), !no_strict, &text, model.pretty)?)
        }
        Command::Embed { model, text } => println!("{}", commands::embedding(model.model.as_deref(), model.depth, &text)?),
        Command::Transcribe { speech, word_timestamps, pretty, audio } => println!("{}", commands::transcription(&speech, word_timestamps, &audio, pretty)?),
        Command::Stream { speech, chunk_seconds, audio } => commands::streaming(&speech, chunk_seconds, &audio)?.iter().for_each(|line| println!("{line}")),
        Command::Intent { model, speech, tools, system, audio } => {
            println!(
                "{}",
                commands::intent(
                    model.model.as_deref(),
                    model.depth,
                    &speech,
                    &tools,
                    AgentSettings { system: system.as_deref(), auto_date: !model.no_date },
                    &audio,
                    model.pretty
                )?
            )
        }
        Command::Download { spec, out } => println!("{}", commands::download(&spec, &out)?),
        Command::Playground { model, host, port, depth } => commands::playground(model.as_deref(), depth, &host, port)?,
    }
    Ok(())
}

fn direct(arguments: DirectArgs) -> Result<(), CliError> {
    let mut needle_model = None;
    let mut whistle_model = None;
    for path in &arguments.models {
        if carries_audio(path)? {
            whistle_model = Some(path.clone());
        } else {
            needle_model = Some(path.clone());
        }
    }
    let tools = arguments.tools.clone().unwrap_or_else(|| "[]".to_owned());
    let speech =
        args::SpeechOptions { whistle: whistle_model.clone(), audio_depth: arguments.audio_depth, language: None, keywords: Vec::new(), keywords_file: None };
    if let Some(audio) = &arguments.audio {
        if needle_model.is_some() {
            println!(
                "{}",
                commands::intent(
                    needle_model.as_deref(),
                    arguments.depth,
                    &speech,
                    &tools,
                    AgentSettings { system: arguments.system.as_deref(), auto_date: !arguments.no_date },
                    audio,
                    arguments.pretty
                )?
            );
        } else {
            println!("{}", commands::transcription(&speech, false, audio, arguments.pretty)?);
        }
        return Ok(());
    }
    if let Some(query) = &arguments.query {
        println!(
            "{}",
            commands::complete(
                needle_model.as_deref(),
                arguments.depth,
                &tools,
                AgentSettings { system: arguments.system.as_deref(), auto_date: !arguments.no_date },
                512,
                query,
                arguments.pretty
            )?
        );
        return Ok(());
    }
    Err(CliError("nothing to do: pass a subcommand, --audio or --query (see --help)".into()))
}
