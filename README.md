# rustyneedle

A pure-Rust runtime for [Needle 3](https://github.com/cactus-compute/needle) and Whistle: on-device tool calling, structured extraction, embeddings, speech-to-text and a direct speech-to-intent path.

rustyneedle is inspired by, and built to be compatible with, **Cactus Compute's [Needle](https://github.com/cactus-compute/needle)**. It reads the same `.cact` archives and follows the behaviour of the original library, whose native engine is closed-source. It is an independent project and is not affiliated with Cactus Compute. Training, fine-tuning and `.cact` export stay in the original Python package; this repository runs what that produces. Like the parent project, it is licensed under Apache-2.0.

Scope: Needle 3 and Whistle. No Needle 2 support and no telemetry.

## Quick start

```sh
git clone https://github.com/mcnairstudios/rustyneedle.git
cd rustyneedle
make weights          # builds, then downloads the Needle and Whistle archives into weights/

./target/release/rustyneedle complete --model weights/needle3.cact \
  --tools examples/tools/home_assistant.json "turn on the overhead light in the kitchen"
```

Needs a stable Rust toolchain (1.87 or newer) and `make`. `make help` lists every task. Speech works the same way with `--audio`; see [docs/cli.md](docs/cli.md).

## Model files are not in this repository

`weights/`, `oracle/assets/`, `oracle/goldens/`, `oracle/clips/` and `examples/clips/` are git-ignored. They are filled from the official sources by make tasks, and the weights stay under the licences published there.

| Task | Fills | Source |
|---|---|---|
| `make weights` | `weights/needle3.cact`, `weights/whistle.cact` | Hugging Face `Cactus-Compute/needle3` and `Cactus-Compute/whistle` |
| `make oracle-assets` | `oracle/assets/` (master checkpoint, tokenizer) | Hugging Face `Cactus-Compute/needle3` |
| `make goldens` | `oracle/goldens/`, `oracle/clips/` | recorded from the original Python package (`make oracle-env` installs it) |
| `make example-clips` | `examples/clips/` | macOS speech synthesis |
| `make setup` | all of the above | |
| `make distclean` | removes all of the above and the build output | |

Re-running a task replaces the files. The tests that need these files are marked `#[ignore]`, so `make test` works on a fresh clone and `make test-all` runs `make setup` first.

## Use it as a library

```toml
[dependencies]
rustyneedle = { git = "https://github.com/mcnairstudios/rustyneedle" }
# features: "speech" adds Whistle and speech-to-intent, "download" adds the weight downloader
```

```rust
use rustyneedle::{tool, Model, Needle, Parameter, RunOptions};
use serde_json::json;

let model = Model::open("weights/needle3.cact")?;
let weather = tool("get_weather").description("Get the current weather for a city.").parameter(Parameter::string("city")).build();

let mut agent = Needle::builder()
    .model(&model)
    .tool_with_handler(weather, |arguments| Ok(json!({"city": arguments["city"], "temp_c": 27})))
    .build();

let response = agent.run("what's it like in Lagos right now?", &RunOptions::default());
println!("{}", response.results.unwrap()[0]);
```

[docs/library.md](docs/library.md) is the integration guide: tools, responses and confidence, the agent loop, extraction, embeddings, speech, testing without a model, threading and errors.

## What it does

| Capability | Entry point |
|---|---|
| Tool calls with grammar-constrained decoding, calibrated confidence, grounding gates and the deterministic repair step | `Needle::complete`, `Needle::run` |
| Structured extraction with grounding validation | `rustyneedle::extract` |
| Text embeddings (unit-norm probe readout) | `Needle::embed` |
| Speech-to-text in seven languages, word timestamps, keywords, 30-second windowing and live streaming | `rustyneedle::speech::{Whistle, Stream}` |
| Audio in, tool calls out | `rustyneedle::speech::Intent` |
| Depth ladder: any rung from 2 to 20 layers (Needle) or 2 to 8 (Whistle decoder) | `Model::with_depth` |
| Command line and browser playground | `rustyneedle`, [docs/cli.md](docs/cli.md) |

## Examples

Tool sets, usable with `--tools` on the command line or `ToolFile` in Rust:

| File | Shows |
|---|---|
| `examples/tools/home_assistant.json` | eight Home Assistant intents with device names as enums |
| `examples/tools/weather_and_time.json` | two tools that need `triggers` to tell weather from time |
| `examples/tools/calendar.json` | date and time phrases passed through verbatim |
| `examples/tools/car.json` | enums, bounded integers and a free-text destination |
| `examples/tools/customer_support.json` | order numbers copied verbatim and a hand-off tool |
| `examples/environments/*.json` | six tool sets with frozen test cases: smart home, media player, productivity, wearable, kitchen appliance, data capture |

Rust programs, each run with `cargo run --release -p rustyneedle --example <name>`:

| Example | Shows |
|---|---|
| `quickstart` | a tool with a handler and the agent loop |
| `tool_file <tools.json> "<request>"` | load any tool file and print the calls and confidence |
| `extract` | structured extraction from free text |
| `embed` | embedding similarity between sentences |
| `custom_engine` | testing an application with a scripted `Engine`, no model needed |
| `speech <tools.json> <clip.wav>` (`--features speech`) | speech to tool call |
| `download` (`--features download`) | fetching the weights from code |

[examples/README.md](examples/README.md) has the details. `examples/try.sh` runs the text and speech paths end to end.

## Layout

The public crate is `rustyneedle`. The others are its implementation and are published only because it depends on them.

| Crate | Responsibility |
|---|---|
| `rustyneedle` | the public API: re-exports from the crates below |
| `rustyneedle-cact` | memory-mapped archive: header geometry, tensor directory, tensor layout |
| `rustyneedle-kernels` | codebook-quantised products (NEON int8 `sdot` kernels with widening and float fallbacks), Walsh rotation, activation quantisation, attention primitives, the worker pool |
| `rustyneedle-tokenizer` | SentencePiece BPE read from the archive |
| `rustyneedle-model` | Needle 3 forward pass: multi-lane routing, engram memory, int8 KV cache, probe heads, batched prefill, depth ladder |
| `rustyneedle-grammar` | JSON-schema subset compiled to a byte-level state machine, and token masking |
| `rustyneedle-generate` | prompt rendering, prefix caching, constrained greedy decoding |
| `rustyneedle-repair` | the deterministic post-decode repair and gate rules |
| `rustyneedle-needle` | the agent: engine trait, tool sets and builder, grounding, run loop, extraction, embeddings |
| `rustyneedle-whistle` | log-mel frontend, conv stem, batched encoder, cross-attention decoder, timestamps, keywords, windowing, streaming |
| `rustyneedle-intent` | speech to intent |
| `rustyneedle-fetch` | weight download from Hugging Face |
| `rustyneedle-playground` | the HTTP playground server and its page |
| `rustyneedle-cli` | the `rustyneedle` binary |
| `rustyneedle-envs`, `rustyneedle-bench`, `rustyneedle-testkit` | acceptance suites, the `rustyneedle-bench` binary and test helpers; not published |

## Development

| Command | Runs |
|---|---|
| `make build` | release build of everything |
| `make test` | the tests that need no downloads |
| `make test-all` | everything, after `make setup` |
| `make lint` | `cargo fmt --check` and clippy with warnings denied |
| `make doc` | the library documentation |
| `make bench` | timings for the engine and Whistle |

The recorded outputs, sampled weight rows and synthesised clips are generated, never committed: [docs/parity.md](docs/parity.md) explains how they are produced and what they are compared with. [docs/benchmarking.md](docs/benchmarking.md) explains how to measure against the original.

## Documentation

- [docs/library.md](docs/library.md): using rustyneedle from Rust
- [docs/cli.md](docs/cli.md): every command
- [docs/tools.md](docs/tools.md): designing tools, dates and times, triggers
- [docs/architecture.md](docs/architecture.md): data flow, numerics, performance
- [docs/format.md](docs/format.md): the `.cact` archive
- [docs/parity.md](docs/parity.md): how behaviour is checked against the original, and the results
- [docs/portability.md](docs/portability.md): supported chips, runtime kernel selection, microcontroller limits
- [docs/benchmarking.md](docs/benchmarking.md): measuring performance

## Licence and credits

Apache-2.0, as is the parent project; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Needle and Whistle are by Cactus Compute: <https://github.com/cactus-compute/needle>. Model weights are not part of this repository and remain under the licences published with them on Hugging Face.
