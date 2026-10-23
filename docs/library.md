# Using rustyneedle from Rust

`rustyneedle` is the only crate you depend on. It re-exports the public API of the implementation crates (`rustyneedle-model`, `rustyneedle-needle`, `rustyneedle-whistle` and so on), which are published only because it needs them.

## Add the dependency

```toml
[dependencies]
rustyneedle = { git = "https://github.com/mcnairstudios/rustyneedle" }
serde_json = "1"
```

| Feature | Adds |
|---|---|
| `speech` | `rustyneedle::speech`: Whistle speech-to-text, streaming and speech-to-intent |
| `download` | `rustyneedle::download`: fetch the weights from Hugging Face (pulls in an HTTPS client) |

Requires Rust 1.87 or newer. Build your application in release mode; a debug build is many times slower. For the best speed set `lto = "fat"` and `codegen-units = 1` in your release profile, as this workspace does.

## Get the model files

The library never bundles weights. Use any of:

- `make weights` in a checkout of this repository, which writes `weights/needle3.cact` and `weights/whistle.cact`;
- `rustyneedle download needle3` and `rustyneedle download whistle` from the command line;
- `rustyneedle::download::base_weights(Base::Needle)` from code (feature `download`), which caches the file in `~/.cache/rustyneedle` or the directory in `RUSTYNEEDLE_CACHE_DIR`. `HF_HUB_OFFLINE=1` forbids downloads.

## Open a model

```rust
use rustyneedle::Model;

let model = Model::open("weights/needle3.cact")?;
```

`Model::open` memory-maps the archive and returns `Arc<Model>`. The model is immutable and `Send + Sync`: open it once and share it between agents and threads. `model.with_depth(n)` returns the same weights cut to the first `n` layers (2 to 20), which is faster and less accurate.

## Describe the tools

Build a tool with the builder:

```rust
use rustyneedle::{tool, Parameter};

let lights = tool("set_lights")
    .description("Turn the lights in a room on or off.")
    .parameter(Parameter::string("room").choices(["kitchen", "bedroom"]))
    .parameter(Parameter::string("state").choices(["on", "off"]))
    .parameter(Parameter::integer("brightness").at_least(0).at_most(100).optional())
    .build();
```

`tool(..).build()` returns a `serde_json::Value`, so a JSON literal works equally well, as does a file:

```rust
use rustyneedle::ToolFile;

let tools = ToolFile::from_path("examples/tools/car.json")?;
```

A tool file is a JSON array of tools, a single tool, or an object with `tools` and an optional `system` string (the format of `examples/environments/*.json`). Tools are flat `{"name", "description", "parameters"}` objects or the OpenAI `{"type": "function", "function": …}` form. Names are converted to snake_case for the model and mapped back in the answer.

How to write tools that the model handles well is in [tools.md](tools.md): closed sets as enums, bounded numbers, verbatim copy for free text, at most five tools, and `triggers` for requests the model must always route to a tool.

## Ask for a call

```rust
use rustyneedle::Needle;

let mut agent = Needle::builder().model(&model).tool_file(tools).auto_date(false).build();
let response = agent.complete("make the driver side 22 degrees");

for call in &response.function_calls {
    println!("{} {:?}", call.name, call.arguments);
}
```

`complete_within(text, max_new_tokens)` limits the output length (default 512).

| `Response` field | Meaning |
|---|---|
| `function_calls` | calls to act on, after repair and grounding |
| `suppressed_calls` | calls the engine withheld: confidence below 0.1, a negated request, or a failed gate |
| `confidence` | `min(calibrated head, decoding probability)` in `0..=1`, or `None` for an engine without the head |
| `validation` | `ungrounded`: `tool.field` names whose value is not supported by the request; `negation` |
| `failure` | `code` `context_overflow` or `truncated` when the turn could not complete |
| `reasoning` | the model's short derivation of the arguments |
| `kind` | `ResponseKind::Call`, or `Respond` after a tool result produced no further call |
| `results` | handler outputs, set by `run` |

Pick a confidence threshold for your product: act at or above it and ask again or escalate below it. A tool with `triggers` always produces a call for a matching request, so the score is the only signal there. `response.to_json()` renders the object the original Python library returns.

The agent keeps the conversation: later calls to `complete` are follow-up turns on the same tools. `reset()` starts over and `.stateless(true)` does that before every request.

## Run an agent loop

Attach handlers, and `run` executes each call, feeds the result back to the model and repeats until it stops calling tools:

```rust
use rustyneedle::RunOptions;

let mut agent = Needle::builder()
    .model(&model)
    .tool_with_handler(weather, |arguments| Ok(json!({"temp_c": 27})))
    .build();
let response = agent.run("what's the weather in Lagos?", &RunOptions::default());
```

A handler is `FnMut(&Map<String, Value>) -> Result<Value, String> + Send`. An error becomes `{"error": …}` in the results and the loop continues. `RunOptions` has `max_steps` (default 8), `max_new_tokens` and `strict`. In strict mode a call with an ungrounded value is not executed and the result reads `ungrounded <field>`.

## Dates and the system prompt

By default the agent prepends `date: YYYY-MM-DD Day HH:MM` to the system prompt, as the original library does. `.auto_date(false)` leaves it out and `.clock(now)` fixes the time (useful in tests). `.system("locale: en-GB; device: phone")` adds facts: state, never instructions. Whether you want the date line depends on how your tools take dates; read [tools.md](tools.md) before deciding.

## Structured extraction

```rust
use rustyneedle::{extract, ExtractOptions};

let record = extract(&model, schema, "Book a table for four at Luigi's on Friday", &ExtractOptions::default())?;
```

`schema` is a tool schema whose parameters are the fields. The result is `Ok(None)` when nothing matched. With `strict: true` (the default) a value not found in the text is an `ExtractionValidationError`.

## Embeddings

`agent.embed(text)` returns the unit-norm vector (3072 values) read from the model's probe head, so cosine similarity is a dot product. It is `None` for an engine without the head. The head was not trained for retrieval: scores for unrelated sentences sit close to those of related ones (about 0.91 against 0.95 in the `embed` example), so compare scores with each other rather than against a fixed threshold.

## Speech (feature `speech`)

```rust
use std::sync::Arc;
use rustyneedle::speech::{Intent, Options, Whistle};

let whistle = Arc::new(Whistle::open("weights/whistle.cact")?);
let text = whistle.transcribe(&samples, &Options::default())?.text;

let mut pipeline = Intent::new(whistle, agent);
let heard = pipeline.process_file("clip.wav")?;
```

Samples are 16 kHz mono `f32`; `read_wav` loads and resamples a WAV file. `Options` carries the language, keywords and word timestamps. `Intent` passes the enumerated values of your tools to Whistle as keywords, so a spoken room name is transcribed the way the schema spells it. `Stream` takes audio in chunks and reports committed and pending text.

## Test your application without a model

`Needle::builder().engine(Box::new(..))` accepts anything that implements `Engine`, three methods that return a scripted `Response`. The `custom_engine` example is a complete one, and the tests in `crates/rustyneedle/tests/api.rs` use it.

## Threads, memory and speed

- Share one `Arc<Model>`. A `Needle` is `Send` but not `Sync`: give each conversation its own agent and move it between threads as needed.
- Kernels use one process-wide worker pool. `RUSTYNEEDLE_THREADS=n` sets its size (default: cores minus one, at most six). Calls from several threads share it.
- Peak memory is about 140 MB for a request on the 20-layer model; most of the 35 MB archive is memory-mapped.
- The first request after `Needle::build` processes the tool prompt; later requests reuse the cached prefix, so build the agent once and keep it. Typical warm requests take 70 to 85 ms on an M1 Max. See [benchmarking.md](benchmarking.md).
- On Arm CPUs without the dot-product instructions the library selects a slower kernel automatically; [portability.md](portability.md) lists supported chips and the environment variables.

## Errors

| Type | Raised by |
|---|---|
| `ModelError` | `Model::open`, `with_depth`: unreadable or malformed archive, depth out of range |
| `ToolFileError` | `ToolFile::from_path` |
| `ExtractionValidationError` | `extract` in strict mode |
| `MissingEngine` | `NeedleBuilder::try_build` when neither a model nor an engine was given (`build` panics instead) |
| `speech::WhistleError` | opening Whistle, transcribing, reading audio |
| `download::FetchError` | downloads |

A model failure during generation is not an error value: it is `Response::failure`.

## Stability

The version is 0.x: the API can change between minor versions. Only items re-exported by `rustyneedle` are public API; anything else in the implementation crates may change without notice.
