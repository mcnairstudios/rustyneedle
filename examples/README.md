# Examples

Everything here runs against the model files that `make weights` downloads into `weights/`.

## Tool sets

Use them with `--tools` on the command line or `ToolFile::from_path` in Rust.

| File | Tools | Notes |
|---|---|---|
| `tools/home_assistant.json` | 8 | Home Assistant intents (lights, thermostat, media) with device names as enums; generated from the Open Home Foundation's intent definitions (see `NOTICE`) |
| `tools/weather_and_time.json` | 2 | `triggers` separate "what time is it" from weather |
| `tools/calendar.json` | 3 | date and time phrases are copied verbatim for the application to resolve |
| `tools/car.json` | 4 | zone and window enums, a bounded temperature, free-text destination, a call trigger |
| `tools/customer_support.json` | 4 | order numbers copied verbatim, a refund reason enum, a hand-off trigger |
| `environments/smart_home.json` | 5 | |
| `environments/media_player.json` | 5 | |
| `environments/productivity.json` | 5 | |
| `environments/wearable.json` | 5 | |
| `environments/kitchen_appliance.json` | 5 | |
| `environments/data_capture.json` | 5 | |

The files in `environments/` also carry a `system` prompt and frozen `cases` (a request and the calls expected); `rustyneedle-envs` runs them as acceptance suites.

```sh
./target/release/rustyneedle complete --model weights/needle3.cact --tools examples/tools/car.json --no-date "make the driver side 22 degrees"
./target/release/rustyneedle complete --model weights/needle3.cact --tools examples/environments/smart_home.json "dim the study lights to 30 percent"
```

`--no-date` leaves out the `date:` line the engine otherwise adds; use it for tool sets that take no dates (see `docs/tools.md`).

## Rust programs

They live in `crates/rustyneedle/examples/` and are run from the repository root.

```sh
cargo run --release -p rustyneedle --example quickstart
cargo run --release -p rustyneedle --example tool_file -- examples/tools/car.json "open both rear windows"
cargo run --release -p rustyneedle --example extract
cargo run --release -p rustyneedle --example embed
cargo run --release -p rustyneedle --example custom_engine
cargo run --release -p rustyneedle --features speech --example speech -- examples/tools/home_assistant.json clip.wav
cargo run --release -p rustyneedle --features download --example download
```

`custom_engine` needs no model files. `speech` takes any 16 kHz mono WAV; on macOS `make example-clips` records `examples/clips/kitchen_light.wav`.

## Script

`examples/try.sh` builds the CLI if needed and runs the text, speech-to-text and speech-to-intent paths in turn.
