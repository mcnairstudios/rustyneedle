# Command line

`rustyneedle` has one subcommand per capability and a compact form that mirrors the original engine binary. Every command prints one JSON object (add `--pretty` where offered). Model paths default to the cached base weights (`~/.cache/rustyneedle`, or `RUSTYNEEDLE_CACHE_DIR`), downloaded on first use; set `HF_HUB_OFFLINE=1` to forbid downloads.

| Command | Purpose |
|---|---|
| `rustyneedle complete --model needle3.cact --tools tools.json [--system …] [--depth N] "query"` | one turn of tool calling |
| `rustyneedle extract --model needle3.cact --schema schema.json [--no-strict] "text"` | structured extraction; the schema is a tool schema or a bare JSON schema |
| `rustyneedle embed --model needle3.cact "text"` | the 3072-value unit-norm embedding |
| `rustyneedle transcribe --whistle whistle.cact [--language en] [--keywords a,b] [--keywords-file f] [--word-timestamps] [--audio-depth N] clip.wav` | speech to text |
| `rustyneedle stream --whistle whistle.cact [--chunk-seconds 1] clip.wav` | live transcription, one JSON line per chunk |
| `rustyneedle intent --model needle3.cact --whistle whistle.cact --tools tools.json clip.wav` | audio in, tool calls out |
| `rustyneedle download needle3 \| whistle \| org/repo[/file.cact] [--out dir]` | fetch weights |
| `rustyneedle playground [--model needle3.cact] [--host 127.0.0.1] [--port 7860]` | the browser playground |
| `rustyneedle --model needle3.cact --model whistle.cact --tools tools.json --audio clip.wav` | compact form; two models mean speech to intent, one model with `--query` is a text turn |

`--tools` and `--schema` take a file path or inline JSON. A tools file is an array of tools, a single tool, or an object with `tools` and an optional `system` string, as in `examples/environments/`; `--system` overrides the file's. Tools are flat `{"name", "description", "parameters"}` objects or the OpenAI `{"type": "function", "function": …}` form; names are converted to snake_case for the model and mapped back in the answer. A tool may carry `"triggers"`, a list of case-insensitive regular expressions that force a call to that tool.

`--no-date` (on `complete`, `intent` and the compact form) leaves out the `date:` fact that is otherwise added to the system prompt. Use it for tool sets that take no dates or times; see [tools.md](tools.md).

`--depth N` runs the N-layer rung of the same weights (2 to 20 for Needle) and `--audio-depth N` does the same for Whistle's decoder (2 to 8).

Fine-tuning, `.cact` export, data generation and the platform commands remain in the Python package.

## Serving

`rustyneedle playground` is also the HTTP server: `GET /` is the page, `GET /model` names the loaded archive, `POST /complete` takes `{"tools": [...], "query": "..."}` and returns the response object, `POST /reset` clears the conversation and `POST /load-model` (body: a `.cact`, header `X-Filename`) swaps the weights. The original engine's native `--serve` mode has no published protocol and is not reproduced.
