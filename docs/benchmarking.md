# Benchmarking

## This port

`make bench` builds `rustyneedle-bench` and runs the main measurements. The subcommands can be run singly:

| Command | Measures |
|---|---|
| `rustyneedle-bench decode [--context 600]` | single-token decode throughput at a given context length |
| `rustyneedle-bench prefill [--tokens 4000]` | batched prefill throughput |
| `rustyneedle-bench feed [--prefix 1100 --turn 25]` | the time to append a short user turn to a cached prefix |
| `rustyneedle-bench reset [--prefix 1100]` | the cost of cloning a cached session for a new request |
| `rustyneedle-bench request [--tools examples/tools/home_assistant.json]` | a full warm tool-call request, median and best of the rounds |
| `rustyneedle-bench whistle [--clip file.wav]` | first and warm transcription of a clip |
| `rustyneedle-bench attention` and `rustyneedle-bench primitives` | kernel micro-benchmarks |

`--weights` selects the directory holding `needle3.cact` and `whistle.cact` (default `weights`). Environment variables that change the result: `RUSTYNEEDLE_THREADS`, `RUSTYNEEDLE_NO_DOTPROD`, `RUSTYNEEDLE_FLOAT_KERNELS` (see [portability.md](portability.md)).

## The original

`oracle/bench/` holds the same measurements against the Python package and its native engine, using the virtualenv from `make oracle-env`:

```sh
.venv/bin/python oracle/bench/warm_request.py     # median and best warm request
.venv/bin/python oracle/bench/cold_start.py       # import, init and first request, with the engine's own prefill and decode rates
.venv/bin/python oracle/bench/whistle.py          # first and second transcription
```

Run each tool with `/usr/bin/time -l` to compare whole-process time and peak resident memory. Compare like with like: both implementations should use the same tool file, the same queries and the same machine state, and the Python figures include its call overhead.

## Reading the numbers

Timings on a shared or power-limited machine swing between two states: a request that takes 83 ms can take 270 ms a minute later with nothing changed in the code, and the original slows down in the same way. The slow state follows heavy work on the same machine (a build or a full test run) and clears after about a minute of idleness. Before trusting a comparison, wait that minute, check `uptime` and that nothing else is busy; run the two sides alternately several times; and compare the best figure of each run as well as the median. The measured gap is the difference between runs taken in the quiet state, not between a quiet run of one and a noisy run of the other.
