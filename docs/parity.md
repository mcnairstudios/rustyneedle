# Parity with the original

The original project's engine is a closed C++ library, so behaviour is checked against recordings of it rather than against its source. `oracle/` holds the recorder scripts and `oracle/goldens` the recordings; the Rust tests read them through the public interfaces.

## Oracles

| Oracle | Recorded by | Used for |
|---|---|---|
| The shipped engine (`needle.Needle`, `needle.extract`, `needle.Whistle`) over the six environments' 192 cases, three extraction cases, five embeddings and 24 synthesised speech clips | `oracle/record.py` | end-to-end responses, extraction, embeddings, transcripts, word times |
| SentencePiece itself, over 624 strings | `oracle/record_tokenizer.py` | tokenizer |
| The exporter's own reader, on sampled tensors | `oracle/record_tensors.py` | dequantisation and products |
| A float32 JAX forward pass with quantisation simulated, as the porting guide prescribes | `oracle/record_reference.py` | logits and per-layer states |

The date fact in the recorded prompts is frozen (`date: 2026-07-21 Tue 14:30`) so the recordings are reproducible. The agent puts the current time into the prompt, and a borderline case can flip with it (a nested-object tool returned a call for some minutes and none for others), so every test that runs the real model pins the clock.

None of the recordings, the sampled weight rows or the synthesised clips are committed. `make weights`, `make oracle-assets`, `make oracle-env` and `make goldens` produce them, and the tests that need them are marked `#[ignore]` so `make test` runs without any download and `make test-all` runs everything.

## Results

| Check | Result |
|---|---|
| Tokenizer vs SentencePiece | 624 of 624 identical, encoding and decoding |
| Dequantised weight rows vs the exporter | within 1e-4 |
| Quantised products vs the dense reference | 0.7% relative RMS (int8 kernel), exact with the float kernel |
| Logits vs the quantised reference | cosine above 0.999 and the same top token on every prompt; every layer above 0.995 |
| Batched prefill vs token-at-a-time | cosine above 0.9995 |
| Environment responses vs the engine | 172 of 192 identical (calls, withheld calls, validation, error code) |
| Environment acceptance suites | 175 of 192, equal to the engine's own 175 of 192 |
| Confidence vs the engine | mean absolute error 0.048 |
| Extraction vs the engine | 3 of 3 identical |
| Embeddings vs the engine | cosine above 0.998 |
| Whistle transcripts vs the engine | 17 of 24 identical, the rest within a word or two |
| Whistle word times | mean start error 0.056 s, end 0.063 s, probability 0.020 |
| Speech to intent vs the text pipeline | 24 of 24 clips reach the same tools |

The original grounding tests (31), the tool-schema tests, the environment registry and harness tests, the agent-level weights tests and the real-engine inference tests are ported. Tests that exercise Python packaging, JAX/MLX, LoRA, the wheel fetcher or Needle 2 are not.

## Known differences

- **Negation.** The original flags `negation` identically but moves the call to `suppressed_calls` for only about half the negated requests, by a rule that is not in any source. This port withholds all of them. Callers that treat a negation flag as a refusal, as the acceptance suites do, see no difference.
- **Out-of-range numbers.** When the model's number falls outside the schema bounds the original sometimes returns an empty call or a truncation error where this port returns the clamped value with `validation.ungrounded` set.
- **Greedy ties.** The two implementations accumulate in different orders and activations are quantised to int8, so a near-tied choice can flip. This is the source of the few differing transcripts and tool choices.
- **Keyword biasing.** The mechanism (a token trie with a start bonus and a continuation bonus, applied only to plausible tokens) is documented behaviour; the constants are this port's own.
- **Word times.** The original does not publish its alignment. This port runs a monotonic alignment over the decoder's cross-attention; start times are one frame off more often than the original's.

## Regenerating the recordings

```sh
make weights oracle-assets oracle-env
make goldens
```

`make goldens` runs the four recorder scripts with telemetry off. The reference script needs the master checkpoint at `oracle/assets/checkpoints/needle3.safetensors` and the tokenizer model at `oracle/assets/tokenizer.model`, both fetched by `make oracle-assets` from `Cactus-Compute/needle3`. The speech clips are synthesised with the macOS `say` command. Quantise kernels and the `mhc_phi` maps along their second-last axis when building the reference, as `record_reference.py` does.
