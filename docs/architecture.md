# Architecture

## Data flow

```
text ──► Needle ──► tokenizer ──► model::Session ──► logits ──► grammar mask ──► greedy token
                                       │                                              │
                                       └── probe head (confidence, embedding)         ▼
                                                                        calls JSON, reasoning
                                                                                      │
                                                         repair ◄─────────────────────┘
                                                            │
                                         gates (negation, floor, quantity, grounding)
                                                            │
                                                            ▼
                                               Response { function_calls, suppressed_calls, … }

audio ──► whistle (log-mel ► stem ► encoder) ──► cross-attention decoder ──► transcript ──► Needle
```

## Model

A Needle 3 block is read directly from the archive. Each token carries four residual lanes. Per layer, a learned routing step mixes the lanes into one input vector (`model::lanes`), the block runs, and its change is spread back over the lanes. The block is gated attention (grouped queries, causal convolution taps on Q, K and V, int8 KV cache, a 1024-token window with four global layers) followed by a Hadamard MLP (three Kronecker products of 32×32 factors with two fixed permutations). Engram sites add a hashed n-gram memory read by gather. After the last layer the lanes are averaged and normalised.

A probe head pools per-layer states over the whole conversation with an online softmax, so the session never stores per-token states. The confidence head is the same pooled readout projected to one logit; `embed` is its unit-normalised readout.

Whistle reuses the same decoder, adding cross-attention to an encoder that runs the same lane machinery over 80 ms frames.

## Numerics

Weights are codebook-quantised: each group of 128 is stored rotated by a Walsh-Hadamard matrix, as an fp16 norm and low-bit codebook indices. Because the rotation is symmetric, a dot product with the original weights equals a dot product of the codes with the rotated activation, so each activation is rotated once and reused by every output row.

The default kernel quantises the rotated activation to int8 per group and accumulates with a 4-way int8 dot product. Weights are repacked on first use into panels of four rows so one `sdot` by lane updates four rows at once: a 16-byte block holds 16 columns for four rows, is expanded to int8 with NEON `tbl`, and is shared across a tile of four tokens when several are batched. The repack is a small bit transpose of packed bytes and takes about 100 ms for the whole model. This is roughly 9× faster than the float kernel and adds about 0.7% relative RMS error, small next to the 2-bit weight error itself. Chips without the dot-product extension use widening multiplies with identical integer sums (see `portability.md`). Set `RUSTYNEEDLE_FLOAT_KERNELS=1` for the exact float path. Without NEON the float path is used.

Activations are fake-quantised to int8 at the same points as the reference: before each projection, after attention, before the head, and for the lane routing input. Queries, keys and values are quantised per head.

## Performance

On an Apple M1 Max, with the weights in this repository:

| | Rate |
|---|---|
| Needle decode | about 830–860 tokens/s at 600 tokens of context, about 720 at 1,100 |
| Needle prefill, 4,000 tokens | about 1,800 tokens/s |
| Prompt feed, 25 tokens after a 1,100-token cached prefix | about 14.5 ms |
| Warm tool-call request (about 12 prompt tokens, about 55 generated) | about 82–85 ms median, 68–70 ms best when the machine is quiet |
| Cold process, one tool call including loading and the 1,100-token prefix | about 0.7 to 1 s |
| Logits matvec (8192 × 768) | about 0.04 ms |
| Whistle | a 2-second clip transcribes in 16 ms warm (original 15–16 ms), 0.04 s as a whole process including start, about 2,100 decoded tokens/s |

The original engine measured on the same machine and request: about 775 tokens/s decode and 76–78 ms median per warm request when the machine is quiet. When the machine is power-limited both slow down, and the original slows more (about 119 ms against 90 ms in the same paired run).

Work is split over a pool of spinning worker threads (`kernels::pool`). Workers spin for 300 µs after the last job and then park, so a warm service pays about 3 µs per fork-join and an idle one uses no CPU. The pool is created on first use. Decode forks three times per layer (the fused q/k/v/gate projections, the attention heads, the output projection); batched prefill runs per-token passes and one fused product for the four attention projections. Attention keeps keys as int8 rows and values in 4-position panels, so softmax weights (12-bit, split into two 6-bit halves) accumulate with `sdot` and the integer sums are exact. The toolset prefix is prefilled once and cloned for each query, so a query costs its own tokens plus the decode.

## Why a trait between the model and the agent

`rustyneedle::Engine` is the seam between the model-backed engine (model, grammar, repair and gates) and the agent layer (date facts, year licensing, number grounding, the run loop, extraction validation). The agent's behaviour is therefore testable with canned responses, which is how the original project's grounding tests are ported.
