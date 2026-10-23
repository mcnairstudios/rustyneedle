# The `.cact` archive

Needle 3 and Whistle share one little-endian archive format, tag `0x05E12A84`. The `cact` crate reads it through a memory map and never copies tensor data.

## Layout

| Section | Size | Contents |
|---|---|---|
| Header | 196 bytes | 48 `u32` fields and an `f32` `rope_theta` |
| Codebook | `codebook_len` × 4 | Lloyd-Max unit-sphere codebooks: 2-bit (4), 3-bit (8), 4-bit (16) |
| Directory | `num_tensors` × 44 | One nameless record per tensor |
| Tensor blobs | 64-byte aligned | Addressed by `offset` and `nbytes` |

Header field order: tag, num_tensors, codebook_len, kv_window, kv_bits, vocab, out_vocab, d_model, num_heads, num_kv_heads, num_layers, qk_head_dim, v_head_dim, max_seq_len, hada_n, mhc_lanes, sliding_window, global_mask_lo, global_mask_hi, qkv_conv_taps, engram_slots, engram_sub_dim, num_engram_tables, engram_conv_taps, engram_conv_dilation, engram_seed_heads, num_engram_orders, engram_orders[4], num_engram_sites, engram_sites[16], rope_theta.

Directory record: `u8 kind, u8 rank, u16 pad, u32 shape[4], u64 offset, u64 nbytes, u32 group_size, u32 bits`. Kind is 1 half, 2 single, 3 codebook-quantised, 4 raw.

## Canonical tensor order

Tensors are positional. `Layout::resolve` walks the directory in this order and fails if any tensor is left over:

1. embedding
2. per decoder layer, 24 tensors, or 27 when `qkv_conv_taps > 0`: norm_in, q, k, v, [q/k/v taps], q_norm, k_norm, gate_proj, out_proj, post_norm, attn_gate, pre_hada, then the 13 Hadamard tensors
3. nine multi-lane hyper-connection tensors, then two Hadamard permutations
4. per engram site: tables, key_proj, value_proj, taps
5. final_norm
6. optional probe heads: a half-precision manifest, then six tensors per head (router adds calibration)
7. optional audio encoder, introduced by a single-precision manifest whose first value is `3246`
8. the raw tokenizer blob

## Codebook-quantised matrices

A logical `[out, in]` matrix pads `in` to a multiple of `group_size`. The blob holds the packed indices (`out × in_pad × bits / 8` bytes, least-significant bit first) and then the per-group L2 norms as `out × in_pad / group_size` halves. A group is reconstructed as `(codebook[idx] × norm) · H` where `H` is the normalised Walsh-Hadamard matrix of the group size. Bit width varies per tensor; `bits = 5` marks ternary crumbs.

## Tokenizer blob

Header `u32 pieces, pad, eos, bos, unk, u8 add_dummy_prefix, u8 byte_fallback, u16 pad`, then per piece `f32 score, u8 type, u16 length, bytes`.
