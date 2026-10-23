# Portability

## What runs where

| Target | Status | Kernel path |
|---|---|---|
| Apple silicon (M1 and later) | Tested | NEON with the `sdot` dot-product instruction |
| 64-bit Arm Linux with dot product (Raspberry Pi 5, Graviton, Ampere, Cortex-A76 and later) | Builds; same code path as Apple silicon | NEON with `sdot` |
| 64-bit Arm Linux without dot product (Raspberry Pi 3, 4 and Zero 2 W, Cortex-A53 and A72) | Builds; the fallback is tested on Apple silicon by forcing it | NEON with widening multiplies |
| x86-64 | Builds and passes the kernel and model tests under Rosetta | Portable float kernels |
| 32-bit Arm (32-bit Raspberry Pi OS) | Builds | Portable float kernels, no SIMD |
| Microcontrollers (ESP32 family) | Not supported, see below | |

## Runtime selection on Arm

The integer kernels need a 4-way int8 dot product. Chips with the Armv8.2 dot-product extension use `sdot`. Everything else uses `vmull_s8` followed by pairwise adds, which every Armv8 NEON chip has. Both produce identical integer sums, so results do not depend on which one runs.

The choice is made once per process with `is_aarch64_feature_detected!("dotprod")`. Set `RUSTYNEEDLE_NO_DOTPROD=1` to force the fallback, which is how the fallback is tested on a chip that has the extension.

On the development machine (M1 Max, 6 threads) the fallback decodes at about 640 tokens per second against about 830 with `sdot`. Smaller cores will be slower in proportion; no measurement exists on a Raspberry Pi.

## Other switches

| Variable | Effect |
|---|---|
| `RUSTYNEEDLE_NO_DOTPROD=1` | Use the widening-multiply kernels on any Arm chip |
| `RUSTYNEEDLE_FLOAT_KERNELS=1` | Use the exact float kernels instead of the integer ones |
| `RUSTYNEEDLE_THREADS=n` | Total threads in the worker pool (default: cores minus one, at most 6) |

## Microcontrollers

An ESP32-class target is a separate port, not a build flag.

- **Flash and memory.** The Needle 3 archive is about 35 MB and the Whistle archive about 17 MB. Most ESP32 modules carry 4 to 16 MB of flash. Only parts with at least 32 MB of flash (some ESP32-S3 boards, the ESP32-P4) can hold Needle alone. The key and value cache for a 1,100-token context is about 5 MB plus the same again for the value panels, so it needs external RAM or a much shorter cached prefix.
- **Speed.** Each generated token reads roughly the whole weight set. At about 100 MB/s from external flash that bounds an ESP32-S3 to a few tokens per second at best. The depth ladder (`--depth`) runs fewer layers of the same weights and is the main lever on devices this small. The ESP32-C3 and C6 have neither SIMD nor a hardware float unit and would be several times slower again.
- **Code.** The engine uses memory-mapped files, a regex engine and an operating-system thread pool. A port needs the core split into a layer that takes weights as a byte slice, runs on one thread and uses its own int8 kernels, with the grammar, tokenizer and repair stages kept portable.
