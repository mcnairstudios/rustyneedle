CARGO ?= cargo
PYTHON ?= python3
HUGGING_FACE := https://huggingface.co/Cactus-Compute
CLI := target/release/rustyneedle
BENCH := target/release/rustyneedle-bench
VENV := .venv

.PHONY: help build install weights oracle-assets oracle-env goldens example-clips setup test test-all lint doc bench clean distclean

help:
	@echo "Everything under weights/ and oracle/{assets,goldens,clips} is git-ignored and is filled by these tasks."
	@echo ""
	@echo "build          compile everything in release mode"
	@echo "install        install the rustyneedle command into ~/.cargo/bin"
	@echo "weights        (re)download needle3.cact and whistle.cact from Hugging Face into weights/"
	@echo "setup          populate every git-ignored directory: weights, oracle assets, goldens and example clips"
	@echo "oracle-assets  (re)download the master checkpoint and tokenizer used to record the references"
	@echo "oracle-env     create $(VENV) with the original Python package and JAX"
	@echo "goldens        record the reference outputs into oracle/goldens (needs oracle-assets and oracle-env)"
	@echo "example-clips  synthesise the sample speech clip (macOS only)"
	@echo "test           run the tests that need no downloads"
	@echo "test-all       run every test; runs setup first"
	@echo "lint           check formatting and run clippy"
	@echo "doc            build the library documentation"
	@echo "bench          measure decode, prefill, prompt feed, request and Whistle latency"
	@echo "clean          remove build output"
	@echo "distclean      remove build output and everything the setup tasks downloaded or recorded"

build:
	$(CARGO) build --release

install:
	$(CARGO) install --path crates/cli

weights: build
	$(CLI) download needle3 --out weights
	$(CLI) download whistle --out weights

oracle-assets:
	curl -fL --create-dirs -o oracle/assets/checkpoints/needle3.safetensors $(HUGGING_FACE)/needle3/resolve/main/checkpoints/needle3.safetensors
	curl -fL --create-dirs -o oracle/assets/tokenizer.model $(HUGGING_FACE)/needle3/resolve/main/tokenizer/tokenizer.model

oracle-env:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install cactus-needle sentencepiece numpy jax flax optax safetensors

goldens:
	mkdir -p oracle/goldens oracle/clips
	NEEDLE_TELEMETRY=0 $(VENV)/bin/python oracle/record.py
	NEEDLE_TELEMETRY=0 $(VENV)/bin/python oracle/record_tokenizer.py
	NEEDLE_TELEMETRY=0 $(VENV)/bin/python oracle/record_tensors.py
	NEEDLE_TELEMETRY=0 $(VENV)/bin/python oracle/record_reference.py

example-clips:
	@if command -v say >/dev/null 2>&1; then \
		mkdir -p examples/clips && \
		say -o examples/clips/kitchen_light.wav --data-format=LEI16@16000 "Turn on the overhead light in the kitchen."; \
	else \
		echo "example-clips needs macOS 'say'; elsewhere put any 16 kHz mono WAV in examples/clips/"; \
	fi

setup: weights oracle-assets oracle-env goldens example-clips

test:
	$(CARGO) test --release --workspace --all-features

test-all: setup
	$(CARGO) test --release --workspace --all-features -- --include-ignored

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --release --workspace --all-targets --all-features -- -D warnings

doc:
	$(CARGO) doc --no-deps --all-features -p rustyneedle

bench: build
	$(CARGO) build --release -p rustyneedle-bench
	$(BENCH) decode
	$(BENCH) prefill
	$(BENCH) feed
	$(BENCH) request
	$(BENCH) whistle

clean:
	$(CARGO) clean

distclean: clean
	rm -rf weights oracle/assets oracle/goldens oracle/clips examples/clips $(VENV)
