#!/bin/sh
set -e
cd "$(dirname "$0")/.."
[ -x target/release/rustyneedle ] || cargo build --release
N=./target/release/rustyneedle
TOOLS=examples/tools/home_assistant.json
CLIP=examples/clips/kitchen_light.wav
echo "== text to tool call"
$N complete --model weights/needle3.cact --tools $TOOLS --no-date --pretty "turn on the overhead light in the kitchen"
if [ ! -f "$CLIP" ]; then
  echo "no $CLIP: run 'make example-clips' on macOS, or put any 16 kHz mono WAV there"
  exit 0
fi
echo "== speech to text"
$N transcribe --whistle weights/whistle.cact "$CLIP"
echo "== speech to tool call"
$N intent --model weights/needle3.cact --whistle weights/whistle.cact --tools $TOOLS --no-date --pretty "$CLIP"
