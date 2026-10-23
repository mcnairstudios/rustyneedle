import os
import sys
import time

os.environ["NEEDLE_TELEMETRY"] = "0"
import needle

weights = sys.argv[1] if len(sys.argv) > 1 else "weights/whistle.cact"
clip = sys.argv[2] if len(sys.argv) > 2 else "examples/clips/kitchen_light.wav"
started = time.perf_counter()
needle.transcribe(clip, weights=weights)
first = time.perf_counter() - started
started = time.perf_counter()
result = needle.transcribe(clip, weights=weights)
second = time.perf_counter() - started
print("first call %.0f ms, second call %.0f ms" % (first * 1e3, second * 1e3), {key: result.get(key) for key in ("text", "ttft_ms", "decode_tps")})
