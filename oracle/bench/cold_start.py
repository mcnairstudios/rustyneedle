import json
import os
import sys
import time

started = time.perf_counter()
os.environ["NEEDLE_TELEMETRY"] = "0"
import needle

imported = time.perf_counter()
weights = sys.argv[1] if len(sys.argv) > 1 else "weights/needle3.cact"
tools = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "examples/tools/home_assistant.json"))
agent = needle.Needle(tools=tools, weights=weights)
initialised = time.perf_counter()
response = agent.complete("turn on the overhead light in the kitchen")
answered = time.perf_counter()
print("import %.0f ms, init %.0f ms, first request %.0f ms, total %.0f ms" % ((imported - started) * 1e3, (initialised - imported) * 1e3, (answered - initialised) * 1e3, (answered - started) * 1e3))
print({key: response.get(key) for key in ("prefill_tps", "decode_tps", "peak_ram_mb", "confidence")})
