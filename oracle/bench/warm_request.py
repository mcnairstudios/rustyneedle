import json
import os
import sys
import time

os.environ["NEEDLE_TELEMETRY"] = "0"
import needle

weights = sys.argv[1] if len(sys.argv) > 1 else "weights/needle3.cact"
tools = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "examples/tools/home_assistant.json"))
queries = ["turn on the overhead light in the kitchen", "set the volume to 30", "pause the media player", "turn off the thermostat"]
agent = needle.Needle(tools=tools, weights=weights)
agent.complete(queries[0])
agent.reset()
samples = []
for round in range(12):
    started = time.perf_counter()
    agent.complete(queries[round % len(queries)])
    samples.append((time.perf_counter() - started) * 1000)
    agent.reset()
samples.sort()
print("warm request: median %.1f ms, best %.1f ms" % (samples[len(samples) // 2], samples[0]))
