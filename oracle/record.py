import json
import os
import subprocess
import sys

os.environ["NEEDLE_TELEMETRY"] = "0"

import needle
from needle.environments import ENVIRONMENTS

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, "..", "weights", "needle3.cact")
WHISTLE = os.path.join(ROOT, "..", "weights", "whistle.cact")
FIXED_DATE = "date: 2026-07-21 Tue 14:30"
TIMING_KEYS = ("prefill_tps", "decode_tps", "peak_ram_mb", "ttft_ms", "pass_ms")

EXTRACTION_CASES = [
    {
        "name": "invoice",
        "text": "Invoice 4471 from Northwind Traders dated 3 March 2025, total 1,250.00 EUR, due in 30 days.",
        "schema": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "vendor": {"type": "string"},
                "date": {"type": "string"},
                "total": {"type": "number"},
                "currency": {"type": "string", "enum": ["EUR", "USD", "GBP"]},
            },
            "required": ["invoice_number", "vendor", "total", "currency"],
        },
    },
    {
        "name": "booking",
        "text": "Book a table for four at Luigi's on Friday at 7:30pm, gluten free options please.",
        "schema": {
            "type": "object",
            "properties": {
                "restaurant": {"type": "string"},
                "party_size": {"type": "integer"},
                "day": {"type": "string"},
                "time": {"type": "string"},
                "dietary": {"type": "string"},
            },
            "required": ["restaurant", "party_size"],
        },
    },
    {
        "name": "notification_class",
        "text": "Your package has been delayed due to weather and will now arrive Thursday.",
        "schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": ["delivery", "billing", "security", "social"]}},
            "required": ["category"],
        },
    },
]

SENTENCES = ["hello world", "turn off the kitchen lights", "what is the weather in Lagos", "", "a"]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def record_environments():
    out = []
    for name, module in ENVIRONMENTS.items():
        system = FIXED_DATE + "; " + module.SYSTEM if module.SYSTEM else FIXED_DATE
        agent = needle.Needle(tools=module.TOOLS, system=system, weights=WEIGHTS)
        for case in module.TEST_CASES:
            agent.reset()
            response = agent.complete(case["query"])
            out.append({"environment": name, "query": case["query"], "expected_calls": case["calls"], "response": clean(response)})
        schemas = [getattr(t, "_needle_tool", None) for t in module.TOOLS]
        out.append({"environment": name, "tool_schemas": schemas, "system": system})
    return out


def record_extraction():
    out = []
    for case in EXTRACTION_CASES:
        tool = {"name": case["name"], "description": "Record the extracted fields.", "parameters": case["schema"]}
        try:
            result = needle.extract(case["text"], tool, weights=WEIGHTS, strict=False)
        except Exception as error:
            result = {"error": str(error)}
        out.append({"name": case["name"], "text": case["text"], "tool": tool, "result": result})
    return out


def record_embeddings():
    agent = needle.Needle(tools=[], weights=WEIGHTS)
    return [{"text": text, "vector": agent.embed(text)} for text in SENTENCES]


def speak(text, path):
    subprocess.run(["say", "-o", path, "--data-format=LEI16@16000", text], check=True)


def record_speech():
    queries = []
    for module in ENVIRONMENTS.values():
        queries += [case["query"] for case in module.TEST_CASES[:4]]
    whistle = needle.Whistle(weights=WHISTLE)
    out = []
    for index, query in enumerate(queries):
        path = os.path.join(ROOT, "clips", f"clip{index:02d}.wav")
        speak(query, path)
        plain = whistle.transcribe(path)
        timed = whistle.transcribe(path, word_timestamps=True)
        out.append({"clip": os.path.basename(path), "spoken": query, "plain": clean(plain), "timed": clean(timed)})
    return out


def main():
    parts = {
        "environments": record_environments,
        "extraction": record_extraction,
        "embeddings": record_embeddings,
        "speech": record_speech,
    }
    wanted = sys.argv[1:] or list(parts)
    for name in wanted:
        data = parts[name]()
        with open(os.path.join(ROOT, "goldens", f"{name}.json"), "w") as handle:
            json.dump(data, handle, indent=1, ensure_ascii=False)
        print(name, len(data))


main()
