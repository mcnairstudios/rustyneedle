import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from needle.model.architecture import SimpleAttentionNetwork
from needle.model.finetune import render_example
from needle.model.quantize import _is_quant_leaf, _reduces_second_last, configure_deploy, cq_quantize
from needle.model.run import load_checkpoint
from needle.model.tokenizer import get_tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(ROOT, "assets", "checkpoints", "needle3.safetensors")
TOKENIZER = os.path.join(ROOT, "assets", "tokenizer.model")

PROMPTS = [
    ("turn on the kitchen lights", [{"name": "set_lights", "description": "Turn lights on or off.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}}, "required": ["room", "on"]}}]),
    ("hello", []),
    ("what is the weather in Lagos", [{"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]),
]


def path_of(path):
    return "/".join(getattr(k, "key", getattr(k, "name", str(k))) for k in path)


def quantise(path, leaf):
    if not _is_quant_leaf(path, leaf):
        return leaf
    name = path_of(path)
    bits = 4 if name == "embedding/embedding" or "mhc_phi" in name else 2
    leaf = jnp.asarray(leaf)
    if _reduces_second_last(path):
        return jnp.swapaxes(cq_quantize(jnp.swapaxes(leaf, -1, -2), bits), -1, -2)
    return cq_quantize(leaf, bits)


def main():
    configure_deploy(act_bits=8, kv_bits=8)
    params, config = load_checkpoint(CHECKPOINT)
    config = dataclasses.replace(config, dtype="float32")
    qparams = jax.tree_util.tree_map_with_path(quantise, params)
    model = SimpleAttentionNetwork(config)
    from needle.model.tokenizer import SANTokenizer
    tokenizer = SANTokenizer(TOKENIZER)
    out = []
    for query, tools in PROMPTS:
        prompt, _ = render_example({"query": query, "tools": tools})
        ids = [2] + tokenizer.encode(prompt)
        tokens = jnp.array([ids])
        logits = np.asarray(model.apply({"params": qparams}, tokens, quant=True))[0]
        cells = np.asarray(model.apply({"params": qparams}, tokens, True, method=SimpleAttentionNetwork.hidden_cells))
        out.append({
            "query": query,
            "ids": ids,
            "last_logits": logits[-1].tolist(),
            "first_logits": logits[0].tolist(),
            "cells_last": cells[0, -1].tolist(),
            "cells_first": cells[0, 0].tolist(),
        })
        print(query, len(ids), int(np.argmax(logits[-1])), flush=True)
    json.dump({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(config).items()}, "cases": out}, open(os.path.join(ROOT, "goldens", "reference.json"), "w"))


main()
