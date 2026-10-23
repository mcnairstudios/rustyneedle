import json
import os

import sentencepiece as spm

ROOT = os.path.dirname(os.path.abspath(__file__))
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
TOOLS_START, TOOLS_END = "<tools>", "</tools>"

EXTRA = [
    "",
    "a",
    " leading space",
    "trailing space ",
    "double  space",
    "tabs\tand\nnewlines\n\n",
    "Ünïcödé façade naïve café",
    "日本語のテキストと한국어",
    "emoji 😀 and 🚀 mixed",
    "numbers 1,250.00 and 3.14159 and 2025-03-03",
    "<think>reasoning</think>",
    "<tool_call>[{\"name\":\"x\",\"arguments\":{}}]</tool_call>",
    "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    "UPPER lower MiXeD",
    "a" * 300,
    "word " * 120,
    " nbsp and ​zero width",
    "ﬁligature and ① circled",
]


def collect_texts():
    texts = list(EXTRA)
    for name in ("environments", "speech"):
        data = json.load(open(os.path.join(ROOT, "goldens", f"{name}.json")))
        for item in data:
            for key in ("query", "spoken"):
                if key in item:
                    texts.append(item[key])
            if "response" in item:
                texts.append(item["response"].get("reasoning") or "")
                texts.append(json.dumps(item["response"].get("function_calls"), separators=(",", ":")))
            if "tool_schemas" in item:
                body = json.dumps(item["tool_schemas"], separators=(",", ":"), ensure_ascii=False)
                texts.append(f"{IM_START}user\n{TOOLS_START}{body}{TOOLS_END}\nturn on the lights{IM_END}\n{IM_START}assistant\n")
    return [t for t in texts if t is not None]


def main():
    model = spm.SentencePieceProcessor()
    model.Load(os.path.join(ROOT, "assets", "tokenizer.model"))
    cases = []
    for text in collect_texts():
        ids = model.Encode(text, out_type=int)
        cases.append({"text": text, "ids": ids, "decoded": model.Decode(ids)})
    json.dump({"pieces": model.GetPieceSize(), "cases": cases}, open(os.path.join(ROOT, "goldens", "tokenizer.json"), "w"), ensure_ascii=False)
    print(len(cases), "cases", model.GetPieceSize(), "pieces")


main()
