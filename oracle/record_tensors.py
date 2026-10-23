import json
import os

import numpy as np

from needle.model.export import read_export

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ROWS = (0, 1, 7, 131, 5000)


def sample(name, index, tensors):
    matrix = tensors[index]
    rows = [r for r in SAMPLE_ROWS if r < matrix.shape[0]]
    rng = np.random.RandomState(index)
    x = rng.randn(matrix.shape[1]).astype(np.float32)
    product = matrix.astype(np.float64) @ x.astype(np.float64)
    return {
        "archive": name,
        "tensor": index,
        "rows": rows,
        "values": [matrix[r].astype(float).tolist() for r in rows],
        "input": x.astype(float).tolist(),
        "product": product.tolist(),
        "columns": int(matrix.shape[1]),
    }


def main():
    out = []
    for name, indexes in (("needle3", (0, 1, 2, 10, 11, 5 + 20 * 27 + 11 + 0)), ("whistle", (0, 2))):
        _, tensors = read_export(os.path.join(ROOT, "..", "weights", f"{name}.cact"))
        for index in indexes:
            if getattr(tensors[index], "ndim", 0) == 2 and tensors[index].shape[0] > 8:
                out.append(sample(name, index, tensors))
    json.dump(out, open(os.path.join(ROOT, "goldens", "tensors.json"), "w"))
    print(len(out), "tensors")


main()
