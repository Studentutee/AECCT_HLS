#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


CHECKPOINT_ORDER = [
    "preproc_x",
    "layer0_ln_in",
    "layer0_q",
    "layer0_k",
    "layer0_v",
    "layer0_attn_scores",
    "layer0_attn_probs",
    "layer0_ctx",
    "layer0_attn_out",
    "layer0_ln_out",
    "layer0_ffn1_out",
    "layer0_act_out",
    "layer0_ffn2_out",
    "layer0_ffn_ln_out",
    "layer1_ln_in",
    "layer1_q",
    "layer1_k",
    "layer1_v",
    "layer1_attn_scores",
    "layer1_attn_probs",
    "layer1_ctx",
    "layer1_attn_out",
    "layer1_ln_out",
    "layer1_ffn1_out",
    "layer1_act_out",
    "layer1_ffn2_out",
    "layer1_ffn_ln_out",
    "final_node_logits",
    "final_out_fc_in",
    "final_logits",
    "final_x_pred",
]


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    inf_a = np.isinf(a)
    inf_b = np.isinf(b)
    inf_mismatch = bool(np.any(inf_a != inf_b))

    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(finite):
        diff = a[finite] - b[finite]
        maxabs = float(np.max(np.abs(diff)))
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
    else:
        maxabs = 0.0
        mae = 0.0
        rmse = 0.0

    return {
        "maxabs": maxabs,
        "mae": mae,
        "rmse": rmse,
        "inf_mismatch": inf_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python/C++ checkpoint dumps")
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    args = parser.parse_args()

    first_diverge = None

    print("=== checkpoint compare (python vs cpp) ===")
    print("name, maxabs, mae, rmse, inf_mismatch")

    for name in CHECKPOINT_ORDER:
        p_py = args.python_dir / f"{name}.npy"
        p_cpp = args.cpp_dir / f"{name}.npy"

        if not p_py.exists() or not p_cpp.exists():
            print(f"{name}, MISSING, MISSING, MISSING, MISSING")
            if first_diverge is None:
                first_diverge = {
                    "name": name,
                    "reason": "missing checkpoint",
                }
            continue

        a = np.load(p_py)
        b = np.load(p_cpp)

        if a.shape != b.shape:
            print(f"{name}, SHAPE_MISMATCH {a.shape} vs {b.shape}, -, -, -")
            if first_diverge is None:
                first_diverge = {
                    "name": name,
                    "reason": f"shape mismatch {a.shape} vs {b.shape}",
                }
            continue

        res = compare_arrays(a, b)
        print(
            f"{name}, {res['maxabs']:.6e}, {res['mae']:.6e}, "
            f"{res['rmse']:.6e}, {res['inf_mismatch']}"
        )

        diverged = res["inf_mismatch"] or (res["maxabs"] > args.threshold)
        if diverged and first_diverge is None:
            first_diverge = {
                "name": name,
                "reason": (
                    f"maxabs={res['maxabs']:.6e}, "
                    f"inf_mismatch={res['inf_mismatch']}"
                ),
            }

    if first_diverge is None:
        print(f"first diverge: none (all <= {args.threshold:.3e})")
    else:
        print(f"first diverge: {first_diverge['name']} ({first_diverge['reason']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())