#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PRIORITY_TRACE_FILES = [
    "input_y_step0.h",
    "output_logits_step0.h",
    "output_x_pred_step0.h",
    "embed_plus_SPE_step0.h",
    "encoder_norm_mid_out_step0.h",
    "encoder_norm_end_out_step0.h",
    "norm_mid_out_step0.h",
    "layer0_attn_Q_step0.h",
    "layer0_attn_K_step0.h",
    "layer0_attn_V_step0.h",
    "layer0_attn_pre_concat_step0.h",
    "layer0_attn_post_concat_step0.h",
    "layer0_attn_out_step0.h",
    "layer0_norm_attn_in_step0.h",
    "layer0_norm_attn_out_step0.h",
    "layer0_ffn_relu_out_step0.h",
    "layer0_ffn_w2_out_step0.h",
    "layer1_norm_attn_out_step0.h",
    "layer1_ffn_w2_out_step0.h",
]


# (trace header file, algorithm_ref npy, cpp ref_model npy)
CHECKPOINT_MAP: List[Tuple[str, Optional[str], Optional[str]]] = [
    ("input_y_step0.h", None, None),
    ("embed_plus_SPE_step0.h", "preproc_x", "preproc_x"),
    ("layer0_attn_Q_step0.h", "layer0_q", "layer0_q"),
    ("layer0_attn_K_step0.h", "layer0_k", "layer0_k"),
    ("layer0_attn_V_step0.h", "layer0_v", "layer0_v"),
    ("layer0_attn_pre_concat_step0.h", "layer0_ctx", "layer0_ctx"),
    ("layer0_attn_post_concat_step0.h", "layer0_post_concat", "layer0_post_concat"),
    ("layer0_attn_out_step0.h", "layer0_attn_out", "layer0_attn_out"),
    ("layer0_norm_attn_in_step0.h", "layer0_ln_in", "layer0_ln_in"),
    ("layer0_norm_attn_out_step0.h", "layer0_ln_out", "layer0_ln_out"),
    ("layer0_ffn_relu_out_step0.h", "layer0_act_out", "layer0_act_out"),
    ("layer0_ffn_w2_out_step0.h", "layer0_ffn2_out", "layer0_ffn2_out"),
    ("encoder_norm_mid_out_step0.h", None, "layer0_mid_norm_dut_aligned"),
    ("norm_mid_out_step0.h", None, "layer0_mid_norm_dut_aligned"),
    ("layer1_norm_attn_out_step0.h", "layer1_ln_out", "layer1_ln_out"),
    ("layer1_ffn_w2_out_step0.h", "layer1_ffn2_out", "layer1_ffn2_out"),
    ("encoder_norm_end_out_step0.h", None, None),
    ("output_logits_step0.h", "final_logits", "final_logits"),
    ("output_x_pred_step0.h", "final_x_pred", "final_x_pred"),
]


def parse_header_tensor(header_path: Path) -> Tuple[np.ndarray, Tuple[int, ...]]:
    text = header_path.read_text(encoding="utf-8")
    shape_name_re = re.compile(r"constexpr\s+int\s+([A-Za-z0-9_]+)_tensor_shape\[\d+\]\s*=")
    m_name = shape_name_re.search(text)
    if m_name is None:
        raise RuntimeError(f"cannot find tensor shape symbol in {header_path}")
    prefix = m_name.group(1)

    shape_re = re.compile(
        rf"constexpr\s+int\s+{re.escape(prefix)}_tensor_shape\[\d+\]\s*=\s*\{{([^}}]+)\}};"
    )
    data_re = re.compile(
        rf"static\s+const\s+double\s+{re.escape(prefix)}_tensor\[\d+\]\s*=\s*\{{(.*?)\}};",
        re.S,
    )

    ms = shape_re.search(text)
    md = data_re.search(text)
    if ms is None or md is None:
        raise RuntimeError(f"cannot parse tensor from {header_path}")

    shape = tuple(int(v.strip()) for v in ms.group(1).split(",") if v.strip())
    data = np.fromstring(md.group(1).replace("\n", " "), sep=",", dtype=np.float64)
    return data.reshape(shape), shape


def squeeze_final(arr: np.ndarray) -> np.ndarray:
    out = arr
    while out.ndim > 1 and out.shape[0] == 1:
        out = out[0]
    return out


def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size > 0 else 0.0


def to_source_logits(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_final(arr)
    if arr.ndim != 1:
        raise RuntimeError(f"unexpected logits ndim: {arr.ndim}")
    return arr.astype(np.float64, copy=False)


def to_source_xpred(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_final(arr)
    if arr.ndim != 1:
        raise RuntimeError(f"unexpected x_pred ndim: {arr.ndim}")
    return (arr != 0.0).astype(np.int32)


def sign_scalar(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze sample-level trace/algorithm_ref/ref_model flips")
    parser.add_argument("--trace-dir", type=Path, default=Path("data/trace"))
    parser.add_argument("--py-root", type=Path, required=True, help="root dir containing pattern_<id>/*.npy")
    parser.add_argument("--cpp-root", type=Path, required=True, help="root dir containing pattern_<id>/*.npy")
    parser.add_argument("--samples", type=int, nargs="+", default=[23, 63])
    parser.add_argument("--tol", type=float, default=1.0e-4)
    args = parser.parse_args()

    trace_tensors: Dict[str, np.ndarray] = {}
    trace_shapes: Dict[str, Tuple[int, ...]] = {}
    present: List[str] = []
    missing: List[str] = []

    for name in PRIORITY_TRACE_FILES:
        p = args.trace_dir / name
        if p.exists():
            arr, shp = parse_header_tensor(p)
            trace_tensors[name] = arr
            trace_shapes[name] = shp
            present.append(name)
        else:
            missing.append(name)

    print("=== trace header inventory ===")
    print(f"trace_dir={args.trace_dir}")
    print(f"present_count={len(present)} missing_count={len(missing)}")
    for n in present:
        print(f"present {n} shape={trace_shapes[n]}")
    for n in missing:
        print(f"missing {n}")

    if "input_y_step0.h" not in trace_tensors or "output_logits_step0.h" not in trace_tensors or "output_x_pred_step0.h" not in trace_tensors:
        raise RuntimeError("required final I/O traces are missing")

    trace_input = trace_tensors["input_y_step0.h"]
    trace_logits = trace_tensors["output_logits_step0.h"]
    trace_xpred = trace_tensors["output_x_pred_step0.h"]
    batch = int(trace_input.shape[0])
    n_vars = int(trace_input.shape[1])

    for sid in args.samples:
        print(f"\n=== sample {sid} ===")
        if sid < 0 or sid >= batch:
            print(f"sample {sid} out_of_range valid=[0,{batch})")
            continue

        py_dir = args.py_root / f"pattern_{sid}"
        cpp_dir = args.cpp_root / f"pattern_{sid}"
        py_logits_np = load_npy_if_exists(py_dir / "final_logits.npy")
        py_xpred_np = load_npy_if_exists(py_dir / "final_x_pred.npy")
        cpp_logits_np = load_npy_if_exists(cpp_dir / "final_logits.npy")
        cpp_xpred_np = load_npy_if_exists(cpp_dir / "final_x_pred.npy")

        if py_logits_np is None or py_xpred_np is None:
            print(f"missing algorithm_ref final dumps in {py_dir}")
            continue
        if cpp_logits_np is None or cpp_xpred_np is None:
            print(f"missing cpp_ref_model final dumps in {cpp_dir}")
            continue

        tr_logits = trace_logits[sid].astype(np.float64)
        tr_x = (trace_xpred[sid] != 0.0).astype(np.int32)
        tr_y = trace_input[sid].astype(np.float64)

        py_logits = to_source_logits(py_logits_np)
        py_x = to_source_xpred(py_xpred_np)
        cpp_logits = to_source_logits(cpp_logits_np)
        cpp_x = to_source_xpred(cpp_xpred_np)

        if py_logits.shape[0] != n_vars or cpp_logits.shape[0] != n_vars:
            print(f"final logits shape mismatch n_vars={n_vars} py={py_logits.shape} cpp={cpp_logits.shape}")
            continue

        mismatch_bits = []
        for i in range(n_vars):
            if not (tr_x[i] == py_x[i] == cpp_x[i]):
                mismatch_bits.append(i)

        print(f"final_x_pred_mismatch_bits={mismatch_bits if mismatch_bits else 'none'}")
        print(
            f"final_logits_maxabs trace_vs_py={maxabs(tr_logits, py_logits):.9e} "
            f"trace_vs_cpp={maxabs(tr_logits, cpp_logits):.9e} py_vs_cpp={maxabs(py_logits, cpp_logits):.9e}"
        )

        if mismatch_bits:
            print("mismatch_bit_details idx src input_y logits x_pred margin")
            for idx in mismatch_bits:
                yv = float(tr_y[idx])
                s = sign_scalar(yv)
                tr_m = float(tr_logits[idx] * s)
                py_m = float(py_logits[idx] * s)
                cpp_m = float(cpp_logits[idx] * s)
                print(f"{idx} trace {yv:.9g} {tr_logits[idx]:.9g} {int(tr_x[idx])} {tr_m:.9g}")
                print(f"{idx} algorithm_ref {yv:.9g} {py_logits[idx]:.9g} {int(py_x[idx])} {py_m:.9g}")
                print(f"{idx} cpp_ref_model {yv:.9g} {cpp_logits[idx]:.9g} {int(cpp_x[idx])} {cpp_m:.9g}")

        earliest = None
        print("checkpoint_scan name status trace_vs_py trace_vs_cpp py_vs_cpp")
        for trace_name, py_name, cpp_name in CHECKPOINT_MAP:
            trace_arr = trace_tensors.get(trace_name)
            if trace_arr is None:
                print(f"{trace_name} missing_trace - - -")
                continue

            trace_sample = trace_arr[sid]
            py_arr = None
            cpp_arr = None
            if py_name is not None:
                py_arr = load_npy_if_exists(py_dir / f"{py_name}.npy")
            if cpp_name is not None:
                cpp_arr = load_npy_if_exists(cpp_dir / f"{cpp_name}.npy")

            if py_arr is None and py_name is not None:
                print(f"{trace_name} missing_algorithm_ref - - -")
                continue
            if cpp_arr is None and cpp_name is not None:
                print(f"{trace_name} missing_cpp_ref_model - - -")
                continue

            # input_y has no dump in py/cpp; treat as non-threeway stage.
            if py_name is None and cpp_name is None:
                print(f"{trace_name} no_threeway_contract 0.0 0.0 0.0")
                continue
            if py_name is None:
                print(f"{trace_name} missing_algorithm_ref - - -")
                continue
            if cpp_name is None:
                print(f"{trace_name} missing_cpp_ref_model - - -")
                continue

            py_sample = squeeze_final(py_arr)
            cpp_sample = squeeze_final(cpp_arr)

            # shape harmonization for final logits/xpred
            if trace_name in ("output_logits_step0.h", "output_x_pred_step0.h"):
                trace_sample = squeeze_final(trace_sample)

            d_tp = maxabs(trace_sample, py_sample)
            d_tc = maxabs(trace_sample, cpp_sample)
            d_pc = maxabs(py_sample, cpp_sample)
            diverged = (d_tp > args.tol) or (d_tc > args.tol) or (d_pc > args.tol)
            status = "diverged" if diverged else "exact_or_tol"
            print(f"{trace_name} {status} {d_tp:.9e} {d_tc:.9e} {d_pc:.9e}")
            if diverged and earliest is None:
                earliest = trace_name

        print(f"earliest_explanatory_checkpoint={earliest if earliest is not None else 'none'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
