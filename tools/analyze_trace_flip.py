#!/usr/bin/env python3
import argparse
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

REQUIRED_ATTN_TRACE_HEADERS = [
    "layer0_attn_Q_step0.h",
    "layer0_attn_K_step0.h",
    "layer0_attn_V_step0.h",
    "layer0_attn_scores_pre_softmax_step0.h",
    "layer0_attn_probs_step0.h",
    "layer0_attn_pre_concat_step0.h",
]

# (display_name, trace_header_name_or_none, algorithm_ref_npy, cpp_ref_npy)
ATTN_CHECKPOINTS: List[Tuple[str, Optional[str], str, str]] = [
    ("layer0_attn_Q_step0", "layer0_attn_Q_step0.h", "layer0_q", "layer0_q"),
    ("layer0_attn_K_step0", "layer0_attn_K_step0.h", "layer0_k", "layer0_k"),
    ("layer0_attn_V_step0", "layer0_attn_V_step0.h", "layer0_v", "layer0_v"),
    ("layer0_attn_scores_pre_softmax", "layer0_attn_scores_pre_softmax_step0.h", "layer0_attn_scores", "layer0_attn_scores"),
    ("layer0_attn_probs", "layer0_attn_probs_step0.h", "layer0_attn_probs", "layer0_attn_probs"),
    ("layer0_attn_pre_concat_step0", "layer0_attn_pre_concat_step0.h", "layer0_ctx", "layer0_ctx"),
]


def parse_header_tensor(header_path: Path) -> np.ndarray:
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
    return data.reshape(shape)


def squeeze_sample(arr: np.ndarray) -> np.ndarray:
    out = arr
    while out.ndim > 1 and out.shape[0] == 1:
        out = out[0]
    return out


def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    both_nan = np.isnan(a) & np.isnan(b)
    both_same_inf = np.isinf(a) & np.isinf(b) & (np.signbit(a) == np.signbit(b))
    keep = ~(both_nan | both_same_inf)
    if not np.any(keep):
        return 0.0
    a_keep = a[keep]
    b_keep = b[keep]
    if (not np.all(np.isfinite(a_keep))) or (not np.all(np.isfinite(b_keep))):
        return float("inf")
    return float(np.max(np.abs(a_keep - b_keep)))


def load_npy(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path)


def sign_scalar(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def resolve_baseline_root(args: argparse.Namespace) -> Path:
    if args.cpp_root_baseline is not None:
        return args.cpp_root_baseline
    if args.cpp_root is not None:
        return args.cpp_root
    raise RuntimeError("need --cpp-root-baseline (or legacy --cpp-root)")


def report_final_case(case_name: str,
                      tr_y: np.ndarray,
                      tr_logits: np.ndarray,
                      tr_x: np.ndarray,
                      py_logits: np.ndarray,
                      py_x: np.ndarray,
                      case_logits: np.ndarray,
                      case_x: np.ndarray,
                      tol: float) -> Tuple[List[int], bool, bool]:
    mismatch_bits: List[int] = []
    for i in range(tr_x.shape[0]):
        if not (int(tr_x[i]) == int(py_x[i]) == int(case_x[i])):
            mismatch_bits.append(i)

    d_tr = maxabs(tr_logits, case_logits)
    d_py = maxabs(py_logits, case_logits)
    x_ok = bool(np.array_equal(case_x, tr_x) and np.array_equal(case_x, py_x))
    logits_ok = bool(d_tr <= tol and d_py <= tol)

    print(
        f"ablation_case={case_name} final_logits_maxabs trace_vs_case={d_tr:.9e} "
        f"py_vs_case={d_py:.9e} x_pred_all_match={x_ok} logits_all_match_tol={logits_ok}"
    )
    print(f"ablation_case={case_name} final_x_pred_mismatch_bits={mismatch_bits if mismatch_bits else 'none'}")
    if mismatch_bits:
        print(f"ablation_case={case_name} mismatch_bit_details idx src input_y logits x_pred margin")
        for idx in mismatch_bits:
            yv = float(tr_y[idx])
            s = sign_scalar(yv)
            tr_m = float(tr_logits[idx] * s)
            py_m = float(py_logits[idx] * s)
            c_m = float(case_logits[idx] * s)
            print(f"{idx} trace {yv:.9g} {tr_logits[idx]:.9g} {int(tr_x[idx])} {tr_m:.9g}")
            print(f"{idx} algorithm_ref {yv:.9g} {py_logits[idx]:.9g} {int(py_x[idx])} {py_m:.9g}")
            print(f"{idx} {case_name} {yv:.9g} {case_logits[idx]:.9g} {int(case_x[idx])} {c_m:.9g}")
    return mismatch_bits, x_ok, logits_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace / algorithm_ref / C++ ref_model flip and ablation analyzer")
    parser.add_argument("--trace-dir", type=Path, default=Path("data/trace"))
    parser.add_argument("--py-root", type=Path, required=True, help="dir containing pattern_<id>/*.npy")
    parser.add_argument("--cpp-root", type=Path, default=None, help="legacy: baseline C++ root")
    parser.add_argument("--cpp-root-baseline", type=Path, default=None)
    parser.add_argument("--cpp-root-softmax-exact", type=Path, default=None)
    parser.add_argument("--cpp-root-ln-exact", type=Path, default=None)
    parser.add_argument("--samples", type=int, nargs="+", default=[23, 63])
    parser.add_argument("--tol", type=float, default=1.0e-4)
    args = parser.parse_args()

    baseline_root = resolve_baseline_root(args)

    trace_headers: Dict[str, np.ndarray] = {}
    all_trace_files = sorted(p.name for p in args.trace_dir.glob("*.h"))
    for name in all_trace_files:
        trace_headers[name] = parse_header_tensor(args.trace_dir / name)

    print("=== trace_header_inventory ===")
    print(f"trace_dir={args.trace_dir}")
    print(f"all_trace_header_count={len(all_trace_files)}")
    for name in all_trace_files:
        print(f"trace_header {name}")

    print("=== priority_trace_headers ===")
    for name in PRIORITY_TRACE_FILES:
        status = "present" if name in trace_headers else "missing"
        print(f"{status} {name}")

    print("=== required_attn_trace_headers ===")
    for name in REQUIRED_ATTN_TRACE_HEADERS:
        status = "present" if name in trace_headers else "missing"
        print(f"{status} {name}")

    if "input_y_step0.h" not in trace_headers or "output_logits_step0.h" not in trace_headers or "output_x_pred_step0.h" not in trace_headers:
        raise RuntimeError("required final trace headers are missing in data/trace")

    trace_input = trace_headers["input_y_step0.h"]
    trace_logits = trace_headers["output_logits_step0.h"]
    trace_xpred = trace_headers["output_x_pred_step0.h"]
    batch = int(trace_input.shape[0])

    for sid in args.samples:
        print(f"\n=== sample {sid} ===")
        if sid < 0 or sid >= batch:
            print(f"sample={sid} out_of_range valid=[0,{batch})")
            continue

        py_dir = args.py_root / f"pattern_{sid}"
        cpp_base_dir = baseline_root / f"pattern_{sid}"

        py_logits_np = load_npy(py_dir / "final_logits.npy")
        py_xpred_np = load_npy(py_dir / "final_x_pred.npy")
        cpp_base_logits_np = load_npy(cpp_base_dir / "final_logits.npy")
        cpp_base_xpred_np = load_npy(cpp_base_dir / "final_x_pred.npy")
        if py_logits_np is None or py_xpred_np is None:
            print(f"missing algorithm_ref final dumps: {py_dir}")
            continue
        if cpp_base_logits_np is None or cpp_base_xpred_np is None:
            print(f"missing cpp_baseline final dumps: {cpp_base_dir}")
            continue

        tr_y = trace_input[sid].astype(np.float64)
        tr_logits = squeeze_sample(trace_logits[sid]).astype(np.float64)
        tr_x = (squeeze_sample(trace_xpred[sid]) != 0.0).astype(np.int32)
        py_logits = squeeze_sample(py_logits_np).astype(np.float64)
        py_x = (squeeze_sample(py_xpred_np) != 0.0).astype(np.int32)
        cpp_base_logits = squeeze_sample(cpp_base_logits_np).astype(np.float64)
        cpp_base_x = (squeeze_sample(cpp_base_xpred_np) != 0.0).astype(np.int32)

        print("pure_compare: final output consistency")
        print(
            f"final_logits_maxabs trace_vs_py={maxabs(tr_logits, py_logits):.9e} "
            f"trace_vs_cpp_baseline={maxabs(tr_logits, cpp_base_logits):.9e} "
            f"py_vs_cpp_baseline={maxabs(py_logits, cpp_base_logits):.9e}"
        )
        report_final_case(
            "cpp_baseline",
            tr_y,
            tr_logits,
            tr_x,
            py_logits,
            py_x,
            cpp_base_logits,
            cpp_base_x,
            args.tol,
        )

        print("pure_compare: attention_fine_checkpoints (sample-only)")
        earliest_baseline: Optional[str] = None
        for display_name, trace_name, py_name, cpp_name in ATTN_CHECKPOINTS:
            py_arr = load_npy(py_dir / f"{py_name}.npy")
            cpp_arr = load_npy(cpp_base_dir / f"{cpp_name}.npy")
            if py_arr is None:
                print(f"{display_name} missing_algorithm_ref")
                continue
            if cpp_arr is None:
                print(f"{display_name} missing_cpp_baseline")
                continue

            py_sample = squeeze_sample(py_arr)
            cpp_sample = squeeze_sample(cpp_arr)
            d_pc = maxabs(py_sample, cpp_sample)

            if trace_name is None:
                d_tp = float("nan")
                d_tc = float("nan")
                trace_status = "no_trace_contract"
                diverged = d_pc > args.tol
            elif trace_name not in trace_headers:
                d_tp = float("nan")
                d_tc = float("nan")
                trace_status = "missing_trace_header"
                diverged = d_pc > args.tol
            else:
                tr_sample = squeeze_sample(trace_headers[trace_name][sid])
                d_tp = maxabs(tr_sample, py_sample)
                d_tc = maxabs(tr_sample, cpp_sample)
                trace_status = "trace_present"
                diverged = (d_tp > args.tol) or (d_tc > args.tol) or (d_pc > args.tol)

            if earliest_baseline is None and diverged:
                earliest_baseline = display_name

            if np.isnan(d_tp):
                print(
                    f"{display_name} status={trace_status} "
                    f"py_vs_cpp_baseline={d_pc:.9e}"
                )
            else:
                print(
                    f"{display_name} status={trace_status} "
                    f"trace_vs_py={d_tp:.9e} trace_vs_cpp_baseline={d_tc:.9e} py_vs_cpp_baseline={d_pc:.9e}"
                )
        print(f"pure_compare_earliest_attn_divergence_cpp_baseline={earliest_baseline if earliest_baseline else 'none'}")

        softmax_x_ok = False
        softmax_logits_ok = False
        ln_x_ok = False
        ln_logits_ok = False
        softmax_earliest: Optional[str] = None
        ln_earliest: Optional[str] = None

        if args.cpp_root_softmax_exact is not None:
            case_dir = args.cpp_root_softmax_exact / f"pattern_{sid}"
            c_logits_np = load_npy(case_dir / "final_logits.npy")
            c_x_np = load_npy(case_dir / "final_x_pred.npy")
            if c_logits_np is None or c_x_np is None:
                print(f"ablation_case=softmax_exact missing_final_dumps dir={case_dir}")
            else:
                c_logits = squeeze_sample(c_logits_np).astype(np.float64)
                c_x = (squeeze_sample(c_x_np) != 0.0).astype(np.int32)
                _, softmax_x_ok, softmax_logits_ok = report_final_case(
                    "softmax_exact",
                    tr_y,
                    tr_logits,
                    tr_x,
                    py_logits,
                    py_x,
                    c_logits,
                    c_x,
                    args.tol,
                )
                for display_name, _trace_name, py_name, cpp_name in ATTN_CHECKPOINTS:
                    py_arr = load_npy(py_dir / f"{py_name}.npy")
                    c_arr = load_npy(case_dir / f"{cpp_name}.npy")
                    if py_arr is None or c_arr is None:
                        continue
                    d = maxabs(squeeze_sample(py_arr), squeeze_sample(c_arr))
                    if softmax_earliest is None and d > args.tol:
                        softmax_earliest = display_name
                print(
                    f"ablation_case=softmax_exact earliest_attn_divergence_vs_algorithm_ref="
                    f"{softmax_earliest if softmax_earliest else 'none'}"
                )

        if args.cpp_root_ln_exact is not None:
            case_dir = args.cpp_root_ln_exact / f"pattern_{sid}"
            c_logits_np = load_npy(case_dir / "final_logits.npy")
            c_x_np = load_npy(case_dir / "final_x_pred.npy")
            if c_logits_np is None or c_x_np is None:
                print(f"ablation_case=ln_exact missing_final_dumps dir={case_dir}")
            else:
                c_logits = squeeze_sample(c_logits_np).astype(np.float64)
                c_x = (squeeze_sample(c_x_np) != 0.0).astype(np.int32)
                _, ln_x_ok, ln_logits_ok = report_final_case(
                    "ln_exact",
                    tr_y,
                    tr_logits,
                    tr_x,
                    py_logits,
                    py_x,
                    c_logits,
                    c_x,
                    args.tol,
                )
                for display_name, _trace_name, py_name, cpp_name in ATTN_CHECKPOINTS:
                    py_arr = load_npy(py_dir / f"{py_name}.npy")
                    c_arr = load_npy(case_dir / f"{cpp_name}.npy")
                    if py_arr is None or c_arr is None:
                        continue
                    d = maxabs(squeeze_sample(py_arr), squeeze_sample(c_arr))
                    if ln_earliest is None and d > args.tol:
                        ln_earliest = display_name
                print(
                    f"ablation_case=ln_exact earliest_attn_divergence_vs_algorithm_ref="
                    f"{ln_earliest if ln_earliest else 'none'}"
                )

        if softmax_x_ok and softmax_logits_ok:
            decision = "softmax_only_replacement_is_sufficient"
        elif ln_x_ok and ln_logits_ok:
            decision = "softmax_only_not_sufficient_ln_only_is_sufficient"
        else:
            decision = "neither_softmax_only_nor_ln_only_restores_final_match"
        print(f"ablation_decision={decision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
