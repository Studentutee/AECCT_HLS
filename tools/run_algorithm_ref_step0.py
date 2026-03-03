#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


def parse_trace_header_tensor(header_path: Path, tensor_name: str) -> np.ndarray:
    text = header_path.read_text(encoding="utf-8")
    shape_pat = re.compile(
        rf"constexpr\s+int\s+{re.escape(tensor_name)}_tensor_shape\[\d+\]\s*=\s*\{{([^}}]+)\}};"
    )
    arr_pat = re.compile(
        rf"static\s+const\s+double\s+{re.escape(tensor_name)}_tensor\[\d+\]\s*=\s*\{{(.*?)\}};",
        re.S,
    )

    m_shape = shape_pat.search(text)
    m_arr = arr_pat.search(text)
    if m_shape is None or m_arr is None:
        raise RuntimeError(f"failed parsing {tensor_name} in {header_path}")

    shape = tuple(int(x.strip()) for x in m_shape.group(1).split(",") if x.strip())
    values = np.fromstring(m_arr.group(1).replace("\n", " "), sep=",", dtype=np.float64)
    return values.reshape(shape)


def validate_notebook(nb_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
    required = [
        "manual_layer_norm",
        "compute_linear_int8",
        "one_ring_mask",
        "second_ring_mask",
        "decoder.norm2",
        "decoder.norm.weight",
    ]
    missing = [k for k in required if k not in src]
    if missing:
        raise RuntimeError(f"algorithm_ref notebook missing keys: {missing}")


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + eps) * w + b


def quant_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, s_x: float, s_w: torch.Tensor) -> torch.Tensor:
    qx = torch.round(x * s_x)
    return qx @ (w.T / (s_x * s_w)) + b


def build_masks(src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vv = src_mask[:63, :63]
    vc = src_mask[:63, 63:]
    cv = src_mask[63:, :63]
    cc = src_mask[63:, 63:]

    def tt(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x, dtype=torch.bool)

    one_ring = torch.cat(
        [torch.cat([tt(vv), vc], dim=1), torch.cat([cv, tt(cc)], dim=1)],
        dim=0,
    )
    second_ring = torch.cat(
        [torch.cat([vv, tt(vc)], dim=1), torch.cat([tt(cv), cc], dim=1)],
        dim=0,
    )
    return one_ring, second_ring


def dump_np(dump_dir: Path, name: str, t: torch.Tensor) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    np.save(dump_dir / f"{name}.npy", t.detach().cpu().numpy())


def metrics(ref: np.ndarray, dut: np.ndarray) -> tuple[float, float, float, float]:
    err = dut - ref
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    return mse, rmse, mae, max_abs


def run_layer(
    layer_idx: int,
    x_in: torch.Tensor,
    state: dict,
    one_ring_mask: torch.Tensor,
    second_ring_mask: torch.Tensor,
    s_x_in: float,
    s_x_o: float,
    s_x_ff1: float,
    s_x_ff2: float,
) -> tuple[torch.Tensor, dict]:
    p = f"decoder.layers.{layer_idx}."

    q = quant_linear(
        x_in,
        state[p + "self_attn.linears.0.weight"],
        state[p + "self_attn.linears.0.bias"],
        s_x_in,
        state[p + "self_attn.linears.0.s_w"],
    )
    k = quant_linear(
        x_in,
        state[p + "self_attn.linears.1.weight"],
        state[p + "self_attn.linears.1.bias"],
        s_x_in,
        state[p + "self_attn.linears.1.s_w"],
    )
    v = quant_linear(
        x_in,
        state[p + "self_attn.linears.2.weight"],
        state[p + "self_attn.linears.2.bias"],
        s_x_in,
        state[p + "self_attn.linears.2.s_w"],
    )

    qh = q.reshape(75, 8, 4).transpose(0, 1)
    kh = k.reshape(75, 8, 4).transpose(0, 1)
    vh = v.reshape(75, 8, 4).transpose(0, 1)

    scores_list = []
    for h in range(8):
        s = (qh[h] @ kh[h].T) / math.sqrt(4.0)
        mask = one_ring_mask if h < 4 else second_ring_mask
        s = s.masked_fill(mask, -float("inf"))
        scores_list.append(s)
    scores = torch.stack(scores_list)

    probs = torch.softmax(scores, dim=2)
    ctx = torch.stack([probs[h] @ vh[h] for h in range(8)])
    post_concat = ctx.transpose(0, 1).reshape(75, 32)

    attn_out = quant_linear(
        post_concat,
        state[p + "self_attn.linears.3.weight"],
        state[p + "self_attn.linears.3.bias"],
        s_x_o,
        state[p + "self_attn.linears.3.s_w"],
    )

    ln_in = attn_out + x_in
    ln_out = layer_norm(ln_in, state[p + "sublayer.0.norm.weight"], state[p + "sublayer.0.norm.bias"])

    ffn1_out = quant_linear(
        ln_out,
        state[p + "feed_forward.w_1.weight"],
        state[p + "feed_forward.w_1.bias"],
        s_x_ff1,
        state[p + "feed_forward.w_1.s_w"],
    )
    act_out = torch.relu(ffn1_out)
    ffn2_out = quant_linear(
        act_out,
        state[p + "feed_forward.w_2.weight"],
        state[p + "feed_forward.w_2.bias"],
        s_x_ff2,
        state[p + "feed_forward.w_2.s_w"],
    )

    ffn_ln_in = ffn2_out + ln_out
    ffn_ln_out = layer_norm(ffn_ln_in, state[p + "sublayer.1.norm.weight"], state[p + "sublayer.1.norm.bias"])

    ckpt = {
        f"layer{layer_idx}_ln_in": ln_in,
        f"layer{layer_idx}_q": q,
        f"layer{layer_idx}_k": k,
        f"layer{layer_idx}_v": v,
        f"layer{layer_idx}_attn_scores": scores,
        f"layer{layer_idx}_attn_probs": probs,
        f"layer{layer_idx}_ctx": ctx,
        f"layer{layer_idx}_attn_out": attn_out,
        f"layer{layer_idx}_ln_out": ln_out,
        f"layer{layer_idx}_ffn1_out": ffn1_out,
        f"layer{layer_idx}_act_out": act_out,
        f"layer{layer_idx}_ffn2_out": ffn2_out,
        f"layer{layer_idx}_ffn_ln_out": ffn_ln_out,
    }
    return ffn_ln_out, ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run algorithm_ref step0 and dump checkpoints")
    parser.add_argument("--pattern", type=int, default=0)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument(
        "--pth",
        type=Path,
        default=Path(r"C:\Users\Peter\Code Data\AECCT\saved_models\BCH_n63_k51__Ndec2_d32_h8\stage2_infer_frozen__BCH_n63_k51__Ndec2_d32_h8__e1941_loss0.017464__with_lpe_token.pth"),
    )
    parser.add_argument(
        "--nb",
        type=Path,
        default=Path(r"C:\Users\Peter\Desktop\大學課程資料\碩士班題目\AECCT-Hardware-Design-Toolkit\algorithm_ref.ipynb"),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(r"C:\Users\Peter\Code Data\AECCT\trace_infer\BCH_n63_k51__Ndec2_d32_h8"),
    )
    parser.add_argument(
        "--h-path",
        type=Path,
        default=Path(r"C:\Users\Peter\Code Data\AECCT\codes\BCH_N63_K51.txt"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    header_input = repo_root / "data" / "trace" / "input_y_step0.h"
    header_logits = repo_root / "data" / "trace" / "output_logits_step0.h"
    header_xpred = repo_root / "data" / "trace" / "output_x_pred_step0.h"

    validate_notebook(args.nb)

    state = torch.load(args.pth, map_location="cpu")
    for k, v in list(state.items()):
        if torch.is_tensor(v):
            state[k] = v.float()

    input_y_pt = torch.load(args.trace_dir / "input_y_step0.pt", map_location="cpu").float()
    logits_pt = torch.load(args.trace_dir / "output_logits_step0.pt", map_location="cpu").float()
    xpred_pt = torch.load(args.trace_dir / "output_x_pred_step0.pt", map_location="cpu").float()

    input_y_h = parse_trace_header_tensor(header_input, "trace_input_y_step0")
    logits_h = parse_trace_header_tensor(header_logits, "trace_output_logits_step0")
    xpred_h = parse_trace_header_tensor(header_xpred, "trace_output_x_pred_step0")

    max_in = np.max(np.abs(input_y_pt.numpy().astype(np.float64) - input_y_h))
    max_log = np.max(np.abs(logits_pt.numpy().astype(np.float64) - logits_h))
    max_xp = np.max(np.abs(xpred_pt.numpy().astype(np.float64) - xpred_h))
    print(f"header consistency: input={max_in:.3e} logits={max_log:.3e} xpred={max_xp:.3e}")

    y = input_y_pt[args.pattern]

    h = torch.from_numpy(np.loadtxt(args.h_path)).long()
    y_hard = (y < 0.0).long()
    syndrome = (h @ y_hard) % 2
    syndrome_pm1 = (1 - 2 * syndrome).float()
    node_feature = torch.cat([y.abs(), syndrome_pm1], dim=0)

    preproc_x = torch.cat(
        [
            node_feature.unsqueeze(1) * state["src_embed"],
            state["lpe_token"],
        ],
        dim=1,
    )

    src_mask = state["src_mask"][0, 0].bool()
    one_ring_mask, second_ring_mask = build_masks(src_mask)

    l0_in_s_x = float(torch.load(args.trace_dir / "layer0_attn_Q_s_x_step0.pt", map_location="cpu"))
    l0_o_s_x = float(torch.load(args.trace_dir / "layer0_attn_Wo_s_x_step0.pt", map_location="cpu"))
    l0_ff1_s_x = float(torch.load(args.trace_dir / "layer0_ffn_w1_s_x_step0.pt", map_location="cpu"))
    l0_ff2_s_x = float(torch.load(args.trace_dir / "layer0_ffn_w2_s_x_step0.pt", map_location="cpu"))

    l1_in_s_x = float(torch.load(args.trace_dir / "layer1_attn_Q_s_x_step0.pt", map_location="cpu"))
    l1_o_s_x = float(torch.load(args.trace_dir / "layer1_attn_Wo_s_x_step0.pt", map_location="cpu"))
    l1_ff1_s_x = float(torch.load(args.trace_dir / "layer1_ffn_w1_s_x_step0.pt", map_location="cpu"))
    l1_ff2_s_x = float(torch.load(args.trace_dir / "layer1_ffn_w2_s_x_step0.pt", map_location="cpu"))

    l0_out, ck0 = run_layer(
        0,
        preproc_x,
        state,
        one_ring_mask,
        second_ring_mask,
        l0_in_s_x,
        l0_o_s_x,
        l0_ff1_s_x,
        l0_ff2_s_x,
    )
    mid = layer_norm(l0_out, state["decoder.norm2.weight"], state["decoder.norm2.bias"])

    l1_out, ck1 = run_layer(
        1,
        mid,
        state,
        one_ring_mask,
        second_ring_mask,
        l1_in_s_x,
        l1_o_s_x,
        l1_ff1_s_x,
        l1_ff2_s_x,
    )

    end = layer_norm(l1_out, state["decoder.norm.weight"], state["decoder.norm.bias"])
    final_node_logits = end @ state["oned_final_embed.0.weight"].T + state["oned_final_embed.0.bias"]
    final_out_fc_in = final_node_logits.squeeze(-1).unsqueeze(0)
    final_logits = final_out_fc_in @ state["out_fc.weight"].T + state["out_fc.bias"]
    final_x_pred = (final_logits[0] * torch.sign(y) < 0).float()

    dump_np(args.dump_dir, "preproc_x", preproc_x)
    for k, v in ck0.items():
        dump_np(args.dump_dir, k, v)
    for k, v in ck1.items():
        dump_np(args.dump_dir, k, v)
    dump_np(args.dump_dir, "final_node_logits", final_node_logits)
    dump_np(args.dump_dir, "final_out_fc_in", final_out_fc_in)
    dump_np(args.dump_dir, "final_logits", final_logits)
    dump_np(args.dump_dir, "final_x_pred", final_x_pred)

    golden_logits = logits_h[args.pattern]
    golden_xpred = xpred_h[args.pattern]
    ref_logits = final_logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
    ref_xpred = final_x_pred.detach().cpu().numpy().astype(np.float64)

    mse, rmse, mae, max_abs = metrics(golden_logits, ref_logits)
    xmatch = float(np.mean((golden_xpred != 0.0) == (ref_xpred != 0.0)) * 100.0)

    print("=== Python step0 vs golden(header) ===")
    print(f"MSE    : {mse:.6e}")
    print(f"RMSE   : {rmse:.6e}")
    print(f"MAE    : {mae:.6e}")
    print(f"MaxAbs : {max_abs:.6e}")
    print(f"x_pred match: {xmatch:.2f}%")
    print("golden_logits[0:8]:", np.array2string(golden_logits[:8], precision=6, separator=", "))
    print("ref_logits[0:8]   :", np.array2string(ref_logits[:8], precision=6, separator=", "))
    print(f"dumped checkpoints: {args.dump_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())