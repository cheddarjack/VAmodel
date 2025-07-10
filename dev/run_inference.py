#!/usr/bin/env python3
"""Robust, end‑to‑end inference for VAmodel.

💡 **Two‑dimensional labels**
Your model predicts **2 label columns**, each with **15 classes** (0, ±2…±8).
The script now reads those numbers from *config.yaml* and rebuilds the TCN with
``output_size = 15`` and ``num_labels = 2`` – exactly like training – so the
checkpoint loads cleanly and predictions come out as two columns.

Other features remain:
* sliding‑window inference (stride = 2048 by default)
* automatic missing‑feature repair and per‑block min‑max scaling
* optional PNG/HTML visualisation via ``preprocess.visualize``
"""
from __future__ import annotations
import os, glob, argparse, time, gc
import yaml, numpy as np, pandas as pd, torch

from src.model import TCN, Transformer, Longformer

# ───────────────────────── helper functions ──────────────────────────
STRIDE = None  # None → non‑overlap; set e.g. 512 for 75 % overlap


def ensure_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        if col not in df.columns:
            df[col] = (
                np.arange(len(df), dtype=np.int32) if col == "unique_id" else 0.0
            )
    return df[features]


def scale_block(block: np.ndarray) -> np.ndarray:
    for j in range(block.shape[1]):
        v = block[:, j]
        msk = v != 0
        if msk.any():
            lo, hi = v[msk].min(), v[msk].max()
            rng = hi - lo or 1.0
            block[:, j] = (v - lo) / rng
    return block


def pad_block(arr: np.ndarray, L: int) -> np.ndarray:
    cur = arr.shape[0]
    if cur == L:
        return arr
    if cur < L:
        pad = np.zeros((L - cur, arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)
    return arr[-L:]


# ────────────────────── model construction helpers ───────────────────

def build_model(cfg: dict) -> torch.nn.Module:
    allowed_vals = cfg["data"]["allowed_values"]
    num_values   = len(allowed_vals)          # 15 classes

    label_cols   = cfg["data"]["label_column"]
    if isinstance(label_cols, str):
        label_cols = [label_cols]
    num_labels   = len(label_cols)            # 2 label columns

    output_size  = cfg["model"]["output_size"]  # 15 (classes per column)

    common = dict(
        d_model = cfg["model"]["d_model"],
        dropout = cfg["model"].get("dropout", 0.0),
        num_features = len(cfg["data"]["features"]),
        output_size  = output_size,
        num_labels   = num_labels,
    )

    t = cfg["model"]["type"].lower()
    if t == "tcn":
        return TCN(
            **common,
            num_levels       = cfg["model"]["tcn_num_levels"],
            kernel_size      = cfg["model"]["tcn_kernel_size"],
            tcn_padding      = cfg["model"]["tcn_padding"],
            tcn_channel_growth = cfg["model"]["tcn_channel_growth"],
            max_dilation     = cfg["model"].get("max_dilations", 128),
            num_values       = num_values,
        )
    if t == "transformer":
        return Transformer(
            **common,
            nhead             = cfg["model"]["nhead"],
            num_hidden_layers = cfg["model"]["num_encoder_layers"],
            dim_feedforward   = cfg["model"]["dim_feedforward"],
            max_seq_len       = cfg["data"]["seq_length"],
        )
    if t == "longformer":
        return Longformer(
            **common,
            num_hidden_layers   = cfg["model"]["num_encoder_layers"],
            num_attention_heads = cfg["model"]["nhead"],
            dim_feedforward     = cfg["model"]["dim_feedforward"],
            attention_window    = cfg["model"].get("attention_window", 2048),
        )
    raise ValueError(f"Unknown model type {t}")


# ──────────────────────────── inference ─────────────────────────────

def infer_file(
    df: pd.DataFrame,
    model,
    features: list[str],
    device,
    seq_len: int,
) -> np.ndarray:
    X = ensure_features(df.copy(), features).values.astype(np.float32)
    N, C_in = X.shape
    _, num_labels, num_values = 1, model.num_labels, model.num_values  # type: ignore
    preds_full = np.empty((N, num_labels), dtype=np.float32)

    step = STRIDE or seq_len
    model.eval()
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + seq_len, N)
            block = X[start:end]
            blk_pad = pad_block(block, seq_len)
            blk_pad = scale_block(blk_pad)
            xt = torch.from_numpy(blk_pad[None, ...]).to(device)
            _, p = model(xt)                  # p shape (1, L, 2)
            p = p.squeeze(0).cpu().numpy()    # shape (L, 2)
            # map class‑index → actual allowed value
            allowed = model.allowed.cpu().numpy() if hasattr(model, "allowed") else np.arange(p.shape[-1])
            p_vals = allowed[p.astype(int)]        # shape (L, 2)
            usable = p_vals[-(end - start) :]
            preds_full[start:end] = usable
    return preds_full


# ───────────────────────────── main() ───────────────────────────────

def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)

    ckpt_p = os.path.join(cfg["training"].get("save_dir", "./"), "best_model.pt")
    model.load_state_dict(torch.load(ckpt_p, map_location=device), strict=True)
    print(f"Loaded weights → {ckpt_p}")

    inf_dir = cfg["data"]["inference_data_dir"]
    out_dir = cfg["data"]["inference_results_dir"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(inf_dir, "**", "*.parquet"), recursive=True))
    print(f"Found {len(files)} parquet file(s) for inference.")

    seq_len    = cfg["data"]["seq_length"]
    features   = cfg["data"]["features"]
    label_cols = cfg["data"]["label_column"]
    if isinstance(label_cols, str):
        label_cols = [label_cols]

    for fp in files:
        df    = pd.read_parquet(fp)
        preds = infer_file(df, model, features, device, seq_len)

        for i, lab in enumerate(label_cols):
            df[f"pred_{lab}"] = preds[:, i]

        out_fp = os.path.join(out_dir, os.path.basename(fp).replace(".parquet", "_results.parquet"))
        df.to_parquet(out_fp, index=False)
        print(f"✓ {os.path.basename(fp):<28} → {out_fp}")
        del df, preds
        gc.collect(); torch.cuda.empty_cache()

    # ─── visuals ---------------------------------------------------
    VISUAL_DIR = os.path.join(out_dir, "visuals")
    os.makedirs(VISUAL_DIR, exist_ok=True)
    try:
        import preprocess.visualize as viz
        viz.create_visuals(inf_dir, out_dir, VISUAL_DIR)
        print(f"Visuals saved to {VISUAL_DIR}")
    except ImportError:
        print("[WARN] preprocess.visualize not found – skipping visualisation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/home/cheddarjackk/Developer/VAmodel/dev/config.yaml")
    args = ap.parse_args()
    t0 = time.time()
    main(args.cfg)
    print(f"Total inference time {time.time() - t0:.1f}s")
