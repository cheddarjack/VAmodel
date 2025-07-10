import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clear_dir(dir_path: str) -> None:
    """Delete every file & sub‑folder inside *dir_path* (keeps the folder)."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    for child in p.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Loss & post‑processing utilities
# ──────────────────────────────────────────────────────────────────────────────

# NOTE: the *exact* list of discrete classes comes from the YAML config and is
#       passed in at runtime.  Nothing hard‑coded here beyond convenience.


def build_lookup(allowed_values: List[int], device: torch.device) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Return (<tensor of allowed>, {value -> class‑index}) on *device*."""
    allowed = torch.tensor(allowed_values, dtype=torch.long, device=device)
    mapping = {int(v): i for i, v in enumerate(allowed_values)}
    return allowed, mapping


def build_class_weights(mapping, multiplier, num_classes, device):
    w = torch.ones(num_classes, device=device)
    for v, idx in mapping.items():
        w[idx] *= multiplier.get(v, 1.0)
    return w




def tensor_to_index(targets: torch.Tensor, mapping: Dict[int, int]) -> torch.LongTensor:
    """Map each integer value in *targets* → class index using *mapping*."""
    flat = targets.view(-1).to(torch.long)
    idx = torch.empty_like(flat)
    for v, i in mapping.items():
        idx[flat == v] = i
    return idx


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mapping: Dict[int, int],
    weight: torch.Tensor,
    mask: torch.Tensor = None,  # new parameter to optionally pass a mask
) -> torch.Tensor:
    """Cross‑entropy on (B, T, L, C) logits vs. (B, T, L) target values.
    Optionally ignores padded rows (where mask == 0).
    """
    B, T, L, C = logits.shape
    logits_2d = logits.view(-1, C)                    # (B*T*L, C)
    idx_targets = tensor_to_index(targets, mapping)   # (B*T*L)
    # Compute per-element loss without reduction.
    loss = F.cross_entropy(logits_2d, idx_targets, weight=weight, reduction="none")
    if mask is not None:
        # reshape loss to match (B, T, L)
        loss = loss.view(B, T, L)
        # if mask is (B, T), expand it to (B, T, L)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).expand(B, T, L)
        mask = mask.float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss


def snap_to_allowed(pred_vals: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Clamp / snap continuous predictions onto nearest allowed discrete value."""
    # pred_vals : (B, T, L)
    diff = (pred_vals.unsqueeze(-1) - allowed).abs()  # (..., C)
    nearest_idx = diff.argmin(dim=-1)
    return allowed[nearest_idx]


# ──────────────────────────────────────────────────────────────────────────────
#  Training / validation loops
# ──────────────────────────────────────────────────────────────────────────────


def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict,
    multiplier: Dict[int, float],
) -> None:
    """Main training entry‑point used by run_training.py."""
    debug_dir = config["data"].get("debug_dir", "./debug")
    clear_dir(debug_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---------------------------------------------
    # look‑up tables & class weights
    # ---------------------------------------------
    allowed_values = config["data"]["allowed_values"]          # e.g. [0,‑8…8]
    allowed, mapping = build_lookup(allowed_values, device)    # <tensor>, {val→idx}

    num_classes = len(allowed_values)
    class_w = build_class_weights(                             # tensor shape = (C,)
        mapping,
        multiplier,            # <-- the `multiplier_dict` you passed in
        num_classes,
        device,
    )

    # ─── optimiser & scheduler ────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=config["training"]["max_lr"], weight_decay=config["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=config["training"]["max_lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["training"]["epochs"],
    )

    best_val = float("inf")
    patience = config["training"].get("patience", 50)
    no_improve = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)
        for step, (X, y) in enumerate(pbar, 1):
            X, y = X.to(device).float(), y.to(device).long()
            mask = (X.sum(dim=-1) != 0).long()          # compute mask where row is not padded
            opt.zero_grad()
            logits, _ = model(X, mask)                  # logits : (B, T, L, C)
            loss = classification_loss(logits, y, mapping, class_w, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        
            running += loss.item()
            pbar.set_postfix(loss=f"{running/step:.4f}")

        val_loss = evaluate(model, val_loader, allowed, mapping, class_w, device)
        print(f"→ epoch {epoch} | train {running/step:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), Path(config["training"]["save_dir"])/"best_model.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early‑stop: no improvement for {patience} epochs.")
                break

    # final save
    torch.save(model.state_dict(), Path(config["training"]["checkpoint_dir"])/"last_model.pt")
    print(f"Training finished. Best val loss = {best_val:.4f}")


def evaluate(
    model: torch.nn.Module,
    loader,
    allowed: torch.Tensor,
    mapping: Dict[int, int],
    class_w: torch.Tensor,
    device: torch.device,
) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    total = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device).float(), y.to(device).long()
            mask = (X.sum(dim=-1) != 0).long()      # compute mask here as well
            logits, pred_vals = model(X, mask)
            total += classification_loss(logits, y, mapping, class_w, mask=mask).item()
    return total / len(loader)