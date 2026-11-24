"""Expert Iteration (ExIt) trainer for QGRU, BeliefNet, and FastPolicy, generalized from
Belote's train_exit.py.

Given a dataset of search-labeled states collected by the game's C++ hybrid search (where
every legal action has a low-variance CRN search value `q_targets` across shared worlds):
  - Fits QGRU with combined multi-action Q-value MSE + soft-policy KL loss over all legal
    actions (eliminating single-trajectory Monte-Carlo variance).
  - Optionally fits BeliefNet on the search-played trajectories using the exact world
    sampler's marginals.
  - Optionally fits FastPolicy directly on the search soft-policy targets (or distill
    afterward via cardbot.train.distill).

Every game-specific piece (collection, symmetry augmentation, per-epoch evaluation) is a
hook passed by the game.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cardbot.train import dmc


@dataclass
class ExitConfig:
    epochs: int = 8
    batch_size: int = 512
    lr: float = 2e-4
    min_lr: float = 1e-5
    belief_lr: float = 5e-4
    belief_min_lr: float = 5e-5
    fast_lr: float = 1.5e-3
    weight_decay: float = 1e-4
    kl_weight: float = 0.05          # weight on soft-policy cross-entropy alongside Q MSE
    kl_temp: float = 0.08            # temperature for softmax(q_targets / kl_temp)
    grad_clip: float = 2.0
    amp: bool = True
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print


def _to_bf16_cpu(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
    return t.to(torch.bfloat16) if t.is_floating_point() else t


def concat_chunks(chunks: list[dict], bf16_keys: tuple[str, ...] = ("state", "history", "legal_actions")) -> dict:
    """Concatenate C++ collection chunks on CPU, keeping large float tensors in BF16 to save RAM."""
    out = {}
    for k in chunks[0]:
        parts = []
        for c in chunks:
            t = c[k] if isinstance(c[k], torch.Tensor) else torch.from_numpy(c[k])
            if k in bf16_keys and t.is_floating_point():
                t = t.to(torch.bfloat16)
            parts.append(t)
        out[k] = torch.cat(parts, dim=0)
    return out


def train(
    dataset: dict,
    q_model,
    cfg: ExitConfig,
    belief_model=None,
    fast_model=None,
    augment: Callable[[dict], dict] | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train QGRU (and optionally BeliefNet and FastPolicy) on a search-labeled ExIt dataset.

    Required dataset keys:
      - state [N, STATE_DIM], history [N, MAX_PLIES, PLY_DIM], n_plies [N]
      - legal_actions [N, MAX_LEGAL, ACTION_DIM], n_legal [N], q_targets [N, MAX_LEGAL]
    Optional keys:
      - belief_targets [N, N_CARDS], belief_marginals [N, N_CARDS, N_SEATS] (if belief_model)
      - legal_slots [N, MAX_LEGAL] (if fast_model)
    """
    device = torch.device(cfg.device)
    amp = cfg.amp and device.type == "cuda"
    q_model.to(device)
    if belief_model is not None:
        belief_model.to(device)
    if fast_model is not None:
        fast_model.to(device)

    total_rows = int(dataset["n_legal"].shape[0])
    max_legal = int(dataset["legal_actions"].shape[1])
    idx_range = torch.arange(max_legal, device=device).unsqueeze(0)

    steps_per_ep = max(1, math.ceil(total_rows / cfg.batch_size))
    total_steps = max(1, cfg.epochs * steps_per_ep)

    q_opt = torch.optim.AdamW(q_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    q_sched = torch.optim.lr_scheduler.CosineAnnealingLR(q_opt, T_max=total_steps, eta_min=cfg.min_lr)

    b_opt = b_sched = None
    if belief_model is not None and "belief_targets" in dataset and "belief_marginals" in dataset:
        b_opt = torch.optim.AdamW(belief_model.parameters(), lr=cfg.belief_lr, weight_decay=cfg.weight_decay)
        b_sched = torch.optim.lr_scheduler.CosineAnnealingLR(b_opt, T_max=total_steps, eta_min=cfg.belief_min_lr)

    f_opt = f_sched = None
    if fast_model is not None and "legal_slots" in dataset:
        f_opt = torch.optim.AdamW(fast_model.parameters(), lr=cfg.fast_lr, weight_decay=cfg.weight_decay)
        f_sched = torch.optim.lr_scheduler.CosineAnnealingLR(f_opt, T_max=total_steps, eta_min=0.0)

    rng = np.random.default_rng(cfg.seed)
    last_metrics: dict[str, float] = {}

    for ep in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        perm = torch.from_numpy(rng.permutation(total_rows))
        q_model.train()
        if belief_model is not None:
            belief_model.train()
        if fast_model is not None:
            fast_model.train()

        sums = {"q_mse": 0.0, "q_kl": 0.0, "q_top1": 0.0, "belief_ce": 0.0, "fast_ce": 0.0, "fast_top1": 0.0}
        n_steps = 0
        b_steps = 0
        f_steps = 0

        for start in range(0, total_rows, cfg.batch_size):
            mb = perm[start : start + cfg.batch_size]
            npb = dataset["n_plies"][mb].to(device, non_blocking=True)
            max_t = max(1, int(npb.max().item()))
            batch = {
                "state": dataset["state"][mb].to(device, non_blocking=True),
                "history": dataset["history"][mb, :max_t, :].to(device, non_blocking=True),
                "n_plies": npb,
                "actions": dataset["legal_actions"][mb].to(device, non_blocking=True),
                "n_legal": dataset["n_legal"][mb].to(device, non_blocking=True).long(),
                "q_targets": dataset["q_targets"][mb].to(device, non_blocking=True).float(),
            }
            if b_opt is not None:
                batch["belief_targets"] = dataset["belief_targets"][mb].to(device, non_blocking=True)
                batch["belief_marginals"] = dataset["belief_marginals"][mb].to(device, non_blocking=True).float()
            if f_opt is not None:
                batch["legal_slots"] = dataset["legal_slots"][mb].to(device, non_blocking=True).long()

            if augment is not None:
                batch = augment(batch)

            valid = idx_range < batch["n_legal"].unsqueeze(1)
            qtb = batch["q_targets"]
            t_masked = qtb.masked_fill(~valid, -1e9)
            t_probs = F.softmax(t_masked / cfg.kl_temp, dim=-1)

            # 1. QGRU step: multi-action MSE + soft-policy KL
            with dmc.autocast(device, amp):
                q_pred = q_model(batch["state"].float() if not amp else batch["state"],
                                 batch["actions"].float() if not amp else batch["actions"],
                                 batch["history"].float() if not amp else batch["history"],
                                 batch["n_plies"], valid_mask=valid)
            qf = q_pred.float()
            vf = valid.float()
            mse = (((qf - qtb) ** 2) * vf).sum() / vf.sum().clamp(min=1.0)
            q_masked = qf.masked_fill(~valid, -1e9)
            kl = -(t_probs * F.log_softmax(q_masked / cfg.kl_temp, dim=-1)).sum(dim=-1).mean()
            q_loss = mse + cfg.kl_weight * kl

            q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), cfg.grad_clip)
            q_opt.step()
            q_sched.step()

            top1 = (q_masked.argmax(dim=-1) == t_masked.argmax(dim=-1)).float().mean()
            sums["q_mse"] += float(mse.item())
            sums["q_kl"] += float(kl.item())
            sums["q_top1"] += float(top1.item())
            n_steps += 1

            # 2. Optional BeliefNet step
            if b_opt is not None:
                bt = batch["belief_targets"]
                bm = batch["belief_marginals"]
                mask = bt >= 0
                if mask.any():
                    with dmc.autocast(device, amp):
                        b_logits = belief_model(
                            batch["state"].float() if not amp else batch["state"],
                            batch["history"].float() if not amp else batch["history"],
                            batch["n_plies"], bm,
                        )
                    b_loss = F.cross_entropy(b_logits[mask], bt[mask].long())
                    b_opt.zero_grad(set_to_none=True)
                    b_loss.backward()
                    torch.nn.utils.clip_grad_norm_(belief_model.parameters(), cfg.grad_clip)
                    b_opt.step()
                    b_sched.step()
                    sums["belief_ce"] += float(b_loss.item())
                    b_steps += 1

            # 3. Optional FastPolicy step directly on search soft-policy targets
            if f_opt is not None:
                slots = batch["legal_slots"]  # [B, MAX_LEGAL], -1 invalid
                bsz = slots.shape[0]
                p_dim = fast_model.out_dim
                tgt = torch.zeros((bsz, p_dim), dtype=torch.float32, device=device)
                msk_cnt = torch.zeros((bsz, p_dim), dtype=torch.float32, device=device)
                ok = valid & (slots >= 0)
                safe_slots = slots.clamp(min=0)
                ok_f = ok.float()
                tgt.scatter_add_(1, safe_slots, t_probs * ok_f)
                msk_cnt.scatter_add_(1, safe_slots, ok_f)
                msk = msk_cnt > 0.0
                f_logits = fast_model(batch["state"].float()).masked_fill(~msk, -1e9)
                f_loss = -(tgt * F.log_softmax(f_logits, dim=-1)).sum(dim=-1).mean()
                f_opt.zero_grad(set_to_none=True)
                f_loss.backward()
                torch.nn.utils.clip_grad_norm_(fast_model.parameters(), cfg.grad_clip)
                f_opt.step()
                f_sched.step()
                f_top1 = (f_logits.argmax(dim=-1) == tgt.argmax(dim=-1)).float().mean()
                sums["fast_ce"] += float(f_loss.item())
                sums["fast_top1"] += float(f_top1.item())
                f_steps += 1

        dt = time.perf_counter() - t0
        metrics = {
            "q_mse": sums["q_mse"] / max(1, n_steps),
            "q_kl": sums["q_kl"] / max(1, n_steps),
            "q_top1": sums["q_top1"] / max(1, n_steps),
        }
        parts = [f"q_mse={metrics['q_mse']:.4f}", f"q_kl={metrics['q_kl']:.4f}", f"q_top1={100*metrics['q_top1']:.1f}%"]
        if b_steps > 0:
            metrics["belief_ce"] = sums["belief_ce"] / b_steps
            parts.append(f"belief_ce={metrics['belief_ce']:.4f}")
        if f_steps > 0:
            metrics["fast_ce"] = sums["fast_ce"] / f_steps
            metrics["fast_top1"] = sums["fast_top1"] / f_steps
            parts.append(f"fast_ce={metrics['fast_ce']:.4f} fast_top1={100*metrics['fast_top1']:.1f}%")
        if cfg.log:
            cfg.log(f"exit epoch {ep}/{cfg.epochs} ({dt:.1f}s): " + " | ".join(parts))
        last_metrics = metrics
        if on_epoch_end is not None:
            on_epoch_end(ep, metrics)

    return last_metrics
