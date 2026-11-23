"""Distilling the Q network into the fast policy (docs/DESIGN.md, step 7), as
the Belote bot does it: the trained Q network (the teacher) plays self-play
games greedily; at every decision with a choice, its Q values over the legal
actions, turned into probabilities by softmax(Q / temperature), are the target
of the small MLP (the student, cardbot.models.FastPolicy), which sees the
state alone and scores the game's action slots (the encoder's actionSlot,
observe()'s legal_slots). The loss is the cross-entropy against those soft
targets over the legal slots. The student then plays the search's play-outs in
C++ (cardbot/policy.h, step 8).

As in dmc.py, the parts are separate so a game can replace any: collect()
gathers the teacher's decisions, soft_ce_loss() is the loss, train() fits
(with an optional augment(state, mask, target) per minibatch, e.g. a symmetry
of the game), choose() and evaluate() play the student.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from cardbot.train import dmc


@dataclass
class DistillConfig:
    rounds: int = 8                    # collection rounds of env.num_envs games each
    epochs: int = 8                    # passes over all collected decisions
    batch_size: int = 2048
    lr: float = 1.5e-3
    min_lr: float = 0.0                # cosine decay over all steps down to this
    weight_decay: float = 1e-4
    temperature: float = 0.04          # of the teacher's softmax, in the game's return units
    eps: float = 0.0                   # the teacher's exploration while collecting (0: greedy)
    min_legal: int = 2                 # decisions with fewer legal actions are left out
    seed: int = 1                      # the minibatch order (and the exploration)
    collect_seed: Callable[[int], int] | None = None   # round -> env seed (default: seed * 100003 + round)
    amp: bool = True                   # the teacher in BF16 autocast on CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print
    extra: dict = field(default_factory=dict)  # anything a customized collect/loss wants


@torch.no_grad()
def collect(env, teacher, cfg: DistillConfig, seed: int) -> dict:
    """Self-play games by the teacher: every kept decision's state, legal-slot mask and
    target probabilities over the policy slots, as NumPy arrays."""
    device = cfg.device
    env.reset_all(seed)
    rng = np.random.default_rng(seed)
    bsz, p, k = env.num_envs, env.POLICY_DIM, env.MAX_LEGAL
    idx_range = torch.arange(k, device=device).unsqueeze(0)
    states, masks, targets = [], [], []
    finished = np.zeros(bsz, dtype=bool)
    rows = np.repeat(np.arange(bsz), k).reshape(bsz, k)
    for _ in range(env.MAX_PLIES):
        if finished.all():
            break
        obs = env.observe()
        nl = obs["n_legal"]
        max_p = int(obs["n_plies"].max())
        s_t = torch.from_numpy(obs["state"]).to(device)
        a_t = torch.from_numpy(obs["legal_actions"]).to(device)
        h_t = (torch.from_numpy(obs["history"][:, :max_p, :]).to(device) if max_p > 0
               else torch.zeros((bsz, 1, env.PLY_DIM), dtype=torch.float32, device=device))
        np_t = torch.from_numpy(obs["n_plies"]).to(device)
        valid = idx_range < torch.from_numpy(nl).to(device).unsqueeze(1)
        with dmc.autocast(device, cfg.amp):
            q = teacher(s_t, a_t, h_t, np_t, valid_mask=valid).float().masked_fill(~valid, -1e9)
        probs = F.softmax(q / cfg.temperature, dim=-1).cpu().numpy()
        greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int32)
        keep = np.nonzero((~finished) & (nl >= max(1, cfg.min_legal)))[0]
        if len(keep):
            slots = obs["legal_slots"][keep]
            ok = slots >= 0
            r = rows[: len(keep)]
            tgt = np.zeros((len(keep), p), dtype=np.float32)
            msk = np.zeros((len(keep), p), dtype=np.float32)
            np.add.at(tgt, (r[ok], slots[ok]), probs[keep][ok])
            msk[r[ok], slots[ok]] = 1.0
            states.append(obs["state"][keep])
            masks.append(msk)
            targets.append(tgt)
        idx = greedy
        if cfg.eps > 0:
            explore = (rng.random(bsz) < cfg.eps) & (nl > 1)
            idx = np.where(explore, (rng.random(bsz) * np.maximum(nl, 1)).astype(np.int32), greedy)
        done, _ = env.step(idx.astype(np.int32), auto_reset=False)
        finished |= done != 0
    return {"state": np.concatenate(states), "mask": np.concatenate(masks), "target": np.concatenate(targets)}


def soft_ce_loss(student, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the student's legal-slot softmax against the targets, and its top-1
    agreement with the target's best slot."""
    logits = student(batch["state"]).masked_fill(batch["mask"] == 0, -1e9)
    loss = -(batch["target"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    top1 = (logits.argmax(dim=-1) == batch["target"].argmax(dim=-1)).float().mean()
    return loss, top1


def train(env, teacher, student, cfg: DistillConfig, collect_fn=collect, loss_fn=soft_ce_loss, augment=None):
    """Collects cfg.rounds rounds of the teacher's decisions, then fits the student to them.
    augment(state, mask, target) -> the same three, per minibatch (optional)."""
    device = cfg.device
    teacher.to(device).eval()
    student.to(device)
    seed_of = cfg.collect_seed or (lambda r: cfg.seed * 100003 + r)
    t0 = time.time()
    parts = [collect_fn(env, teacher, cfg, seed_of(r)) for r in range(cfg.rounds)]
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    n = len(data["state"])
    if cfg.log:
        cfg.log(f"distill: {n} decisions from {cfg.rounds} x {env.num_envs} games, {time.time() - t0:.1f}s")
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * math.ceil(n / cfg.batch_size),
                                                       eta_min=cfg.min_lr)
    rng = np.random.default_rng(cfg.seed)
    student.train()
    for ep in range(cfg.epochs):
        perm = rng.permutation(n)
        tot_loss = tot_acc = 0.0
        steps = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            batch = {k: torch.from_numpy(v[idx]).to(device) for k, v in data.items()}
            if augment is not None:
                batch["state"], batch["mask"], batch["target"] = augment(batch["state"], batch["mask"], batch["target"])
            loss, top1 = loss_fn(student, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            tot_acc += float(top1.item())
            steps += 1
        if cfg.log:
            cfg.log(f"distill epoch {ep + 1}/{cfg.epochs}: soft CE {tot_loss / steps:.4f}, "
                    f"top-1 agreement {100 * tot_acc / steps:.1f}%")
    student.eval()
    return student


@torch.no_grad()
def choose(student, obs: dict, device) -> np.ndarray:
    """The student's highest-logit legal action index per env (0 for envs without a decision)."""
    slots = torch.as_tensor(obs["legal_slots"], device=device).long()
    logits = student(torch.as_tensor(obs["state"], device=device))
    picked = logits.gather(1, slots.clamp(min=0)).masked_fill(slots < 0, -1e9)
    return picked.argmax(dim=1).cpu().numpy().astype(np.int32)


@torch.no_grad()
def evaluate(env, student, seed: int = 7, device=None) -> tuple[float, float]:
    """The greedy student against random play on mirrored deals (dmc.play_mirrored)."""
    device = device or next(student.parameters()).device
    student.eval()
    return dmc.play_mirrored(env, lambda obs, rng: choose(student, obs, device), seed)
