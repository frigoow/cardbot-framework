"""Distilling the Q network into the fast policy (docs/DESIGN.md, step 7), as
the Belote bot does: the trained Q network (the teacher) plays self-play
games; at every decision its Q values over the legal actions, turned into
probabilities by softmax(Q / temperature), are the target of the small MLP
(the student, cardbot.models.FastPolicy), which sees the state alone and
scores the game's action slots (the encoder's actionSlot, observe()'s
legal_slots). The loss is the cross-entropy against those soft targets over
the legal slots. The student then plays the search's play-outs in C++
(cardbot/policy.h, step 8).

As in dmc.py, the parts are separate so a game can replace any: collect()
gathers the teacher's decisions, soft_ce_loss() is the loss, train() fits,
choose() and evaluate() play the student.
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
    epochs: int = 10                   # passes over all collected decisions
    batch_size: int = 4096
    lr: float = 1.5e-3
    min_lr: float = 3e-5               # cosine decay to this
    weight_decay: float = 1e-4
    temperature: float = 0.04          # of the teacher's softmax, in the game's return units
    eps: float = 0.05                  # the teacher's exploration while collecting (wider coverage)
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print
    extra: dict = field(default_factory=dict)  # anything a customized collect/loss wants


@torch.no_grad()
def collect(env, teacher, cfg: DistillConfig, seed: int) -> dict:
    """Self-play games by the teacher; every decision's state, legal-slot mask and target
    probabilities over the policy slots."""
    device = cfg.device
    env.reset_all(seed)
    rng = np.random.default_rng(seed)
    p = env.POLICY_DIM
    states, masks, targets = [], [], []
    while True:
        obs = env.observe()
        active = np.nonzero(obs["n_legal"] > 0)[0]
        if len(active) == 0:
            break
        t = dmc._to(device, state=obs["state"][active], history=obs["history"][active],
                    n_plies=obs["n_plies"][active], actions=obs["legal_actions"][active])
        n_legal = torch.as_tensor(obs["n_legal"][active], device=device)
        k = t["actions"].shape[1]
        valid = torch.arange(k, device=device).unsqueeze(0) < n_legal.unsqueeze(1)
        q = teacher(t["state"], t["actions"], t["history"], t["n_plies"], valid).float().masked_fill(~valid, -1e9)
        probs = F.softmax(q / cfg.temperature, dim=-1).cpu().numpy()
        slots = obs["legal_slots"][active]
        tgt = np.zeros((len(active), p), dtype=np.float32)
        msk = np.zeros((len(active), p), dtype=np.float32)
        rows = np.repeat(np.arange(len(active)), k).reshape(len(active), k)
        ok = slots >= 0
        np.add.at(tgt, (rows[ok], slots[ok]), probs[ok])
        msk[rows[ok], slots[ok]] = 1.0
        states.append(obs["state"][active])
        masks.append(msk)
        targets.append(tgt)
        greedy = q.argmax(dim=1).cpu().numpy()
        explore = rng.random(len(active)) < cfg.eps
        idx = np.zeros(env.num_envs, dtype=np.int32)
        idx[active] = np.where(explore, (rng.random(len(active)) * obs["n_legal"][active]).astype(np.int32), greedy)
        env.step(idx, auto_reset=False)
    return {"state": np.concatenate(states), "mask": np.concatenate(masks), "target": np.concatenate(targets)}


def soft_ce_loss(student, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy of the student's legal-slot softmax against the targets, and its top-1
    agreement with the target's best slot."""
    logits = student(batch["state"]).masked_fill(batch["mask"] == 0, -1e9)
    loss = -(batch["target"] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    top1 = (logits.argmax(dim=-1) == batch["target"].argmax(dim=-1)).float().mean()
    return loss, top1


def train(env, teacher, student, cfg: DistillConfig, collect_fn=collect, loss_fn=soft_ce_loss):
    """Collects cfg.rounds rounds of the teacher's decisions, then fits the student to them."""
    device = cfg.device
    teacher.to(device).eval()
    student.to(device)
    t0 = time.time()
    parts = [collect_fn(env, teacher, cfg, cfg.seed * 100003 + r) for r in range(cfg.rounds)]
    data = dmc._to(device, **{k: np.concatenate([d[k] for d in parts]) for k in parts[0]})
    n = len(data["state"])
    if cfg.log:
        cfg.log(f"distill: {n} decisions from {cfg.rounds} x {env.num_envs} games, {time.time() - t0:.1f}s")
    steps = cfg.epochs * math.ceil(n / cfg.batch_size)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=cfg.min_lr)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    student.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(n, generator=g).to(device)
        losses, accs = [], []
        for i in range(0, n, cfg.batch_size):
            sel = perm[i:i + cfg.batch_size]
            loss, top1 = loss_fn(student, {k: v[sel] for k, v in data.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            accs.append(top1.item())
        if cfg.log:
            cfg.log(f"distill epoch {ep + 1}/{cfg.epochs}: soft CE {np.mean(losses):.4f}, "
                    f"top-1 agreement {100 * np.mean(accs):.1f}%")
    student.eval()
    return student


@torch.no_grad()
def choose(student, obs: dict, device: str) -> np.ndarray:
    """The student's highest-logit legal action index per env (0 for envs without a decision)."""
    slots = torch.as_tensor(obs["legal_slots"], device=device).long()
    logits = student(torch.as_tensor(obs["state"], device=device))
    picked = logits.gather(1, slots.clamp(min=0)).masked_fill(slots < 0, -1e9)
    return picked.argmax(dim=1).cpu().numpy().astype(np.int32)


@torch.no_grad()
def evaluate(env, student, seed: int = 7, device: str | None = None) -> tuple[float, float]:
    """The greedy student against random play on mirrored deals (dmc.play_mirrored)."""
    device = device or next(student.parameters()).device.type
    student.eval()
    return dmc.play_mirrored(env, lambda obs, rng: choose(student, obs, device), seed)
