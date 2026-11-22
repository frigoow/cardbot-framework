"""Deep Monte-Carlo (DMC) self-play for the action-as-input Q network
(docs/DESIGN.md, step 4), as DouZero introduced it and the Manille and Belote
bots use it: every seat plays the current Q network epsilon-greedily, each
decision's target is the final reward of the deciding seat's team, and
Q(s, a) is fitted to it by mean squared error.

Works on any game module made with cardbot/python.h (its BatchEnv). The parts
are separate so a game can replace any of them: collect() gathers decisions,
q_loss() is the loss, train() runs rounds of both, evaluate() measures a
greedy network against another policy on mirrored deals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DMCConfig:
    rounds: int = 100                  # collect + train cycles
    num_envs: int = 1024               # games collected per round
    batch_size: int = 1024
    epochs: int = 2                    # passes over each round's decisions
    lr: float = 3e-4
    eps_start: float = 0.1             # exploration, decaying linearly over the rounds
    eps_end: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1
    threads: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print
    extra: dict = field(default_factory=dict)  # anything a customized collect/loss wants


def _to(device, **arrays):
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


@torch.no_grad()
def choose(model, obs: dict, eps: float, rng: np.random.Generator, device: str) -> np.ndarray:
    """Epsilon-greedy action index per env (0 for envs without a decision)."""
    n_legal = obs["n_legal"]
    idx = np.zeros(len(n_legal), dtype=np.int32)
    active = np.nonzero(n_legal > 0)[0]
    if len(active) == 0:
        return idx
    t = _to(device, state=obs["state"][active], history=obs["history"][active],
            n_plies=obs["n_plies"][active], actions=obs["legal_actions"][active])
    k = t["actions"].shape[1]
    valid = torch.arange(k, device=device).unsqueeze(0) < torch.as_tensor(n_legal[active], device=device).unsqueeze(1)
    q = model(t["state"], t["actions"], t["history"], t["n_plies"], valid)
    q = q.masked_fill(~valid, -1e9)
    greedy = q.argmax(dim=1).cpu().numpy()
    explore = rng.random(len(active)) < eps
    random_pick = (rng.random(len(active)) * n_legal[active]).astype(np.int32)
    idx[active] = np.where(explore, random_pick, greedy)
    return idx


def collect(env, model, eps: float, seed: int, device: str) -> dict:
    """Plays every env's game to the end once; returns each decision with its target return."""
    env.reset_all(seed)
    rng = np.random.default_rng(seed)
    n = env.num_envs
    per_env = [[] for _ in range(n)]  # (state, history, n_plies, legal actions, n_legal, chosen index, team)
    targets = {}
    while True:
        obs = env.observe()
        if not (obs["n_legal"] > 0).any():
            break
        idx = choose(model, obs, eps, rng, device)
        for b in np.nonzero(obs["n_legal"] > 0)[0]:
            per_env[b].append((obs["state"][b], obs["history"][b], obs["n_plies"][b], obs["legal_actions"][b],
                               obs["n_legal"][b], idx[b], obs["team"][b]))
        done, reward = env.step(idx, auto_reset=False)
        for b in np.nonzero(done)[0]:
            targets[b] = reward[b]
    cols = {k: [] for k in ("state", "history", "n_plies", "actions", "n_legal", "chosen", "ret")}
    for b in range(n):
        if b not in targets:
            continue
        for s, h, p, la, nl, i, team in per_env[b]:
            for k, v in zip(cols, (s, h, p, la, nl, i, targets[b][team])):
                cols[k].append(v)
    return {"state": np.stack(cols["state"]), "history": np.stack(cols["history"]),
            "n_plies": np.asarray(cols["n_plies"], dtype=np.int32), "actions": np.stack(cols["actions"]),
            "n_legal": np.asarray(cols["n_legal"], dtype=np.int64), "chosen": np.asarray(cols["chosen"], dtype=np.int64),
            "ret": np.asarray(cols["ret"], dtype=np.float32)}


def q_loss(model, batch: dict) -> torch.Tensor:
    """MSE between Q(s, chosen action) and the decision's return. The network sees every
    legal action (its dueling head centres the advantages over them), and the chosen one's Q
    is taken; given the chosen action alone the advantage would cancel out."""
    k = batch["actions"].shape[1]
    valid = torch.arange(k, device=batch["actions"].device).unsqueeze(0) < batch["n_legal"].unsqueeze(1)
    q = model(batch["state"], batch["actions"], batch["history"], batch["n_plies"], valid)
    return F.mse_loss(q.gather(1, batch["chosen"].unsqueeze(1)).squeeze(1), batch["ret"])


def train(env, model, cfg: DMCConfig, collect_fn=collect, loss_fn=q_loss, on_round=None):
    """cfg.rounds of: collect cfg.num_envs games with the current network, then fit it."""
    device = cfg.device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for r in range(cfg.rounds):
        t0 = time.time()
        eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * r / max(1, cfg.rounds - 1)
        model.eval()
        data = collect_fn(env, model, eps, cfg.seed * 100003 + r, device)
        model.train()
        n = len(data["ret"])
        tensors = _to(device, **data)
        losses = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=g).to(device)
            for i in range(0, n, cfg.batch_size):
                sel = perm[i:i + cfg.batch_size]
                loss = loss_fn(model, {k: v[sel] for k, v in tensors.items()})
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(loss.item())
        if cfg.log:
            cfg.log(f"round {r + 1}/{cfg.rounds}: {n} decisions, eps {eps:.3f}, loss {np.mean(losses):.4f}, "
                    f"{time.time() - t0:.1f}s")
        if on_round:
            on_round(r, model)
    return model


def play_mirrored(env, pick, seed: int = 7) -> tuple[float, float]:
    """Team 0's and team 1's seats played by pick(obs, rng) -> action indices against uniformly
    random play, each env's deal twice with the teams swapped. Returns pick's mean return per
    deal and its 95% half-width over the env pairs."""
    totals = np.zeros(env.num_envs)
    for net_team in (0, 1):
        env.reset_all(seed)
        rng = np.random.default_rng(seed + 1 + net_team)
        while True:
            obs = env.observe()
            if not (obs["n_legal"] > 0).any():
                break
            mine = pick(obs, rng)
            rand = (rng.random(env.num_envs) * np.maximum(obs["n_legal"], 1)).astype(np.int32)
            idx = np.where(obs["team"] == net_team, mine, rand).astype(np.int32)
            done, reward = env.step(idx, auto_reset=False)
            totals += np.where(done, reward[:, net_team], 0.0) / 2.0
    mean = float(totals.mean())
    half = 1.96 * float(totals.std(ddof=1)) / math.sqrt(len(totals))
    return mean, half


@torch.no_grad()
def evaluate(env, model, seed: int = 7, device: str | None = None) -> tuple[float, float]:
    """The greedy Q network against random play on mirrored deals (play_mirrored)."""
    device = device or next(model.parameters()).device.type
    model.eval()
    return play_mirrored(env, lambda obs, rng: choose(model, obs, 0.0, rng, device), seed)
