"""The bid net (docs/DESIGN.md, step 5), trained as the Belote bot trains its
bidder: counterfactual bid values from the game's BidEnv (cardbot/bidding.h).
Every legal bid at an auction decision is played out on the same deal, so the
net is fitted to each bid's return for the bidder's team, not only to the
one bid that was made.

The recipe, each part replaceable:

- streaming: every iteration generates fresh deals and uses them once, so
  the net cannot memorize deals (generate);
- warm-up: the first iterations' auctions follow the game's baseline bidder;
  after that the learning seats bid with the current net (self-play);
- the bidder is an exponential moving average of the weights (EMA), which is
  exported to the C++ side every iteration and is what train() returns;
- the loss is the squared error on the legal bids, plus a paired term on each
  bid's advantage over a reference (the mean of the legal bids, or a fixed
  slot such as pass), which compares bids on the same deal and cancels the
  deal's luck (bid_loss).

Step 6 (refit) is the DMC trainer again, with the env's auctions bid by the
trained net: env.use_bid_model(path, tag), then dmc.train() warm-started from
the step-4 network.
"""

from __future__ import annotations

import copy
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cardbot.train import dmc


@dataclass
class BidConfig:
    iters: int = 200
    deals_per_iter: int = 4096
    samples: int = 1                   # play-outs per bid branch (averaged; same random streams for every bid)
    batch_size: int = 1024
    lr: float = 1e-3
    min_lr: float = 1e-4               # cosine decay to this
    weight_decay: float = 1e-4
    eps_start: float = 0.15            # random bids on the auction's actual course, decaying linearly
    eps_end: float = 0.03
    adv_weight: float = 1.0            # weight of the paired advantage term
    ref_slot: int | None = None        # advantage over this slot (e.g. pass); None: over the legal bids' mean
    ema_decay: float = 0.995
    warmup_iters: int = 5              # auctions by the baseline bidder until then
    tag: bytes = b"CBB1"
    out: str | Path | None = None      # where the EMA net is exported (default: a temporary file)
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print
    extra: dict = field(default_factory=dict)  # anything a customized generate/loss wants


def generate(env, cfg: BidConfig, it: int, eps: float) -> dict:
    """One iteration's fresh counterfactual rows from the env."""
    return env.generate(cfg.deals_per_iter, cfg.seed * 1_000_003 + it, eps, cfg.samples)


def bid_loss(model, batch: dict, cfg: BidConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared error over the legal bids plus cfg.adv_weight times the paired advantage error.
    Returns the loss and the value MSE alone (for the log)."""
    pred = model(batch["feat"], batch["hist"], batch["n_hist"])
    mask, tgt = batch["legal_mask"], batch["targets"]
    mse = (((pred - tgt) * mask) ** 2).sum() / mask.sum().clamp(min=1.0)
    if cfg.adv_weight <= 0:
        return mse, mse
    if cfg.ref_slot is None:
        n = mask.sum(-1, keepdim=True).clamp(min=1.0)
        pred_ref = (pred * mask).sum(-1, keepdim=True) / n
        tgt_ref = (tgt * mask).sum(-1, keepdim=True) / n
        adv_mask = mask
    else:
        pred_ref = pred[:, cfg.ref_slot:cfg.ref_slot + 1]
        tgt_ref = tgt[:, cfg.ref_slot:cfg.ref_slot + 1]
        adv_mask = mask * mask[:, cfg.ref_slot:cfg.ref_slot + 1]
    adv = ((((pred - pred_ref) - (tgt - tgt_ref)) * adv_mask) ** 2).sum() / adv_mask.sum().clamp(min=1.0)
    return mse + cfg.adv_weight * adv, mse


@torch.no_grad()
def _update_ema(ema, model, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


def train(env, model, cfg: BidConfig, generate_fn=generate, loss_fn=bid_loss, on_iter=None):
    """cfg.iters of: generate fresh rows, one pass over them, EMA update; from the end of
    the warm-up the EMA net bids for the env's learning seats. Returns the EMA net."""
    device = cfg.device
    model.to(device)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tmp = None
    if cfg.out is None:
        tmp = tempfile.TemporaryDirectory()
        out = Path(tmp.name) / "bid.bin"
    else:
        out = Path(cfg.out)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    steps = 0
    try:
        for it in range(cfg.iters):
            t0 = time.time()
            frac = it / max(1, cfg.iters - 1)
            lr = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * frac))
            for pg in opt.param_groups:
                pg["lr"] = lr
            eps = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac
            data = dmc._to(device, **generate_fn(env, cfg, it, eps))
            n = len(data["targets"])
            model.train()
            perm = torch.randperm(n, generator=g).to(device)
            mses = []
            for i in range(0, n, cfg.batch_size):
                sel = perm[i:i + cfg.batch_size]
                loss, mse = loss_fn(model, {k: v[sel] for k, v in data.items()}, cfg)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                _update_ema(ema, model, min(cfg.ema_decay, (1.0 + steps) / (10.0 + steps)))
                mses.append(mse.item())
            if it + 1 >= cfg.warmup_iters:
                ema.export_bin(out, tag=cfg.tag)
                env.use_bid_model(str(out), cfg.tag.decode())
            if cfg.log:
                phase = "self-play" if it + 1 > cfg.warmup_iters else "warm-up"
                cfg.log(f"bid iter {it + 1}/{cfg.iters} ({phase}): {n} decisions, value RMSE "
                        f"{math.sqrt(max(0.0, float(np.mean(mses)))):.4f}, lr {lr:.1e}, eps {eps:.3f}, "
                        f"{time.time() - t0:.1f}s")
            if on_iter:
                on_iter(it, ema)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return ema


def evaluate(env, pairs: int = 2000, seed: int = 7, against: int = 1) -> tuple[float, float]:
    """The env's bid net (bidder 0) against bidder `against` (1: the baseline) on duplicate
    deals, both playing the cards with the env's play agent. Returns the mean return per deal
    (in the game's payoff units) and its 95% half-width."""
    return env.evaluate(0, against, pairs, seed)
