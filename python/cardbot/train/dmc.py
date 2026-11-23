"""Deep Monte-Carlo (DMC) self-play for the action-as-input Q network
(docs/DESIGN.md, step 4), with the recipe of the Manille and Belote bots:

- every seat plays the current Q network epsilon-greedily (only where it has a
  choice), every decision's target is the final return of the deciding seat's
  team, and Q(s, chosen action) is fitted to it by mean squared error;
- the decisions of the last `replay_rounds` rounds stay on the training
  device (a rolling replay window), and each round makes `mb_per_round`
  minibatch steps over it; minibatches are cut from buckets of similar
  history length, so the GRU runs over little padding;
- AdamW with a cosine learning rate over the run, gradient clipping, BF16
  autocast and storage on CUDA, exploration decaying linearly over the run.

The trainer is a class whose game-specific parts are methods to override:
env_flags() (what the env should observe this round), augment() (e.g. Belot's
suit permutations), loss() (e.g. an auxiliary head), after_round() (e.g.
training a belief net on the round's decisions), on_round_end() (evaluation,
checkpoints). The env is any game module made with cardbot/python.h, or an
adapter with the same observe()/step() protocol.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DMCConfig:
    episodes: int = 5_000_000          # games played over the run (rounds of env.num_envs games)
    batch_size: int = 1024
    replay_rounds: int = 4             # rounds kept in the replay window
    mb_per_round: int = 32             # minibatch steps per round over the window
    lr: float = 5e-4
    min_lr: float = 3e-5               # cosine decay over the run down to this
    weight_decay: float = 1e-4
    eps_start: float = 0.08            # exploration, decaying linearly over the run
    eps_end: float = 0.01
    grad_clip: float = 2.0
    ply_bucket: int = 8                # minibatches come from buckets of n_plies // ply_bucket ...
    n_buckets: int = 4                 # ... capped at n_buckets - 1
    max_decisions: int | None = None   # decisions per game at most (default: the env's MAX_PLIES)
    extra_keys: tuple = ()             # further observe() arrays to keep per decision
    ring_keys: tuple = ()              # those of them the replay window keeps as well
    amp: bool = True                   # BF16 autocast and storage on CUDA
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log: Callable[[str], None] | None = print
    log_every: int = 25                # rounds between log lines
    extra: dict = field(default_factory=dict)  # anything an overriding method wants


def _to(device, **arrays):
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def autocast(device, enabled: bool = True):
    """BF16 autocast on CUDA, nothing elsewhere."""
    if enabled and torch.device(device).type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class DMCTrainer:
    """DMC on a batched env. Override the hooks for a game's own parts; run() trains."""

    CORE = ("state", "history", "n_plies", "actions", "n_legal", "chosen", "ret")

    def __init__(self, env, model, cfg: DMCConfig):
        self.env, self.model, self.cfg = env, model, cfg
        self.device = torch.device(cfg.device)
        self.amp = cfg.amp and self.device.type == "cuda"
        self.dtype = torch.bfloat16 if self.amp else torch.float32
        model.to(self.device)
        self.total_rounds = max(1, math.ceil(cfg.episodes / env.num_envs))
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.total_rounds, eta_min=cfg.min_lr)
        self.rng = np.random.default_rng(cfg.seed + 999)
        self.max_decisions = cfg.max_decisions or env.MAX_PLIES
        self.episodes_done = 0
        self.round_idx = 0
        self.ring: list[dict] = []
        self.ring_ptr = 0
        self._buf: dict[str, torch.Tensor] = {}
        self.idx_range = torch.arange(env.MAX_LEGAL, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------ hooks

    def env_flags(self) -> None:
        """Before the round's games: set what the env observes (e.g. extra targets)."""

    def augment(self, batch: dict) -> dict:
        """A training minibatch before the loss (e.g. a random symmetry of the game)."""
        return batch

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """The minibatch loss, and the numbers to log: the MSE of Q(s, chosen) against the return."""
        valid = self.idx_range < batch["n_legal"].unsqueeze(1)
        with autocast(self.device, self.amp):
            q_all = self.model(batch["state"], batch["actions"], batch["history"], batch["n_plies"], valid_mask=valid)
        q_taken = q_all.float().gather(1, batch["chosen"].unsqueeze(1)).squeeze(1)
        q_mse = F.mse_loss(q_taken, batch["ret"])
        return q_mse, {"q_mse": q_mse}

    def after_round(self, new: dict) -> None:
        """After the round's Q steps, with the round's own decisions (e.g. train a belief net)."""

    def on_round_end(self) -> None:
        """After the scheduler step and the log line (evaluation, checkpoints)."""

    def should_log(self) -> bool:
        return self.round_idx % self.cfg.log_every == 0 or self.episodes_done >= self.cfg.episodes

    def log_line(self, logs: dict, t_start: float, start_episodes: int) -> str:
        dps = (self.episodes_done - start_episodes) / max(time.perf_counter() - t_start, 1e-6)
        parts = " ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "eps")
        return (f"[{self.episodes_done:>9,}/{self.cfg.episodes:,}] {dps:6.0f} d/s | eps={logs['eps']:.3f} "
                f"lr={self.opt.param_groups[0]['lr']:.2e} | {parts}")

    # ------------------------------------------------------------------ the recipe

    def eps(self) -> float:
        frac = min(1.0, self.episodes_done / max(1, self.cfg.episodes))
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    def _buffer(self, key: str, shape, dtype) -> torch.Tensor:
        """A [max_decisions, *shape] buffer kept from round to round (zeros when created)."""
        b = self._buf.get(key)
        if b is None or b.shape[1:] != torch.Size(shape) or b.dtype != dtype:
            b = torch.zeros((self.max_decisions, *shape), dtype=dtype, device=self.device)
            self._buf[key] = b
        return b

    @torch.no_grad()
    def collect(self, eps: float, round_seed: int) -> dict:
        """Plays every env's game to its end with the current network; returns the round's
        decisions, flattened: the CORE keys, "team", and the extra keys the env observed."""
        env, dev, bsz = self.env, self.device, self.env.num_envs
        env.reset_all(round_seed)
        finished = np.zeros(bsz, dtype=bool)
        final = np.zeros((bsz, 2), dtype=np.float32)
        valid_step = torch.zeros((self.max_decisions, bsz), dtype=torch.bool, device=dev)
        teams = torch.zeros((self.max_decisions, bsz), dtype=torch.int8, device=dev)
        hists = self._buffer("history", (bsz, env.MAX_PLIES, env.PLY_DIM), self.dtype)
        seen = []
        self.model.eval()
        steps_run = 0
        with autocast(dev, self.amp):
            for step in range(self.max_decisions):
                if finished.all():
                    break
                steps_run = step + 1
                active = ~finished
                obs = env.observe()
                s_t = torch.from_numpy(obs["state"]).to(dev)
                a_t = torch.from_numpy(obs["legal_actions"]).to(dev)
                np_np, nl_np = obs["n_plies"], obs["n_legal"]
                np_t = torch.from_numpy(np_np).to(dev)
                nl_t = torch.from_numpy(nl_np).to(dev)
                max_p = int(np_np.max()) if bsz > 0 else 0
                if max_p > 0:
                    h_in = torch.from_numpy(obs["history"][:, :max_p, :]).to(dev)
                    hists[step, :, :max_p, :].copy_(h_in)
                else:
                    h_in = hists[0, :, :1, :]
                self._buffer("state", (bsz, env.STATE_DIM), self.dtype)[step].copy_(s_t)
                self._buffer("n_plies", (bsz,), torch.int32)[step].copy_(np_t)
                self._buffer("actions", (bsz, env.MAX_LEGAL, env.ACTION_DIM), self.dtype)[step].copy_(a_t)
                self._buffer("n_legal", (bsz,), torch.int8)[step].copy_(nl_t)
                teams[step].copy_(torch.from_numpy(obs["team"]).to(dev))
                valid_step[step].copy_(torch.from_numpy(active).to(dev))
                for k in self.cfg.extra_keys:
                    if k in obs:
                        arr = obs[k]
                        dt = self.dtype if np.issubdtype(arr.dtype, np.floating) else torch.from_numpy(arr[:0]).dtype
                        self._buffer(k, arr.shape, dt)[step].copy_(torch.from_numpy(arr).to(dev))
                        if k not in seen:
                            seen.append(k)

                valid = self.idx_range < nl_t.unsqueeze(1)
                q = self.model(s_t, a_t, h_in, np_t, valid_mask=valid).float().masked_fill(~valid, -1e9)
                greedy = q.argmax(dim=-1).cpu().numpy().astype(np.int32)
                if eps > 0.0:
                    explore = (self.rng.random(bsz) < eps) & (nl_np > 1)
                    rand_idx = (self.rng.integers(0, 10000, size=bsz) % np.maximum(nl_np, 1)).astype(np.int32)
                    action_idx = np.where(explore, rand_idx, greedy)
                else:
                    action_idx = greedy
                self._buffer("chosen", (bsz,), torch.int8)[step].copy_(
                    torch.from_numpy(action_idx.astype(np.int8)).to(dev))
                done, reward = env.step(action_idx.astype(np.int32), auto_reset=False)
                newly = active & (done != 0)
                if newly.any():
                    final[newly] = reward[newly]
                    finished |= newly
        if not finished.all():  # a game longer than max_decisions is left out
            valid_step[:, torch.from_numpy(~finished).to(dev)] = False

        fin = torch.from_numpy(final).to(dev)
        t = teams[:steps_run].long()
        ret = torch.where(t == 0, fin[:, 0].unsqueeze(0), fin[:, 1].unsqueeze(0))
        v_idx = torch.nonzero(valid_step[:steps_run].reshape(-1), as_tuple=False).squeeze(-1)

        def flat(key, *shape):
            return self._buf[key][:steps_run].reshape(-1, *shape)[v_idx]

        new = {
            "state": flat("state", env.STATE_DIM),
            "history": flat("history", env.MAX_PLIES, env.PLY_DIM),
            "n_plies": flat("n_plies"),
            "actions": flat("actions", env.MAX_LEGAL, env.ACTION_DIM),
            "n_legal": flat("n_legal").long(),
            "chosen": flat("chosen").long(),
            "ret": ret.reshape(-1)[v_idx],
            "team": t.reshape(-1)[v_idx],
        }
        for k in seen:
            b = self._buf[k]
            x = b[:steps_run].reshape(-1, *b.shape[2:])[v_idx]
            new[k] = x if x.is_floating_point() else x.long()
        return new

    def _push(self, new: dict) -> dict:
        keys = self.CORE + tuple(k for k in self.cfg.ring_keys if k in new)
        entry = {k: new[k] for k in keys}
        cap = max(1, self.cfg.replay_rounds)
        if len(self.ring) < cap:
            self.ring.append(entry)
        else:
            self.ring[self.ring_ptr] = entry
        self.ring_ptr = (self.ring_ptr + 1) % cap
        if len(self.ring) == 1:
            return self.ring[0]
        return {k: torch.cat([e[k] for e in self.ring], dim=0) for k in keys}

    def minibatches(self, data: dict) -> list[tuple[int, torch.Tensor]]:
        """(history length, indices) of the minibatches of `data`, by history-length bucket."""
        n_plies = data["n_plies"]
        buckets = (n_plies // self.cfg.ply_bucket).clamp(0, self.cfg.n_buckets - 1)
        out = []
        for bkt in range(self.cfg.n_buckets):
            idx = torch.nonzero(buckets == bkt, as_tuple=False).squeeze(-1)
            nb = idx.numel()
            if nb == 0:
                continue
            perm = idx[torch.randperm(nb, device=self.device)]
            h_len = max(1, int(n_plies[idx].max().item()))
            for start in range(0, nb, self.cfg.batch_size):
                out.append((h_len, perm[start:start + self.cfg.batch_size]))
        return out

    def train_round(self) -> dict | None:
        """One round: collect, the replay window, mb_per_round steps, after_round, the
        scheduler. None when the round produced no decision."""
        cfg = self.cfg
        eps = self.eps()
        self.env_flags()
        new = self.collect(eps, cfg.seed * 1000000 + self.round_idx)
        self.episodes_done += self.env.num_envs
        self.round_idx += 1
        if new["ret"].numel() == 0:
            return None
        data = self._push(new)
        self.model.train()
        sums: dict[str, float] = {}
        n_mb = 0
        schedule = self.minibatches(data)
        if schedule:
            order = self.rng.permutation(len(schedule))
            for oi in order[:min(len(order), max(1, cfg.mb_per_round))]:
                h_len, idx = schedule[int(oi)]
                batch = {k: (v[idx, :h_len, :] if k == "history" else v[idx]) for k, v in data.items()}
                batch = self.augment(batch)
                loss, parts = self.loss(batch)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.opt.step()
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.item())
                n_mb += 1
        self.after_round(new)
        self.sched.step()
        logs = {k: v / max(1, n_mb) for k, v in sums.items()}
        logs["eps"] = eps
        return logs

    def run(self):
        """Rounds until cfg.episodes games are played; returns the model."""
        t_start = time.perf_counter()
        start_episodes = self.episodes_done
        while self.episodes_done < self.cfg.episodes:
            logs = self.train_round()
            if logs is None:
                continue
            if self.cfg.log and self.should_log():
                self.cfg.log(self.log_line(logs, t_start, start_episodes))
            self.on_round_end()
        return self.model


def train(env, model, cfg: DMCConfig):
    """DMCTrainer(env, model, cfg).run(), for a game with nothing to override."""
    return DMCTrainer(env, model, cfg).run()


@torch.no_grad()
def choose(model, obs: dict, device, amp: bool = False) -> np.ndarray:
    """The greedy Q network's action index per env (0 for envs without a decision)."""
    t = _to(device, state=obs["state"], history=obs["history"], n_plies=obs["n_plies"], actions=obs["legal_actions"])
    valid = torch.arange(t["actions"].shape[1], device=device).unsqueeze(0) < torch.as_tensor(
        obs["n_legal"], device=device).unsqueeze(1)
    with autocast(device, amp):
        q = model(t["state"], t["actions"], t["history"], t["n_plies"], valid)
    return q.float().masked_fill(~valid, -1e9).argmax(dim=1).cpu().numpy().astype(np.int32)


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
def evaluate(env, model, seed: int = 7, device=None) -> tuple[float, float]:
    """The greedy Q network against random play on mirrored deals (play_mirrored)."""
    device = device or next(model.parameters()).device
    model.eval()
    return play_mirrored(env, lambda obs, rng: choose(model, obs, device), seed)
