"""Unit test for cardbot.train.exit on synthetic multi-action search Q-targets."""

import numpy as np
import torch

from cardbot.models import BeliefNet, FastPolicy, QGRU
from cardbot.train import exit as cb_exit


def test_exit_trains_q_belief_and_fast():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, state_dim, ply_dim, max_plies, action_dim, max_legal, policy_dim = 256, 32, 12, 8, 16, 4, 16
    state = rng.normal(size=(n, state_dim)).astype(np.float32)
    history = rng.normal(size=(n, max_plies, ply_dim)).astype(np.float32)
    n_plies = rng.integers(0, max_plies + 1, size=(n,), dtype=np.int32)
    actions = rng.normal(size=(n, max_legal, action_dim)).astype(np.float32)
    n_legal = np.full((n,), max_legal, dtype=np.int32)
    # Linearly separable synthetic Q targets so QGRU and FastPolicy must fit cleanly
    w = rng.normal(size=(action_dim,)).astype(np.float32)
    q_targets = (actions @ w).astype(np.float32)
    legal_slots = np.tile(np.arange(max_legal, dtype=np.int32), (n, 1))
    belief_targets = rng.integers(0, 3, size=(n, policy_dim), dtype=np.int8)
    belief_marginals = np.full((n, policy_dim, 3), 1.0 / 3.0, dtype=np.float32)

    ds = cb_exit.concat_chunks([{
        "state": state,
        "history": history,
        "n_plies": n_plies,
        "legal_actions": actions,
        "n_legal": n_legal,
        "q_targets": q_targets,
        "legal_slots": legal_slots,
        "belief_targets": belief_targets,
        "belief_marginals": belief_marginals,
    }])

    q = QGRU(state_dim, ply_dim, action_dim, state_hidden=32, gru_hidden=16, act_hidden=16, adv_hidden=32)
    b = BeliefNet(state_dim, ply_dim, n_cards=policy_dim, n_seats=3, state_hidden=32, gru_hidden=16)
    f = FastPolicy(state_dim, policy_dim, h1=32, h2=16)

    cfg = cb_exit.ExitConfig(epochs=6, batch_size=64, lr=3e-3, fast_lr=3e-3, kl_temp=0.5, log=None, device="cpu")
    metrics = cb_exit.train(ds, q, cfg, belief_model=b, fast_model=f)
    assert metrics["q_top1"] > 0.75
    assert metrics["belief_ce"] < 1.2
    assert metrics["fast_ce"] > 0.0
