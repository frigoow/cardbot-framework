"""cardbot's Python side on the toy game (cpp/tests/mini_tricks.h through its
module mini_tricks_cpp, built with -DCARDBOT_PYTHON=ON): the environment, the
model's export against the C++ forward pass, and DMC learning to beat random."""

import numpy as np
import pytest
import torch

mt = pytest.importorskip("mini_tricks_cpp")

from cardbot.models import QGRU
from cardbot.train import dmc


def small_model(env):
    return QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM, state_hidden=64, gru_hidden=32, act_hidden=32,
                adv_hidden=64)


def test_env_plays_zero_sum_games():
    env = mt.BatchEnv(16, 1, 2)
    obs = env.observe()
    assert obs["state"].shape == (16, env.STATE_DIM)
    assert obs["legal_actions"].shape == (16, env.MAX_LEGAL, env.ACTION_DIM)
    rng = np.random.default_rng(0)
    finished = 0
    while True:
        obs = env.observe()
        if not (obs["n_legal"] > 0).any():
            break
        idx = (rng.random(16) * np.maximum(obs["n_legal"], 1)).astype(np.int32)
        done, reward = env.step(idx, auto_reset=False)
        for b in np.nonzero(done)[0]:
            finished += 1
            assert reward[b, 0] == -reward[b, 1]
    assert finished == 16


def test_export_matches_cpp_forward(tmp_path):
    torch.manual_seed(0)
    env = mt.BatchEnv(8, 2, 1)
    model = small_model(env).eval()
    path = tmp_path / "q.bin"
    model.export_bin(path, tag=b"TEST", version=1)
    cpp = mt.QGruModel()
    assert cpp.load(str(path), "TEST")
    # a few plies into the games, so the history matters
    for _ in range(5):
        env.step(np.zeros(8, dtype=np.int32), auto_reset=False)
    obs = env.observe()
    with torch.no_grad():
        q = model(torch.as_tensor(obs["state"]), torch.as_tensor(obs["legal_actions"]),
                  torch.as_tensor(obs["history"]), torch.as_tensor(obs["n_plies"]),
                  torch.arange(env.MAX_LEGAL).unsqueeze(0) < torch.as_tensor(obs["n_legal"]).unsqueeze(1)).numpy()
    for b in range(8):
        n = int(obs["n_legal"][b])
        qc = cpp.forward_q(obs["state"][b], obs["history"][b], int(obs["n_plies"][b]), obs["legal_actions"][b], n)
        np.testing.assert_allclose(qc, q[b, :n], atol=1e-4)


def test_dmc_learns_to_beat_random():
    torch.manual_seed(1)
    env = mt.BatchEnv(1024, 3, 4)
    model = small_model(env)
    before, _ = dmc.evaluate(env, model.to(dmc.DMCConfig().device))
    cfg = dmc.DMCConfig(episodes=40 * 1024, batch_size=1024, lr=1e-3, eps_start=0.1, log=None)
    dmc.train(env, model, cfg)
    after, half = dmc.evaluate(env, model)
    print(f"greedy Q vs random: {before:+.3f} before training, {after:+.3f} +- {half:.3f} after (return = tricks/4)")
    assert after - half > 0.15
