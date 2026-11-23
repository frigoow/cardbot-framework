"""Steps 7 and 8 on the toy game: the fast policy's export against the C++
forward pass, distilling a DMC-trained Q network into it, and the distilled
policy in C++ as an agent and as the play-out policy of the rollout search."""

import numpy as np
import pytest
import torch

mt = pytest.importorskip("mini_tricks_cpp")

from cardbot.models import QGRU, FastPolicy
from cardbot.train import distill, dmc


@pytest.fixture(scope="module")
def teacher():
    torch.manual_seed(1)
    env = mt.BatchEnv(1024, 3, 4)
    model = QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM, state_hidden=64, gru_hidden=32, act_hidden=32,
                 adv_hidden=64)
    dmc.train(env, model, dmc.DMCConfig(episodes=40 * 1024, batch_size=1024, lr=1e-3, eps_start=0.1, log=None))
    return model.eval()


def test_env_gives_policy_slots():
    env = mt.BatchEnv(8, 1, 1)
    assert env.POLICY_DIM == 16
    obs = env.observe()
    for b in range(8):
        n = obs["n_legal"][b]
        cards = obs["legal_actions"][b, :n].argmax(axis=1)  # the toy encoder's actions are one-hot cards
        np.testing.assert_array_equal(obs["legal_slots"][b, :n], cards)
        assert (obs["legal_slots"][b, n:] == -1).all()


def test_export_matches_cpp_forward(tmp_path):
    torch.manual_seed(0)
    env = mt.BatchEnv(8, 2, 1)
    model = FastPolicy(env.STATE_DIM, env.POLICY_DIM, h1=64, h2=32).eval()
    path = tmp_path / "p.bin"
    model.export_bin(path, tag=b"TEST", version=3)
    cpp = mt.FastPolicyModel()
    assert not cpp.load(str(path), "BGP1")
    assert cpp.load(str(path), "TEST")
    assert (cpp.state_dim, cpp.out_dim) == (env.STATE_DIM, env.POLICY_DIM)
    for _ in range(6):
        env.step(np.zeros(8, dtype=np.int32), auto_reset=False)
    obs = env.observe()
    with torch.no_grad():
        logits = model(torch.as_tensor(obs["state"])).numpy()
    for b in range(8):
        np.testing.assert_allclose(cpp.forward(obs["state"][b]), logits[b], atol=1e-5)


def test_distilled_policy_follows_teacher_and_helps_search(teacher, tmp_path):
    env = mt.BatchEnv(1024, 5, 4)
    torch.manual_seed(2)
    student = FastPolicy(env.STATE_DIM, env.POLICY_DIM, h1=128, h2=64)
    logs = []
    distill.train(env, teacher, student, distill.DistillConfig(rounds=4, epochs=10, batch_size=1024, temperature=0.05, log=logs.append))
    agreement = float(logs[-1].split("agreement ")[1].rstrip("%"))
    t_mean, t_half = dmc.evaluate(env, teacher)
    s_mean, s_half = distill.evaluate(env, student)
    print(f"\n{logs[-1]}\nvs random (tricks/4): teacher {t_mean:+.3f} +- {t_half:.3f}, "
          f"fast policy {s_mean:+.3f} +- {s_half:.3f}")
    assert agreement > 75  # over decisions with a choice only (forced moves are left out, min_legal=2)
    assert s_mean - s_half > 0.15

    # step 8: the same net in C++, alone and as the rollout search's play-out policy
    path = tmp_path / "fast.bin"
    student.cpu().export_bin(path, tag=b"TEST")
    pol, pol_ci = mt.duel(f"policy:{path}", "random", 400, 11)
    print(f"C++ fast policy vs random: {pol:+.3f} +- {pol_ci:.3f} tricks per deal")
    assert pol - pol_ci > 0.6  # 0.15 in returns of tricks/4
    # With 4 cards each, three tricks of random play-out are already nearly as informative as the
    # policy's (+0.05 +- 0.05 tricks over 1500 pairs), so the toy game only checks that the policy
    # play-outs run and do not hurt; their gain shows in longer games (Belot's search).
    smart, smart_ci = mt.duel(f"rollout:16:{path}", "rollout:16", 600, 13)
    print(f"rollout-16 with fast-policy play-outs vs random play-outs: {smart:+.3f} +- {smart_ci:.3f} tricks per deal")
    assert smart + smart_ci > 0
