"""Steps 5 and 6 on the toy game with an auction (cpp/tests/mini_trump.h): the
bid net's export against the C++ forward pass, counterfactual bid training
beating the baseline bidder, and the card-play env's auctions by a bid net
(the refit)."""

import numpy as np
import pytest
import torch

mt = pytest.importorskip("mini_tricks_cpp")

from cardbot.models import QGRU, BidNet
from cardbot.train import bidding, dmc


def small_bid_net(env):
    return BidNet(env.FEAT_DIM, env.HIST_DIM, env.NUM_BIDS, hidden=64, gru_hidden=32)


def test_bid_export_matches_cpp_forward(tmp_path):
    env = mt.TrumpBidEnv(2)
    torch.manual_seed(0)
    net = small_bid_net(env).eval()
    path = tmp_path / "b.bin"
    net.export_bin(path, tag=b"TEST")
    cpp = mt.BidModel()
    assert not cpp.load(str(path))
    assert cpp.load(str(path), "TEST")
    d = env.generate(64, 5, 0.3)
    assert (d["n_hist"] > 0).any()  # later seats too, so the auction GRU matters
    with torch.no_grad():
        v = net(torch.as_tensor(d["feat"]), torch.as_tensor(d["hist"]), torch.as_tensor(d["n_hist"])).numpy()
    for i in range(len(v)):
        np.testing.assert_allclose(cpp.forward(d["feat"][i], d["hist"][i], int(d["n_hist"][i])), v[i], atol=1e-5)


def test_bid_net_beats_baseline_bidder():
    env = mt.TrumpBidEnv(4)
    before, before_ci = bidding.evaluate(env, 2000)  # bidder 0 is the baseline until a net is loaded
    assert abs(before) < 3 * before_ci + 1e-9
    torch.manual_seed(1)
    net = bidding.train(env, small_bid_net(env), bidding.BidConfig(iters=30, deals_per_iter=2048, log=None))
    after, after_ci = bidding.evaluate(env, 3000)
    print(f"\nbid net vs baseline bidder (random card play): {after:+.3f} +- {after_ci:.3f} per deal "
          f"(scores -5..+3)")
    assert after - after_ci > 0.1


def test_refit_env_bids_with_the_bid_net(tmp_path):
    env = mt.TrumpEnv(256, 3, 2)
    # a rigged bid net: always name suit 0 (slot 1)
    net = small_bid_net(mt.TrumpBidEnv)
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.copy_(torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0]))
    path = tmp_path / "rigged.bin"
    net.export_bin(path, tag=b"TEST")
    trump = slice(mt.TrumpEnv.STATE_DIM - 8, mt.TrumpEnv.STATE_DIM - 4)
    env.reset_all(4)
    baseline_trumps = env.observe()["state"][:, trump].argmax(1)
    assert len(set(baseline_trumps)) > 1
    env.use_bid_model(str(path), "TEST")
    env.reset_all(4)
    assert (env.observe()["state"][:, trump].argmax(1) == 0).all()
    # the refit itself: DMC on those auctions runs as usual
    q = QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM, state_hidden=32, gru_hidden=16, act_hidden=16, adv_hidden=32)
    dmc.train(env, q, dmc.DMCConfig(rounds=2, num_envs=256, batch_size=256, epochs=1, log=None))
    env.use_auto_action()
    env.reset_all(4)
    np.testing.assert_array_equal(env.observe()["state"][:, trump].argmax(1), baseline_trumps)
