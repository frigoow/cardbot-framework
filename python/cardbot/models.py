"""The standard networks, in PyTorch, with their export to the C++ format.

QGRU is the action-as-input Q network of the Manille and Belote bots: a state
MLP and a GRU over the play history feed a residual trunk, and a dueling head
scores every legal action from its features. FastPolicy is the small MLP
distilled from it for the search's play-outs: state in, one logit per action
slot out. BidNet values each bid of an auction for the bidder's team (steps 5
and 6). Their export_bin() writes the layouts cardbot/models.h reads
(Bulgarian Belot's BGQ1 and BGP1, and CBB1 for the bid net, with a 4-byte tag of the game's choosing).
Sizes are the game's; layers are customizable by subclassing.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def write_linear(f, layer: nn.Linear) -> None:
    f.write(layer.weight.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    f.write(layer.bias.detach().cpu().contiguous().numpy().astype("<f4").tobytes())


def write_ln(f, layer: nn.LayerNorm) -> None:
    f.write(layer.weight.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    f.write(layer.bias.detach().cpu().contiguous().numpy().astype("<f4").tobytes())


class ResBlock(nn.Module):
    """out = silu(x + ln2(fc2(silu(ln1(fc1(x))))))"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.silu(x + h)

    def export_weights(self, f) -> None:
        write_linear(f, self.fc1)
        write_ln(f, self.ln1)
        write_linear(f, self.fc2)
        write_ln(f, self.ln2)


class HistoryEncoder(nn.Module):
    """A one-layer GRU over the play history; a learned embedding when it is empty."""

    def __init__(self, ply_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(ply_dim, hidden, num_layers=1, batch_first=True)
        self.empty_emb = nn.Parameter(torch.zeros(hidden))

    def forward(self, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        bsz = history.shape[0]
        max_len = int(n_plies.max().item()) if bsz > 0 else 0
        empty = self.empty_emb.unsqueeze(0).expand(bsz, -1)
        if max_len == 0:
            return empty
        out, _ = self.gru(history[:, :max_len, :])
        picked = out[torch.arange(bsz, device=history.device), (n_plies - 1).clamp(min=0).long()]
        return torch.where((n_plies > 0).unsqueeze(-1), picked, empty)

    def export_weights(self, f) -> None:
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):  # gates r, z, n
            f.write(getattr(self.gru, name).detach().cpu().contiguous().numpy().astype("<f4").tobytes())
        f.write(self.empty_emb.detach().cpu().contiguous().numpy().astype("<f4").tobytes())


class DuelingQHead(nn.Module):
    """V(s) + A(s, a) - mean of A over the legal actions."""

    def __init__(self, trunk_dim: int, action_dim: int, act_hidden: int, adv_hidden: int):
        super().__init__()
        self.act_hidden = act_hidden
        self.adv_hidden = adv_hidden
        self.v_fc1 = nn.Linear(trunk_dim, adv_hidden)
        self.v_ln1 = nn.LayerNorm(adv_hidden)
        self.v_fc2 = nn.Linear(adv_hidden, 1)
        self.act_fc = nn.Linear(action_dim, act_hidden)
        self.act_ln = nn.LayerNorm(act_hidden)
        self.adv_fc1 = nn.Linear(trunk_dim + act_hidden, adv_hidden)
        self.adv_ln1 = nn.LayerNorm(adv_hidden)
        self.adv_res = ResBlock(adv_hidden)
        self.adv_fc2 = nn.Linear(adv_hidden, 1)

    def forward(self, trunk: torch.Tensor, actions: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        v = self.v_fc2(F.silu(self.v_ln1(self.v_fc1(trunk))))
        b, k, _ = actions.shape
        ae = F.silu(self.act_ln(self.act_fc(actions)))
        te = trunk.unsqueeze(1).expand(b, k, -1)
        h = F.silu(self.adv_ln1(self.adv_fc1(torch.cat([te, ae], dim=-1))))
        adv = self.adv_fc2(self.adv_res(h)).squeeze(-1)
        if valid_mask is not None:
            vm = valid_mask.to(adv.dtype)
            mean_adv = (adv * vm).sum(-1, keepdim=True) / vm.sum(-1, keepdim=True).clamp(min=1.0)
        else:
            mean_adv = adv.mean(-1, keepdim=True)
        return v + (adv - mean_adv)

    def export_weights(self, f) -> None:
        write_linear(f, self.v_fc1)
        write_ln(f, self.v_ln1)
        write_linear(f, self.v_fc2)
        write_linear(f, self.act_fc)
        write_ln(f, self.act_ln)
        write_linear(f, self.adv_fc1)
        write_ln(f, self.adv_ln1)
        self.adv_res.export_weights(f)
        write_linear(f, self.adv_fc2)


class QGRU(nn.Module):
    """Q(state, history, action) for every legal action (an action-as-input DMC network)."""

    def __init__(self, state_dim: int, ply_dim: int, action_dim: int, state_hidden: int = 384,
                 gru_hidden: int = 192, act_hidden: int = 128, adv_hidden: int = 256):
        super().__init__()
        self.state_dim, self.ply_dim, self.action_dim = state_dim, ply_dim, action_dim
        self.state_hidden, self.gru_hidden = state_hidden, gru_hidden
        trunk_dim = state_hidden + gru_hidden
        self.state_fc = nn.Linear(state_dim, state_hidden)
        self.state_ln = nn.LayerNorm(state_hidden)
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk1 = ResBlock(trunk_dim)
        self.trunk2 = ResBlock(trunk_dim)
        self.q_head = DuelingQHead(trunk_dim, action_dim, act_hidden, adv_hidden)

    def trunk(self, state: torch.Tensor, history: torch.Tensor, n_plies: torch.Tensor) -> torch.Tensor:
        se = F.silu(self.state_ln(self.state_fc(state)))
        he = self.hist_enc(history, n_plies)
        return self.trunk2(self.trunk1(torch.cat([se, he], dim=-1)))

    def forward(self, state, actions, history, n_plies, valid_mask=None) -> torch.Tensor:
        return self.q_head(self.trunk(state, history, n_plies), actions, valid_mask)

    def export_bin(self, path: str | Path, tag: bytes = b"BGQ1", version: int = 1) -> None:
        """The layout cardbot::QGruModel::load reads: tag, 8 int32 sizes, then the layers."""
        if len(tag) != 4:
            raise ValueError("tag must be 4 bytes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(tag)
            f.write(struct.pack("<8i", version, self.state_dim, self.ply_dim, self.action_dim, self.state_hidden,
                                self.gru_hidden, self.q_head.act_hidden, self.q_head.adv_hidden))
            write_linear(f, self.state_fc)
            write_ln(f, self.state_ln)
            self.hist_enc.export_weights(f)
            self.trunk1.export_weights(f)
            self.trunk2.export_weights(f)
            self.q_head.export_weights(f)


class FastPolicy(nn.Module):
    """logits over the game's action slots = fc3(relu(fc2(relu(fc1(state))))), for C++ play-outs."""

    def __init__(self, state_dim: int, out_dim: int, h1: int = 256, h2: int = 128):
        super().__init__()
        self.state_dim, self.out_dim, self.h1, self.h2 = state_dim, out_dim, h1, h2
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))

    def export_bin(self, path: str | Path, tag: bytes = b"BGP1", version: int = 1) -> None:
        """The layout cardbot::FastPolicyModel::load reads: tag, 5 int32 sizes, then the three layers."""
        if len(tag) != 4:
            raise ValueError("tag must be 4 bytes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(tag)
            f.write(struct.pack("<5i", version, self.state_dim, self.h1, self.h2, self.out_dim))
            for layer in (self.fc1, self.fc2, self.fc3):
                write_linear(f, layer)


class BidNet(nn.Module):
    """values over the bid slots = head(trunk2(trunk1([silu(ln(fc(features))), GRU(auction)])))."""

    def __init__(self, feat_dim: int, hist_dim: int, num_bids: int, hidden: int = 128, gru_hidden: int = 48):
        super().__init__()
        self.feat_dim, self.hist_dim, self.num_bids = feat_dim, hist_dim, num_bids
        self.hidden, self.gru_hidden = hidden, gru_hidden
        trunk_dim = hidden + gru_hidden
        self.feat_fc = nn.Linear(feat_dim, hidden)
        self.feat_ln = nn.LayerNorm(hidden)
        self.hist_enc = HistoryEncoder(hist_dim, gru_hidden)
        self.trunk1 = ResBlock(trunk_dim)
        self.trunk2 = ResBlock(trunk_dim)
        self.head = nn.Linear(trunk_dim, num_bids)

    def forward(self, feat: torch.Tensor, hist: torch.Tensor, n_hist: torch.Tensor) -> torch.Tensor:
        f = F.silu(self.feat_ln(self.feat_fc(feat)))
        h = self.hist_enc(hist, n_hist)
        return self.head(self.trunk2(self.trunk1(torch.cat([f, h], dim=-1))))

    def export_bin(self, path: str | Path, tag: bytes = b"CBB1", version: int = 1) -> None:
        """The layout cardbot::BidModel::load reads: tag, 6 int32 sizes, then the layers."""
        if len(tag) != 4:
            raise ValueError("tag must be 4 bytes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(tag)
            f.write(struct.pack("<6i", version, self.feat_dim, self.hist_dim, self.num_bids, self.hidden,
                                self.gru_hidden))
            write_linear(f, self.feat_fc)
            write_ln(f, self.feat_ln)
            self.hist_enc.export_weights(f)
            self.trunk1.export_weights(f)
            self.trunk2.export_weights(f)
            write_linear(f, self.head)
