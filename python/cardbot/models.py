"""The standard networks, in PyTorch, with their export to the C++ format.

QGRU is the action-as-input Q network of the Manille and Belote bots: a state
MLP and a GRU over the play history feed a residual trunk, and a dueling head
scores every legal action from its features. BeliefNet corrects the world
sampler's card marginals. FastPolicy is the small MLP
distilled from it for the search's play-outs: state in, one logit per action
slot out. BidNet values each bid of an auction for the bidder's team (steps 5
and 6). Their export_bin() writes the layouts cardbot/models.h reads
(Bulgarian Belot's BGQ1, BGB1 and BGP1, and CBB1 for the bid net, with a 4-byte tag of the game's choosing).
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
        """Q for each of the K action rows; with valid_mask, only the first k_active rows any
        sample uses are computed (the rest are 0), and the advantages are centred over the
        valid ones."""
        v = self.v_fc2(F.silu(self.v_ln1(self.v_fc1(trunk))))
        b, k_full, _ = actions.shape
        if valid_mask is not None and valid_mask.any():
            k_active = int(valid_mask.any(dim=0).nonzero(as_tuple=False).max().item()) + 1
            act_in = actions[:, :k_active, :]
            vm = valid_mask[:, :k_active].to(v.dtype)
        else:
            k_active = k_full
            act_in = actions
            vm = None
        ae = F.silu(self.act_ln(self.act_fc(act_in)))
        te = trunk.unsqueeze(1).expand(b, k_active, -1)
        h = F.silu(self.adv_ln1(self.adv_fc1(torch.cat([te, ae], dim=-1))))
        h = self.adv_res(h)
        adv = self.adv_fc2(h).squeeze(-1)
        if vm is not None:
            mean_adv = (adv * vm).sum(dim=-1, keepdim=True) / vm.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            mean_adv = adv.mean(dim=-1, keepdim=True)
        q_active = v + (adv - mean_adv)
        if k_active == k_full:
            return q_active
        q_out = torch.zeros((b, k_full), dtype=q_active.dtype, device=q_active.device)
        q_out[:, :k_active] = q_active
        return q_out

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
        # A subclass can add training-only heads on the trunk (e.g. Belot's card ownership),
        # which export_bin() leaves out.
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


class BeliefNet(nn.Module):
    """Where each hidden card is (one of n_seats owners), as a correction to the exact
    marginals of the world sampler (cardbot/sampler.h's marginals()): logits =
    log(marginals) + delta(state, history, marginals), the delta head starting at zero so
    the untrained net is the sampler. Cards the sampler rules out stay impossible."""

    def __init__(self, state_dim: int, ply_dim: int, n_cards: int = 32, n_seats: int = 3,
                 state_hidden: int = 256, gru_hidden: int = 128):
        super().__init__()
        self.state_dim, self.ply_dim = state_dim, ply_dim
        self.n_cards, self.n_seats = n_cards, n_seats
        self.state_hidden, self.gru_hidden = state_hidden, gru_hidden
        trunk_dim = state_hidden + gru_hidden
        self.state_fc = nn.Linear(state_dim + n_cards * n_seats, state_hidden)
        self.state_ln = nn.LayerNorm(state_hidden)
        self.hist_enc = HistoryEncoder(ply_dim, gru_hidden)
        self.trunk1 = ResBlock(trunk_dim)
        self.trunk2 = ResBlock(trunk_dim)
        self.delta_head = nn.Linear(trunk_dim, n_cards * n_seats)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, state, history, n_plies, sampler_marginals) -> torch.Tensor:
        """sampler_marginals: [B, n_cards, n_seats]; returns logits of the same shape."""
        m_flat = sampler_marginals.reshape(state.shape[0], self.n_cards * self.n_seats).to(state.dtype)
        se = F.silu(self.state_ln(self.state_fc(torch.cat([state, m_flat], dim=-1))))
        he = self.hist_enc(history, n_plies)
        h = self.trunk2(self.trunk1(torch.cat([se, he], dim=-1)))
        delta = self.delta_head(h).view(-1, self.n_cards, self.n_seats).float()
        base = torch.log(sampler_marginals.float().clamp(min=1e-6))
        return (base + delta).masked_fill(sampler_marginals <= 0.0, -1e9)

    def export_bin(self, path: str | Path, tag: bytes = b"BGB1", version: int = 1) -> None:
        """Tag, 5 int32 sizes (version, state, ply, state hidden, GRU hidden), then the layers
        (Bulgarian Belot's BGB1 layout; the reader knows the cards and seats)."""
        if len(tag) != 4:
            raise ValueError("tag must be 4 bytes")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(tag)
            f.write(struct.pack("<5i", version, self.state_dim, self.ply_dim, self.state_hidden, self.gru_hidden))
            write_linear(f, self.state_fc)
            write_ln(f, self.state_ln)
            self.hist_enc.export_weights(f)
            self.trunk1.export_weights(f)
            self.trunk2.export_weights(f)
            write_linear(f, self.delta_head)


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
