#pragma once
// The standard networks' C++ forward passes, loading what cardbot's Python
// models export (python/cardbot/models.py):
//
// - QGruModel (step 4): a state MLP and a GRU over the play history feed a
//   residual trunk; a dueling head scores each legal action from its
//   features: Q(s, a) = V(s) + A(s, a) - mean A. Bulgarian Belot's BGQ1
//   layout (same order of layers).
// - FastPolicyModel (steps 7 and 8): a small ReLU MLP from the state alone to
//   one logit per action slot, distilled from the Q network, cheap enough for
//   the search's play-outs. Bulgarian Belot's BGP1 layout.
// - BidModel (steps 5 and 6): the value of each bid for the bidder's team,
//   from the bidder's features and a GRU over the auction so far.
//
// The 4-byte tag of each file is the game's choosing.

#include "cardbot/nn.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace cardbot {

class QGruModel {
public:
    /**
     * Loads an exported Q-GRU. tag is the file's 4-byte magic (e.g. "BGQ1");
     * expect_* are the game's sizes (0: accept what the file says).
     */
    bool load(const std::string& path, const char tag[4] = "BGQ1", int expect_version = -1, int expect_state = 0,
              int expect_ply = 0, int expect_action = 0) {
        using namespace cardbot::nn;
        loaded_ = false;
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[4]{};
        std::int32_t hdr[8]{};
        if (!readBytes(in, magic, 4) || std::memcmp(magic, tag, 4) != 0) return false;
        if (!readBytes(in, hdr, sizeof(hdr))) return false;
        if ((expect_version >= 0 && hdr[0] != expect_version) || (expect_state && hdr[1] != expect_state) ||
            (expect_ply && hdr[2] != expect_ply) || (expect_action && hdr[3] != expect_action))
            return false;
        version_ = hdr[0];
        state_dim_ = hdr[1];
        ply_dim_ = hdr[2];
        action_dim_ = hdr[3];
        state_hidden_ = hdr[4];
        gru_hidden_ = hdr[5];
        act_hidden_ = hdr[6];
        adv_hidden_ = hdr[7];
        const int trunk_dim = state_hidden_ + gru_hidden_;
        if (!readLinear(in, state_fc_, state_dim_, state_hidden_) || !readLN(in, state_ln_, state_hidden_) ||
            !readGru(in, hist_enc_, ply_dim_, gru_hidden_) || !readResBlock(in, trunk1_, trunk_dim) ||
            !readResBlock(in, trunk2_, trunk_dim) || !readLinear(in, v_fc1_, trunk_dim, adv_hidden_) ||
            !readLN(in, v_ln1_, adv_hidden_) || !readLinear(in, v_fc2_, adv_hidden_, 1) ||
            !readLinear(in, act_fc_, action_dim_, act_hidden_) || !readLN(in, act_ln_, act_hidden_) ||
            !readLinear(in, adv_fc1_, trunk_dim + act_hidden_, adv_hidden_) || !readLN(in, adv_ln1_, adv_hidden_) ||
            !readResBlock(in, adv_res_, adv_hidden_) || !readLinear(in, adv_fc2_, adv_hidden_, 1)) {
            return false;
        }
        loaded_ = true;
        return true;
    }
    bool loaded() const { return loaded_; }
    int stateDim() const { return state_dim_; }
    int plyDim() const { return ply_dim_; }
    int actionDim() const { return action_dim_; }

    /** Q of each of the n_legal candidate actions (legal_actions: n_legal rows of actionDim()). */
    void forwardQ(const float* state, const float* history, int n_plies, const float* legal_actions, int n_legal,
                  float* q_out) const {
        using namespace cardbot::nn;
        const int trunk_dim = state_hidden_ + gru_hidden_;
        std::vector<float> cat(static_cast<std::size_t>(trunk_dim));
        applyLinear(state_fc_, state, cat.data());
        applyLN(state_ln_, cat.data());
        for (int i = 0; i < state_hidden_; ++i) cat[std::size_t(i)] = silu(cat[std::size_t(i)]);
        applyGru(hist_enc_, history, n_plies, cat.data() + state_hidden_);

        std::vector<float> h1(static_cast<std::size_t>(trunk_dim));
        std::vector<float> trunk(static_cast<std::size_t>(trunk_dim));
        std::vector<float> scratch;
        applyResBlock(trunk1_, cat.data(), h1.data(), scratch);
        applyResBlock(trunk2_, h1.data(), trunk.data(), scratch);

        std::vector<float> vh(static_cast<std::size_t>(adv_hidden_));
        applyLinear(v_fc1_, trunk.data(), vh.data());
        applyLN(v_ln1_, vh.data());
        for (int i = 0; i < adv_hidden_; ++i) vh[std::size_t(i)] = silu(vh[std::size_t(i)]);
        float v_val = 0.0f;
        applyLinear(v_fc2_, vh.data(), &v_val);

        std::vector<float> ae(static_cast<std::size_t>(act_hidden_));
        std::vector<float> ta(static_cast<std::size_t>(trunk_dim + act_hidden_));
        std::memcpy(ta.data(), trunk.data(), std::size_t(trunk_dim) * sizeof(float));
        std::vector<float> ah(static_cast<std::size_t>(adv_hidden_));
        std::vector<float> ar(static_cast<std::size_t>(adv_hidden_));
        std::vector<float> adv(static_cast<std::size_t>(std::max(1, n_legal)), 0.0f);
        float mean_adv = 0.0f;
        for (int k = 0; k < n_legal; ++k) {
            applyLinear(act_fc_, legal_actions + std::size_t(k * action_dim_), ae.data());
            applyLN(act_ln_, ae.data());
            for (int i = 0; i < act_hidden_; ++i) ta[std::size_t(trunk_dim + i)] = silu(ae[std::size_t(i)]);
            applyLinear(adv_fc1_, ta.data(), ah.data());
            applyLN(adv_ln1_, ah.data());
            for (int i = 0; i < adv_hidden_; ++i) ah[std::size_t(i)] = silu(ah[std::size_t(i)]);
            applyResBlock(adv_res_, ah.data(), ar.data(), scratch);
            applyLinear(adv_fc2_, ar.data(), &adv[std::size_t(k)]);
            mean_adv += adv[std::size_t(k)];
        }
        mean_adv /= float(std::max(1, n_legal));
        for (int k = 0; k < n_legal; ++k) q_out[k] = v_val + (adv[std::size_t(k)] - mean_adv);
    }

private:
    bool loaded_ = false;
    int version_ = 0, state_dim_ = 0, ply_dim_ = 0, action_dim_ = 0;
    int state_hidden_ = 0, gru_hidden_ = 0, act_hidden_ = 0, adv_hidden_ = 0;
    nn::Linear state_fc_;
    nn::LayerNorm state_ln_;
    nn::GruLayer hist_enc_;
    nn::ResBlock trunk1_, trunk2_;
    nn::Linear v_fc1_, v_fc2_;
    nn::LayerNorm v_ln1_;
    nn::Linear act_fc_;
    nn::LayerNorm act_ln_;
    nn::Linear adv_fc1_, adv_fc2_;
    nn::LayerNorm adv_ln1_;
    nn::ResBlock adv_res_;
};

/**
 * The fast policy: logits = W3 relu(W2 relu(W1 x + b1) + b2) + b3 over the
 * game's action slots (Encoder::actionSlot). The first layer walks only the
 * state's nonzero inputs (states are mostly one-hot), so a forward pass costs
 * a few microseconds. Same arithmetic, in the same order, as Bulgarian
 * Belot's BgFastPolicyModel.
 */
class FastPolicyModel {
public:
    /** expect_*: the game's sizes (version -1, sizes 0: accept what the file says). */
    bool load(const std::string& path, const char tag[4] = "BGP1", int expect_version = -1, int expect_state = 0,
              int expect_out = 0) {
        using namespace cardbot::nn;
        loaded_ = false;
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[4]{};
        std::int32_t hdr[5]{};
        if (!readBytes(in, magic, 4) || std::memcmp(magic, tag, 4) != 0) return false;
        if (!readBytes(in, hdr, sizeof(hdr))) return false;
        if ((expect_version >= 0 && hdr[0] != expect_version) || (expect_state && hdr[1] != expect_state) ||
            (expect_out && hdr[4] != expect_out) || hdr[1] <= 0 || hdr[2] <= 0 || hdr[3] <= 0 || hdr[4] <= 0)
            return false;
        version_ = hdr[0];
        state_dim_ = hdr[1];
        h1_ = hdr[2];
        h2_ = hdr[3];
        out_dim_ = hdr[4];
        std::vector<float> w1(std::size_t(h1_) * std::size_t(state_dim_));
        b1_.resize(std::size_t(h1_));
        w2_.resize(std::size_t(h2_) * std::size_t(h1_));
        b2_.resize(std::size_t(h2_));
        w3_.resize(std::size_t(out_dim_) * std::size_t(h2_));
        b3_.resize(std::size_t(out_dim_));
        if (!readBytes(in, w1.data(), w1.size() * sizeof(float)) || !readBytes(in, b1_.data(), b1_.size() * sizeof(float)) ||
            !readBytes(in, w2_.data(), w2_.size() * sizeof(float)) || !readBytes(in, b2_.data(), b2_.size() * sizeof(float)) ||
            !readBytes(in, w3_.data(), w3_.size() * sizeof(float)) || !readBytes(in, b3_.data(), b3_.size() * sizeof(float)))
            return false;
        // W1 [h1, state_dim] -> [state_dim, h1]: one contiguous column per input
        w1_t_.resize(w1.size());
        for (int j = 0; j < h1_; ++j)
            for (int i = 0; i < state_dim_; ++i)
                w1_t_[std::size_t(i) * std::size_t(h1_) + std::size_t(j)] = w1[std::size_t(j) * std::size_t(state_dim_) + std::size_t(i)];
        loaded_ = true;
        return true;
    }
    bool loaded() const { return loaded_; }
    int version() const { return version_; }
    int stateDim() const { return state_dim_; }
    int outDim() const { return out_dim_; }

    /** outDim() logits for the state (stateDim() floats). Thread-safe. */
    void forward(const float* state, float* logits_out) const {
        using namespace cardbot::nn;
        thread_local std::vector<float> z1, z2;
        z1.assign(b1_.begin(), b1_.end());
        z2.resize(std::size_t(h2_));
        for (int i = 0; i < state_dim_; ++i) {
            const float xi = state[i];
            if (xi == 0.0f) continue;
            axpy(xi, w1_t_.data() + std::size_t(i) * std::size_t(h1_), z1.data(), h1_);
        }
        for (int j = 0; j < h1_; ++j)
            if (z1[std::size_t(j)] < 0.0f) z1[std::size_t(j)] = 0.0f;
        for (int o = 0; o < h2_; ++o) {
            const float acc = b2_[std::size_t(o)] + dotProduct(w2_.data() + std::size_t(o) * std::size_t(h1_), z1.data(), h1_);
            z2[std::size_t(o)] = acc > 0.0f ? acc : 0.0f;
        }
        for (int o = 0; o < out_dim_; ++o)
            logits_out[o] = b3_[std::size_t(o)] + dotProduct(w3_.data() + std::size_t(o) * std::size_t(h2_), z2.data(), h2_);
    }

private:
    bool loaded_ = false;
    int version_ = 0, state_dim_ = 0, h1_ = 0, h2_ = 0, out_dim_ = 0;
    std::vector<float> w1_t_, b1_, w2_, b2_, w3_, b3_;
};

/**
 * The bid net: trunk = ResBlock2(ResBlock1([silu(LN(fc(features))), GRU(auction)]));
 * values = head(trunk), one per bid slot (BidEncoder::bidSlot): the expected
 * return of the deal for the bidder's team if it makes that bid. A game with a
 * richer bid net (Belot's per-suit encoder) keeps its own model and agent.
 */
class BidModel {
public:
    bool load(const std::string& path, const char tag[4] = "CBB1", int expect_version = -1, int expect_feat = 0,
              int expect_hist = 0, int expect_bids = 0) {
        using namespace cardbot::nn;
        loaded_ = false;
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[4]{};
        std::int32_t hdr[6]{};
        if (!readBytes(in, magic, 4) || std::memcmp(magic, tag, 4) != 0) return false;
        if (!readBytes(in, hdr, sizeof(hdr))) return false;
        if ((expect_version >= 0 && hdr[0] != expect_version) || (expect_feat && hdr[1] != expect_feat) ||
            (expect_hist && hdr[2] != expect_hist) || (expect_bids && hdr[3] != expect_bids))
            return false;
        for (int i = 1; i < 6; ++i)
            if (hdr[i] <= 0) return false;
        version_ = hdr[0];
        feat_dim_ = hdr[1];
        hist_dim_ = hdr[2];
        num_bids_ = hdr[3];
        hidden_ = hdr[4];
        gru_hidden_ = hdr[5];
        const int trunk_dim = hidden_ + gru_hidden_;
        if (!readLinear(in, feat_fc_, feat_dim_, hidden_) || !readLN(in, feat_ln_, hidden_) ||
            !readGru(in, hist_enc_, hist_dim_, gru_hidden_) || !readResBlock(in, trunk1_, trunk_dim) ||
            !readResBlock(in, trunk2_, trunk_dim) || !readLinear(in, head_, trunk_dim, num_bids_))
            return false;
        loaded_ = true;
        return true;
    }
    bool loaded() const { return loaded_; }
    int featDim() const { return feat_dim_; }
    int histDim() const { return hist_dim_; }
    int numBids() const { return num_bids_; }

    /** numBids() values for the bidder (features: featDim(); history: n_hist rows of histDim()). */
    void forward(const float* features, const float* history, int n_hist, float* values_out) const {
        using namespace cardbot::nn;
        const int trunk_dim = hidden_ + gru_hidden_;
        thread_local std::vector<float> cat, h1, trunk, scratch;
        cat.resize(std::size_t(trunk_dim));
        h1.resize(std::size_t(trunk_dim));
        trunk.resize(std::size_t(trunk_dim));
        applyLinear(feat_fc_, features, cat.data());
        applyLN(feat_ln_, cat.data());
        for (int i = 0; i < hidden_; ++i) cat[std::size_t(i)] = silu(cat[std::size_t(i)]);
        applyGru(hist_enc_, history, n_hist, cat.data() + hidden_);
        applyResBlock(trunk1_, cat.data(), h1.data(), scratch);
        applyResBlock(trunk2_, h1.data(), trunk.data(), scratch);
        applyLinear(head_, trunk.data(), values_out);
    }

private:
    bool loaded_ = false;
    int version_ = 0, feat_dim_ = 0, hist_dim_ = 0, num_bids_ = 0, hidden_ = 0, gru_hidden_ = 0;
    nn::Linear feat_fc_;
    nn::LayerNorm feat_ln_;
    nn::GruLayer hist_enc_;
    nn::ResBlock trunk1_, trunk2_;
    nn::Linear head_;
};

}  // namespace cardbot
