#pragma once
// The standard networks' C++ forward passes, loading what cardbot's Python
// models export (python/cardbot/models.py). So far the Q-GRU: a state MLP and
// a GRU over the play history feed a residual trunk; a dueling head scores
// each legal action from its features: Q(s, a) = V(s) + A(s, a) - mean A.
// The file layout is Bulgarian Belot's BGQ1 (same order of layers), with the
// 4-byte tag of the game's choosing.

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

}  // namespace cardbot
