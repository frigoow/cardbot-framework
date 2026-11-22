#pragma once
// The distilled fast policy as an agent and as a prior (docs/DESIGN.md, steps 7
// and 8). It reads the state the game's Encoder writes (cardbot/env.h) and
// scores each legal action by its output slot (Encoder::actionSlot). As the
// play-out agent of rolloutAgent (cardbot/agents.h), or a game's own search,
// it replaces random play-outs; legalLogits() gives a search its move prior.

#include "cardbot/agents.h"
#include "cardbot/env.h"
#include "cardbot/models.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace cardbot {

/** The fast policy's logit for each of g's legal actions (in legalActions() order). */
template <class G, class Enc>
std::vector<float> legalLogits(const FastPolicyModel& model, const Enc& enc, const G& g,
                               const std::vector<ActionOf<G>>& legal) {
    thread_local std::vector<float> state, history, logits;
    state.assign(std::size_t(Enc::kStateDim), 0.0f);
    history.assign(std::size_t(Enc::kMaxPlies) * Enc::kPlyDim, 0.0f);
    logits.resize(std::size_t(model.outDim()));
    const int seat = g.toMove();
    enc.encode(g, seat, state.data(), history.data());
    model.forward(state.data(), logits.data());
    std::vector<float> out(legal.size());
    for (std::size_t k = 0; k < legal.size(); ++k) {
        const int slot = actionSlot(enc, g, seat, legal[k], int(k));
        if (slot < 0 || slot >= model.outDim()) throw std::out_of_range("cardbot: action slot outside the policy's outputs");
        out[k] = logits[std::size_t(slot)];
    }
    return out;
}

/**
 * Plays the fast policy: the highest-logit legal action, or with temperature
 * > 0 one sampled from softmax(logits / temperature) over the legal actions.
 */
template <class G, class Enc>
Agent<G> fastPolicyAgent(std::shared_ptr<const FastPolicyModel> model, Enc enc = Enc(), double temperature = 0.0) {
    if (!model || !model->loaded()) throw std::invalid_argument("fastPolicyAgent: no loaded model");
    if (model->stateDim() != Enc::kStateDim) throw std::invalid_argument("fastPolicyAgent: state size differs from the encoder's");
    return [model, enc, temperature](const G& g, std::mt19937_64& rng) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const auto lg = legalLogits(*model, enc, g, legal);
        if (temperature <= 0.0) return legal[std::size_t(std::max_element(lg.begin(), lg.end()) - lg.begin())];
        const float mx = *std::max_element(lg.begin(), lg.end());
        std::vector<double> w(lg.size());
        for (std::size_t k = 0; k < lg.size(); ++k) w[k] = std::exp((double(lg[k]) - mx) / temperature);
        return legal[std::discrete_distribution<std::size_t>(w.begin(), w.end())(rng)];
    };
}

}  // namespace cardbot
