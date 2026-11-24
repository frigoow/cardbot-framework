#pragma once
// The batched self-play environment over any cardbot game (docs/DESIGN.md,
// step 4): B games advanced together, every decision encoded for the
// networks, rewards at the end of each game. Python drives it through
// cardbot/python.h; the trainers in python/cardbot/train use it.
//
// The game's Encoder decides everything game-specific, and each part is a
// hook the game can shape as it likes:
//
//   struct MyEncoder {
//       static constexpr int kStateDim, kPlyDim, kMaxPlies, kActionDim, kMaxLegal;
//       G newGame(std::mt19937_64&) const;                 // a fresh game (deal, first seat ...)
//       bool learns(const G&) const;                       // decisions the trainer makes (e.g. card play only)
//       ActionOf<G> autoAction(const G&, std::mt19937_64&) const;   // the others (e.g. bidding by rule)
//       int encode(const G&, int seat, float* state, float* history) const;   // returns the history length
//       void encodeAction(const G&, int seat, ActionOf<G>, float* out) const;
//       float reward(const G&, int team) const;            // a finished game's return for a team
//
//       // optional, for the fast policy (steps 7 and 8): its output slots, and
//       // the slot of each action (e.g. the card id). Without them a policy
//       // scores the legal actions by position (kPolicyDim = kMaxLegal).
//       static constexpr int kPolicyDim;
//       int actionSlot(const G&, int seat, ActionOf<G>) const;
//   };

#include "cardbot/game.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace cardbot {

namespace detail {
template <class Enc, class = void>
struct HasPolicySlots : std::false_type {};
template <class Enc>
struct HasPolicySlots<Enc, std::void_t<decltype(Enc::kPolicyDim)>> : std::true_type {};

template <class Enc, class = void>
struct HasBelief : std::false_type {};
template <class Enc>
struct HasBelief<Enc, std::void_t<decltype(Enc::kBeliefCards), decltype(Enc::kBeliefSeats)>> : std::true_type {};
}  // namespace detail

/** The fast policy's number of outputs for the encoder. */
template <class Enc>
constexpr int policyDim() {
    if constexpr (detail::HasPolicySlots<Enc>::value) return Enc::kPolicyDim;
    else return Enc::kMaxLegal;
}

/** The policy output slot of legal action k (a) at the seat to move. */
template <class G, class Enc>
int actionSlot(const Enc& enc, const G& g, int seat, const ActionOf<G>& a, int k) {
    if constexpr (detail::HasPolicySlots<Enc>::value) return enc.actionSlot(g, seat, a);
    else { (void)enc; (void)g; (void)seat; (void)a; return k; }
}

template <class G, class Enc>
class BatchEnv {
public:
    using Action = ActionOf<G>;

    BatchEnv(int num_envs, std::uint64_t seed, int threads, Enc enc = Enc())
        : enc_(std::move(enc)), n_(num_envs), threads_(std::max(1, threads)) {
        resetAll(seed);
    }

    int numEnvs() const { return n_; }

    /**
     * The agent for the decisions the trainer does not make (!learns), instead
     * of Encoder::autoAction: e.g. step 6's learned bidder (cardbot/bidding.h).
     * An empty agent goes back to autoAction. Applies from the next reset.
     */
    void setAutoAgent(Agent<G> a) { auto_agent_ = std::move(a); }
    const G& game(int env) const { return games_[std::size_t(env)]; }
    const Enc& encoder() const { return enc_; }

    /** Fresh games everywhere, each at a decision the trainer makes. */
    void resetAll(std::uint64_t seed) {
        rngs_.clear();
        games_.clear();
        legal_.assign(std::size_t(n_), {});
        finished_.assign(std::size_t(n_), 0);
        for (int b = 0; b < n_; ++b) rngs_.emplace_back(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(b) + 1);
        for (int b = 0; b < n_; ++b) games_.push_back(fresh(rngs_[std::size_t(b)]));
        for (int b = 0; b < n_; ++b) legal_[std::size_t(b)] = games_[std::size_t(b)].legalActions();
    }

    /**
     * Every env's decision: state [B, kStateDim], history [B, kMaxPlies,
     * kPlyDim], n_plies [B], legal_actions [B, kMaxLegal, kActionDim], n_legal
     * [B], team [B] (the deciding seat's). Finished envs (when not reset) give
     * n_legal 0.
     */
    void observe(float* state, float* history, std::int32_t* n_plies, float* legal_actions, std::int32_t* n_legal,
                 std::int32_t* team) const {
        // checked here: an exception inside the worker threads would end the process
        for (int b = 0; b < n_; ++b)
            if (int(legal_[std::size_t(b)].size()) > Enc::kMaxLegal)
                throw std::length_error("cardbot::BatchEnv: more legal actions than kMaxLegal");
        parallel([&](int b) {
            const G& g = games_[std::size_t(b)];
            float* st = state + std::size_t(b) * Enc::kStateDim;
            float* hi = history + std::size_t(b) * Enc::kMaxPlies * Enc::kPlyDim;
            float* la = legal_actions + std::size_t(b) * Enc::kMaxLegal * Enc::kActionDim;
            std::fill(st, st + Enc::kStateDim, 0.0f);
            std::fill(hi, hi + std::size_t(Enc::kMaxPlies) * Enc::kPlyDim, 0.0f);
            std::fill(la, la + std::size_t(Enc::kMaxLegal) * Enc::kActionDim, 0.0f);
            if (finished_[std::size_t(b)] || g.finished()) {
                n_plies[b] = 0;
                n_legal[b] = 0;
                team[b] = 0;
                return;
            }
            const int seat = g.toMove();
            n_plies[b] = enc_.encode(g, seat, st, hi);
            const auto& legal = legal_[std::size_t(b)];
            for (std::size_t k = 0; k < legal.size(); ++k) enc_.encodeAction(g, seat, legal[k], la + k * Enc::kActionDim);
            n_legal[b] = int(legal.size());
            team[b] = g.team(seat);
        });
    }

    /** Each env's legal actions' policy slots, [B, kMaxLegal], -1 past n_legal. */
    void legalSlots(std::int32_t* slots) const {
        for (int b = 0; b < n_; ++b)
            if (int(legal_[std::size_t(b)].size()) > Enc::kMaxLegal)
                throw std::length_error("cardbot::BatchEnv: more legal actions than kMaxLegal");
        parallel([&](int b) {
            std::int32_t* out = slots + std::size_t(b) * Enc::kMaxLegal;
            std::fill(out, out + Enc::kMaxLegal, -1);
            const G& g = games_[std::size_t(b)];
            if (finished_[std::size_t(b)] || g.finished()) return;
            const auto& legal = legal_[std::size_t(b)];
            for (std::size_t k = 0; k < legal.size(); ++k)
                out[k] = actionSlot(enc_, g, g.toMove(), legal[k], int(k));
        });
    }

    /** Optional BeliefNet targets [B, kBeliefCards] and sampler marginals [B, kBeliefCards, kBeliefSeats]. */
    void setCollectBelief(bool on) { collect_belief_ = on; }
    bool collectBelief() const { return collect_belief_; }

    template <class E = Enc, std::enable_if_t<detail::HasBelief<E>::value, int> = 0>
    void observeBelief(std::int8_t* targets, float* marginals) const {
        constexpr std::size_t kC = std::size_t(Enc::kBeliefCards);
        constexpr std::size_t kS = std::size_t(Enc::kBeliefSeats);
        parallel([&](int b) {
            std::int8_t* t_out = targets + std::size_t(b) * kC;
            float* m_out = marginals + std::size_t(b) * kC * kS;
            std::fill(t_out, t_out + kC, std::int8_t(-1));
            std::fill(m_out, m_out + kC * kS, 0.0f);
            const G& g = games_[std::size_t(b)];
            if (finished_[std::size_t(b)] || g.finished()) return;
            enc_.encodeBelief(g, g.toMove(), t_out, m_out);
        });
    }

    /**
     * Plays legal action action_idx[b] in every env. A game that ends writes
     * done[b] = 1 and its return per team (reward[b*2], reward[b*2+1]); with
     * auto_reset it starts over at once, otherwise it waits for resetAll().
     */
    void step(const std::int32_t* action_idx, std::uint8_t* done, float* reward, bool auto_reset) {
        for (int b = 0; b < n_; ++b)
            if (!finished_[std::size_t(b)] && (action_idx[b] < 0 || action_idx[b] >= int(legal_[std::size_t(b)].size())))
                throw std::out_of_range("cardbot::BatchEnv: action index out of the legal actions");
        parallel([&](int b) {
            done[b] = 0;
            reward[2 * b] = reward[2 * b + 1] = 0.0f;
            if (finished_[std::size_t(b)]) return;
            G& g = games_[std::size_t(b)];
            const auto& legal = legal_[std::size_t(b)];
            const int k = action_idx[b];
            g.apply(legal[std::size_t(k)]);
            advance(g, rngs_[std::size_t(b)]);
            if (g.finished()) {
                done[b] = 1;
                reward[2 * b] = enc_.reward(g, 0);
                reward[2 * b + 1] = enc_.reward(g, 1);
                if (auto_reset) g = fresh(rngs_[std::size_t(b)]);
                else finished_[std::size_t(b)] = 1;
            }
            legal_[std::size_t(b)] = g.finished() ? std::vector<Action>{} : g.legalActions();
        });
    }

private:
    G fresh(std::mt19937_64& rng) const {
        for (;;) {
            G g = enc_.newGame(rng);
            advance(g, rng);
            if (!g.finished()) return g;  // e.g. a passed-out deal: deal again
        }
    }
    void advance(G& g, std::mt19937_64& rng) const {
        while (!g.finished() && !enc_.learns(g)) g.apply(auto_agent_ ? auto_agent_(g, rng) : enc_.autoAction(g, rng));
    }
    template <class F>
    void parallel(F&& f) const {
        if (threads_ <= 1 || n_ < 64) {
            for (int b = 0; b < n_; ++b) f(b);
            return;
        }
        std::vector<std::thread> pool;
        const int per = (n_ + threads_ - 1) / threads_;
        for (int t = 0; t < threads_; ++t)
            pool.emplace_back([&, t] {
                for (int b = t * per; b < std::min(n_, (t + 1) * per); ++b) f(b);
            });
        for (auto& th : pool) th.join();
    }

    Enc enc_;
    Agent<G> auto_agent_;
    int n_ = 0, threads_ = 1;
    bool collect_belief_ = false;
    std::vector<std::mt19937_64> rngs_;
    std::vector<G> games_;
    std::vector<std::vector<Action>> legal_;
    std::vector<std::uint8_t> finished_;
};

}  // namespace cardbot
