#pragma once
// Agents for any game. Baselines (docs/DESIGN.md, step 2): uniform random, and
// ISMCTS (single-observer information-set Monte Carlo tree search, through the
// bundled ismcsolver, MIT, Steven Franzen). ISMCTS needs only the game's
// determinize(); the result it maximizes is the game's payoff, scaled by
// IsmctsOptions::scale into the range its exploration constant expects.
//
// And a determinized play-out search (steps 3 and 8): every legal action is
// tried in sampled worlds and played to the end by a play-out agent, random
// at first and the distilled fast policy later (cardbot/policy.h).

#include "cardbot/game.h"

#include <ismcts/game.h>
#include <ismcts/sosolver.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cardbot {

/** Uniformly random among the legal actions. */
template <class G>
Agent<G> randomAgent() {
    return [](const G& g, std::mt19937_64& rng) {
        const auto legal = g.legalActions();
        if (legal.empty()) throw std::logic_error("randomAgent: no legal action");
        return legal[std::uniform_int_distribution<std::size_t>(0, legal.size() - 1)(rng)];
    };
}

struct IsmctsOptions {
    std::size_t iterations = 1000;
    unsigned threads = 1;      // > 1: root parallel, iterations per thread
    double scale = 1.0;        // payoff * scale is the reward the tree backs up
};

/** A cardbot game as ismcsolver's partially observable game. */
template <class G>
class IsmctsGame : public ISMCTS::POMGame<ActionOf<G>> {
public:
    using Base = ISMCTS::POMGame<ActionOf<G>>;
    using typename Base::Clone;
    using typename Base::Player;

    IsmctsGame(G g, double scale) : g_(std::move(g)), scale_(scale) {}

    Clone cloneAndRandomise(Player observer) const override {
        thread_local std::mt19937_64 rng{std::random_device{}()};
        return std::make_unique<IsmctsGame>(g_.determinize(int(observer), rng), scale_);
    }
    Player currentPlayer() const override { return Player(g_.toMove()); }
    std::vector<Player> players() const override {
        std::vector<Player> p;
        for (int s = 0; s < G::kPlayers; ++s) p.push_back(Player(s));
        return p;
    }
    std::vector<ActionOf<G>> validMoves() const override { return g_.finished() ? std::vector<ActionOf<G>>{} : g_.legalActions(); }
    void doMove(ActionOf<G> a) override { g_.apply(a); }
    double getResult(Player player) const override { return g_.payoff(int(player)) * scale_; }

private:
    G g_;
    double scale_;
};

/** ISMCTS from the seat to move; a single legal action is played at once. */
template <class G>
Agent<G> ismctsAgent(IsmctsOptions o = {}) {
    return [o](const G& g, std::mt19937_64&) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const IsmctsGame<G> root(g, o.scale);
        if (o.threads <= 1) {
            ISMCTS::SOSolver<ActionOf<G>> solver(o.iterations);
            return solver(root);
        }
        ISMCTS::SOSolver<ActionOf<G>, ISMCTS::RootParallel> solver(o.iterations, o.threads);
        return solver(root);
    };
}

struct RolloutOptions {
    int worlds = 32;           // determinizations; each tries every legal action
    unsigned threads = 1;      // worlds split over threads (same result for any count)
};

/**
 * Flat determinized search: in each of o.worlds worlds drawn with
 * determinize(), every legal action is applied and the game played to the end
 * with `playout` at every seat (the same random stream for each action of a
 * world, so the actions are compared on equal terms); the action with the best
 * mean payoff for the seat to move is played. A single legal action is played
 * at once. With a leaf, a play-out stops where the leaf values it (e.g. the
 * exact endgame of the last tricks). The playout agent, the leaf and the world
 * count are the knobs; a game with
 * its own search (Belot's) builds it from cardbot/search.h instead.
 */
/**
 * A play-out leaf: true and the value for `seat` in v when it can value the
 * position without playing on (an exact endgame, cardbot/endgame.h; a value
 * net), false to go on playing.
 */
template <class G>
using Leaf = std::function<bool(const G&, int seat, double& v)>;

/** g played on by `agents` until the end or until `leaf` values it; the value for `seat`. */
template <class G>
double playOutValue(G g, int seat, const std::vector<Agent<G>>& agents, const Leaf<G>& leaf, std::mt19937_64& rng) {
    double v = 0;
    while (!g.finished()) {
        if (leaf && leaf(g, seat, v)) return v;
        g.apply(agents[std::size_t(g.toMove())](g, rng));
    }
    return g.payoff(seat);
}

template <class G>
Agent<G> rolloutAgent(Agent<G> playout, RolloutOptions o = {}, Leaf<G> leaf = {}) {
    return [playout, o, leaf](const G& g, std::mt19937_64& rng) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const int seat = g.toMove();
        const int worlds = std::max(1, o.worlds);
        std::vector<std::uint64_t> seeds(static_cast<std::size_t>(worlds));
        for (auto& s : seeds) s = rng();
        const std::vector<Agent<G>> agents(std::size_t(G::kPlayers), playout);
        // per[w * n + k]: world w, action k
        std::vector<double> per(static_cast<std::size_t>(worlds) * legal.size());
        auto run = [&](int w) {
            std::mt19937_64 wr(seeds[std::size_t(w)]);
            const G world = g.determinize(seat, wr);
            const std::uint64_t play_seed = wr();
            for (std::size_t k = 0; k < legal.size(); ++k) {
                G next = world;
                next.apply(legal[k]);
                std::mt19937_64 pr(play_seed);
                per[std::size_t(w) * legal.size() + k] = playOutValue(std::move(next), seat, agents, leaf, pr);
            }
        };
        const int threads = int(std::max(1u, std::min(o.threads, unsigned(worlds))));
        if (threads <= 1) {
            for (int w = 0; w < worlds; ++w) run(w);
        } else {
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&, t] { for (int w = t; w < worlds; w += threads) run(w); });
            for (auto& th : pool) th.join();
        }
        std::size_t best = 0;
        double best_v = -1e300;
        for (std::size_t k = 0; k < legal.size(); ++k) {
            double v = 0.0;
            for (int w = 0; w < worlds; ++w) v += per[std::size_t(w) * legal.size() + k];
            if (v > best_v) { best_v = v; best = k; }
        }
        return legal[best];
    };
}

}  // namespace cardbot
