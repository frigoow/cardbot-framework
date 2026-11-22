#pragma once
// Baseline agents for any game (docs/DESIGN.md, step 2): uniform random, and
// ISMCTS (single-observer information-set Monte Carlo tree search, through the
// bundled ismcsolver, MIT, Steven Franzen). ISMCTS needs only the game's
// determinize(); the result it maximizes is the game's payoff, scaled by
// IsmctsOptions::scale into the range its exploration constant expects.

#include "cardbot/game.h"

#include <ismcts/game.h>
#include <ismcts/sosolver.h>

#include <memory>
#include <random>
#include <stdexcept>
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

}  // namespace cardbot
