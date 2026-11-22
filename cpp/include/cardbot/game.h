#pragma once
// The game a cardbot agent plays (docs/DESIGN.md, step 1). Not a base class:
// any copyable type with these members works, so a game keeps its own
// representation and nothing costs a virtual call.
//
//   struct MyGame {
//       using Action = int;                      // cards, bids, declarations... as the game numbers them
//       static constexpr int kPlayers = 4;
//       int toMove() const;                      // the seat to act
//       bool finished() const;
//       std::vector<Action> legalActions() const;
//       void apply(Action a);
//       double payoff(int seat) const;           // when finished: the seat's result (e.g. its team's margin)
//       int team(int seat) const;                // partnerships: seats with the same team share results
//       // The same position with what `observer` cannot see redrawn consistently
//       // with everything that seat knows (a determinization).
//       MyGame determinize(int observer, std::mt19937_64& rng) const;
//   };
//
// Agents get the whole game but must use only what the seat to move knows:
// the ones here do, through legalActions() and determinize().

#include <functional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace cardbot {

template <class G>
using ActionOf = typename G::Action;

/** An agent: the action for the seat to move. Any callable with this shape. */
template <class G>
using Agent = std::function<ActionOf<G>(const G&, std::mt19937_64&)>;

/** Plays g to the end with agents[seat] for every seat; returns the finished game. */
template <class G>
G playOut(G g, const std::vector<Agent<G>>& agents, std::mt19937_64& rng) {
    while (!g.finished()) g.apply(agents[std::size_t(g.toMove())](g, rng));
    return g;
}

}  // namespace cardbot
