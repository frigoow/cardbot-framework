#pragma once
// The exact endgame (docs/DESIGN.md, step 3), as the Belote bot solves its
// last tricks: alpha-beta over a position with every hand visible (a
// determinized world), the seat's team maximizing its payoff and the other
// team minimizing it, with a transposition table and move ordering.
//
// Everything game-specific is an optional hook. Derive from EndgameHooks<G>
// and redefine what helps (the solver calls them on your type, no virtual
// calls):
//
//   struct MyHooks : cardbot::EndgameHooks<MyGame> {
//       bool cacheable(const MyGame&) const;          // positions worth a table entry (e.g. trick starts)
//       std::uint64_t key(const MyGame&) const;       // their hash: everything the rest of the game depends on
//       int priority(const MyGame&, Action) const;    // move ordering: higher first (good moves cut early)
//       std::vector<Action> moves(const MyGame&) const;   // legal moves minus equivalent ones (inside the tree)
//       double value(const MyGame&, int seat) const;  // a finished game's value (default: payoff)
//   };
//
// Without hooks it is a plain alpha-beta; values are the same either way.

#include "cardbot/game.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cardbot {

template <class G>
struct EndgameHooks {
    using Action = ActionOf<G>;
    bool cacheable(const G&) const { return false; }
    std::uint64_t key(const G&) const { return 0; }
    int priority(const G&, const Action&) const { return 0; }
    std::vector<Action> moves(const G& g) const { return g.legalActions(); }
    double value(const G& g, int seat) const { return g.payoff(seat); }
};

struct EndgameOptions {
    int tt_log2 = 16;   // transposition-table entries: 2^tt_log2 (when the hooks cache anything)
};

template <class G, class Hooks = EndgameHooks<G>>
class EndgameSolver {
public:
    using Action = ActionOf<G>;

    struct Result {
        double value = 0;                               // the position's exact value for the seat
        std::vector<std::pair<Action, double>> moves;   // every legal move of the seat to move, with its value
        long nodes = 0;
    };

    explicit EndgameSolver(Hooks hooks = Hooks(), EndgameOptions o = EndgameOptions())
        : hooks_(std::move(hooks)), tt_(std::size_t(1) << std::max(4, std::min(26, o.tt_log2))) {}

    /** The exact value of g for `seat` (its payoff under best play by both teams). */
    double value(const G& g, int seat) {
        begin(seat);
        return g.finished() ? hooks_.value(g, seat) : search(g, -kInf, kInf);
    }

    /** value() and the value of each legal move of the seat to move (each searched with a full window). */
    Result solve(const G& g, int seat) {
        begin(seat);
        Result r;
        if (g.finished()) {
            r.value = hooks_.value(g, seat);
            return r;
        }
        const bool maxing = g.team(g.toMove()) == g.team(seat);
        r.value = maxing ? -kInf : kInf;
        for (const Action& a : g.legalActions()) {
            G n = g;
            n.apply(a);
            const double v = n.finished() ? hooks_.value(n, seat) : search(n, -kInf, kInf);
            r.moves.push_back({a, v});
            r.value = maxing ? std::max(r.value, v) : std::min(r.value, v);
        }
        r.nodes = nodes_;
        return r;
    }

    long nodes() const { return nodes_; }
    const Hooks& hooks() const { return hooks_; }
    void setHooks(Hooks h) { hooks_ = std::move(h); }
    std::size_t tableSize() const { return tt_.size(); }

private:
    static constexpr double kInf = std::numeric_limits<double>::infinity();
    enum class Bound : std::uint8_t { kNone, kExact, kLower, kUpper };
    struct Entry {
        std::uint64_t key = 0;
        std::uint32_t gen = 0;
        Bound bound = Bound::kNone;
        double score = 0;
        int best = -1;   // index into hooks.moves() of the best move, for ordering
    };

    void begin(int seat) {
        seat_ = seat;
        nodes_ = 0;
        if (++gen_ == 0) {
            std::fill(tt_.begin(), tt_.end(), Entry{});
            gen_ = 1;
        }
    }

    double search(const G& g, double alpha, double beta) {
        ++nodes_;
        const bool maxing = g.team(g.toMove()) == g.team(seat_);
        const bool cache = hooks_.cacheable(g);
        std::uint64_t key = 0;
        int tt_best = -1;
        Entry* e = nullptr;
        if (cache) {
            key = hooks_.key(g);
            e = &tt_[key & (tt_.size() - 1)];
            if (e->gen == gen_ && e->bound != Bound::kNone && e->key == key) {
                tt_best = e->best;
                if (e->bound == Bound::kExact) return e->score;
                if (e->bound == Bound::kLower) alpha = std::max(alpha, e->score);
                else beta = std::min(beta, e->score);
                if (alpha >= beta) return e->score;
            }
        }
        const std::vector<Action> moves = hooks_.moves(g);
        std::vector<int> order(moves.size());
        std::vector<int> prio(moves.size());
        for (std::size_t i = 0; i < moves.size(); ++i) {
            order[i] = int(i);
            prio[i] = int(i) == tt_best ? std::numeric_limits<int>::max() : hooks_.priority(g, moves[i]);
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prio[std::size_t(a)] > prio[std::size_t(b)]; });

        const double alpha0 = alpha, beta0 = beta;
        double best = maxing ? -kInf : kInf;
        int best_i = order.empty() ? -1 : order[0];
        for (int i : order) {
            G n = g;
            n.apply(moves[std::size_t(i)]);
            const double v = n.finished() ? hooks_.value(n, seat_) : search(n, alpha, beta);
            if (maxing ? v > best : v < best) {
                best = v;
                best_i = i;
            }
            if (maxing) alpha = std::max(alpha, v);
            else beta = std::min(beta, v);
            if (alpha >= beta) break;
        }
        if (cache) {
            e->key = key;
            e->gen = gen_;
            e->score = best;
            e->best = best_i;
            e->bound = best <= alpha0 ? Bound::kUpper : best >= beta0 ? Bound::kLower : Bound::kExact;
        }
        return best;
    }

    Hooks hooks_;
    std::vector<Entry> tt_;
    std::uint32_t gen_ = 0;
    int seat_ = 0;
    long nodes_ = 0;
};

/**
 * A play-out leaf (rolloutAgent's) that solves the rest exactly once
 * from(g) holds, e.g. in the last tricks: the value for `seat` with every hand
 * of the world visible. One solver per thread.
 */
template <class G, class Hooks = EndgameHooks<G>>
std::function<bool(const G&, int, double&)> endgameLeaf(std::function<bool(const G&)> from, Hooks hooks = Hooks(),
                                                        EndgameOptions o = EndgameOptions()) {
    return [from, hooks, o](const G& g, int seat, double& v) {
        if (!from(g)) return false;
        // the thread's solver keeps its table between calls; it takes this leaf's hooks every time
        thread_local std::unique_ptr<EndgameSolver<G, Hooks>> solver;
        const std::size_t size = std::size_t(1) << std::max(4, std::min(26, o.tt_log2));
        if (!solver || solver->tableSize() != size) solver = std::make_unique<EndgameSolver<G, Hooks>>(hooks, o);
        else solver->setHooks(hooks);
        v = solver->value(g, seat);
        return true;
    };
}

}  // namespace cardbot
