// endgame.h on the toy games: exact values against brute-force minimax, the
// same with and without the hooks (table, ordering, equivalent moves), and the
// rollout search with the exact endgame as its leaf.
#include "cardbot/agents.h"
#include "cardbot/endgame.h"
#include "cardbot/eval.h"
#include "mini_tricks.h"
#include "mini_trump.h"

#include <cmath>
#include <cstdio>
#include <random>

namespace {

int failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

using G = cardbot_test::MiniTricks;

template <class Game>
double bruteForce(const Game& g, int seat) {
    if (g.finished()) return g.payoff(seat);
    const bool maxing = g.team(g.toMove()) == g.team(seat);
    double best = maxing ? -1e300 : 1e300;
    for (int a : g.legalActions()) {
        Game n = g;
        n.apply(a);
        const double v = bruteForce(n, seat);
        best = maxing ? std::max(best, v) : std::min(best, v);
    }
    return best;
}

// The toy game's hooks: a table at trick starts, high cards first, and of two
// own cards of adjacent ranks in a suit (nothing can fall between them) only
// the lower is tried.
struct TrickHooks : cardbot::EndgameHooks<G> {
    bool cacheable(const G& g) const { return g.trick.empty(); }
    std::uint64_t key(const G& g) const {
        std::uint64_t k = 0;
        for (int s = 0; s < 4; ++s) k |= std::uint64_t(g.hands[std::size_t(s)]) << (16 * s);
        k ^= (std::uint64_t(g.to_move) * 8 + std::uint64_t(g.tricks[0])) * 0x9E3779B97F4A7C15ull;
        return k;
    }
    int priority(const G&, int c) const { return c % 4; }
    std::vector<int> moves(const G& g) const {
        const auto legal = g.legalActions();
        const std::uint16_t mine = g.hands[std::size_t(g.to_move)];
        std::vector<int> out;
        for (int c : legal) {
            const bool twin = c % 4 != 0 && (mine >> (c - 1) & 1);  // the card just below is mine: same effect
            if (!twin) out.push_back(c);
        }
        return out;
    }
};

}  // namespace

int main() {
    std::mt19937_64 rng(11);
    cardbot::EndgameSolver<G> plain;
    cardbot::EndgameSolver<G, TrickHooks> hooked;
    long plain_nodes = 0, hooked_nodes = 0;
    for (int i = 0; i < 300; ++i) {
        G g = G::deal(rng());
        const int skip = int(rng() % 10);
        for (int k = 0; k < skip; ++k) {
            const auto legal = g.legalActions();
            g.apply(legal[rng() % legal.size()]);
        }
        const int seat = int(rng() % 4);
        const double bf = bruteForce(g, seat);
        CHECK(plain.value(g, seat) == bf);
        plain_nodes += plain.nodes();
        CHECK(hooked.value(g, seat) == bf);
        hooked_nodes += hooked.nodes();
        const auto r = hooked.solve(g, g.toMove());
        for (const auto& [a, v] : r.moves) {
            G n = g;
            n.apply(a);
            CHECK(v == bruteForce(n, g.toMove()));
        }
    }
    std::printf("exact endgame: 300 positions match brute force; nodes %ld plain, %ld with the hooks\n", plain_nodes,
                hooked_nodes);
    CHECK(hooked_nodes < plain_nodes);
    // the game with trumps, default hooks
    cardbot::EndgameSolver<cardbot_test::MiniTrump> trump_solver;
    for (int i = 0; i < 100; ++i) {
        auto g = cardbot_test::MiniTrump::deal(rng());
        g.apply(cardbot_test::MiniTrump::kName + int(rng() % 4));
        for (int k = int(rng() % 6); k > 0; --k) {
            const auto legal = g.legalActions();
            g.apply(legal[rng() % legal.size()]);
        }
        CHECK(trump_solver.value(g, 0) == bruteForce(g, 0));
    }

    // the rollout search with the exact rest as its leaf: double-dummy values of each move per world
    const std::function<G(std::uint64_t)> deal = [](std::uint64_t s) { return G::deal(s); };
    cardbot::RolloutOptions ro;
    ro.worlds = 16;
    const auto exact = cardbot::endgameLeaf<G, TrickHooks>([](const G&) { return true; });
    const auto dd = cardbot::rolloutAgent<G>(cardbot::randomAgent<G>(), ro, exact);
    const auto r1 = cardbot::duel<G>(deal, dd, cardbot::rolloutAgent<G>(cardbot::randomAgent<G>(), ro), 400, 9, 4);
    std::printf("rollout-16 exact vs random play-outs: %+.3f +- %.3f tricks per deal\n", r1.mean, r1.ci95);
    cardbot::IsmctsOptions io;
    io.iterations = 300;
    io.scale = 0.25;
    const auto r2 = cardbot::duel<G>(deal, dd, cardbot::ismctsAgent<G>(io), 400, 9, 4);
    std::printf("rollout-16 exact vs ISMCTS-300: %+.3f +- %.3f tricks per deal\n", r2.mean, r2.ci95);
    CHECK(r1.mean + r1.ci95 > 0);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardbot endgame: all checks passed\n");
    return 0;
}
