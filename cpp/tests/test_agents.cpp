// game.h, agents.h and eval.h on a toy trick game: 4 players in two teams, 16
// cards (4 suits x 4 ranks), 4 each, follow suit, the highest card of the led
// suit wins, the winner leads; payoff = own team's tricks minus the others'.
#include "cardbot/agents.h"
#include "cardbot/eval.h"
#include "cardbot/game.h"
#include "mini_tricks.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

using G = cardbot_test::MiniTricks;

}  // namespace

int main() {
    const std::function<G(std::uint64_t)> deal = [](std::uint64_t s) { return G::deal(s); };
    // random against random: even, and reproducible whatever the thread count
    const auto r1 = cardbot::duel<G>(deal, cardbot::randomAgent<G>(), cardbot::randomAgent<G>(), 400, 7, 1);
    const auto r2 = cardbot::duel<G>(deal, cardbot::randomAgent<G>(), cardbot::randomAgent<G>(), 400, 7, 4);
    CHECK(r1.mean == r2.mean && r1.ci95 == r2.ci95);
    CHECK(std::fabs(r1.mean) < 3 * r1.ci95 + 1e-9);
    std::printf("random vs random: %+.3f +- %.3f tricks per deal\n", r1.mean, r1.ci95);
    // ISMCTS beats random clearly
    cardbot::IsmctsOptions o;
    o.iterations = 300;
    o.scale = 0.25;
    const auto r3 = cardbot::duel<G>(deal, cardbot::ismctsAgent<G>(o), cardbot::randomAgent<G>(), 150, 3, 4);
    std::printf("ISMCTS-300 vs random: %+.3f +- %.3f tricks per deal (better on %.0f%% of deals)\n", r3.mean, r3.ci95,
                100 * r3.a_better);
    CHECK(r3.mean - r3.ci95 > 0);
    // flat play-out search with random play-outs: beats random, same result for any thread count
    cardbot::RolloutOptions ro;
    ro.worlds = 16;
    const auto r4 = cardbot::duel<G>(deal, cardbot::rolloutAgent<G>(cardbot::randomAgent<G>(), ro), cardbot::randomAgent<G>(), 150, 5, 4);
    ro.threads = 3;
    const auto r5 = cardbot::duel<G>(deal, cardbot::rolloutAgent<G>(cardbot::randomAgent<G>(), ro), cardbot::randomAgent<G>(), 150, 5, 1);
    std::printf("rollout-16 vs random: %+.3f +- %.3f tricks per deal\n", r4.mean, r4.ci95);
    CHECK(r4.mean - r4.ci95 > 0);
    CHECK(r4.mean == r5.mean);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardbot agents: all checks passed\n");
    return 0;
}
