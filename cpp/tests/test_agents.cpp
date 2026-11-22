// game.h, agents.h and eval.h on a toy trick game: 4 players in two teams, 16
// cards (4 suits x 4 ranks), 4 each, follow suit, the highest card of the led
// suit wins, the winner leads; payoff = own team's tricks minus the others'.
#include "cardbot/agents.h"
#include "cardbot/eval.h"
#include "cardbot/game.h"

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

struct MiniTricks {
    using Action = int;
    static constexpr int kPlayers = 4;
    std::array<std::uint16_t, 4> hands{};
    std::array<std::uint8_t, 4> voids{};  // bit s: the seat showed out of suit s
    std::vector<std::pair<int, int>> trick;  // (seat, card)
    int leader = 0, to_move = 0;
    std::array<int, 2> tricks{};

    static int suit(int c) { return c / 4; }
    static MiniTricks deal(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::array<int, 16> d{};
        for (int i = 0; i < 16; ++i) d[std::size_t(i)] = i;
        std::shuffle(d.begin(), d.end(), rng);
        MiniTricks g;
        for (int i = 0; i < 16; ++i) g.hands[std::size_t(i / 4)] |= std::uint16_t(1u << d[std::size_t(i)]);
        g.leader = g.to_move = int(seed % 4);
        return g;
    }
    int toMove() const { return to_move; }
    bool finished() const { return tricks[0] + tricks[1] == 4; }
    int team(int seat) const { return seat % 2; }
    std::vector<int> legalActions() const {
        std::vector<int> all, follow;
        for (int c = 0; c < 16; ++c)
            if (hands[std::size_t(to_move)] >> c & 1) {
                all.push_back(c);
                if (!trick.empty() && suit(c) == suit(trick[0].second)) follow.push_back(c);
            }
        return follow.empty() ? all : follow;
    }
    void apply(int c) {
        if (!trick.empty() && suit(c) != suit(trick[0].second)) voids[std::size_t(to_move)] |= std::uint8_t(1u << suit(trick[0].second));
        hands[std::size_t(to_move)] &= std::uint16_t(~(1u << c));
        trick.push_back({to_move, c});
        if (trick.size() < 4) { to_move = (to_move + 1) % 4; return; }
        auto best = trick[0];
        for (auto& p : trick) if (suit(p.second) == suit(trick[0].second) && p.second > best.second) best = p;
        ++tricks[std::size_t(team(best.first))];
        leader = to_move = best.first;
        trick.clear();
    }
    double payoff(int seat) const { return tricks[std::size_t(team(seat))] - tricks[std::size_t(1 - team(seat))]; }
    MiniTricks determinize(int observer, std::mt19937_64& rng) const {
        std::vector<int> unseen;
        for (int s = 0; s < 4; ++s)
            if (s != observer) for (int c = 0; c < 16; ++c) if (hands[std::size_t(s)] >> c & 1) unseen.push_back(c);
        for (int attempt = 0; attempt < 1000; ++attempt) {
            std::shuffle(unseen.begin(), unseen.end(), rng);
            MiniTricks w = *this;
            std::size_t k = 0;
            bool ok = true;
            for (int s = 0; s < 4 && ok; ++s) {
                if (s == observer) continue;
                const int n = int(std::bitset<16>(hands[std::size_t(s)]).count());
                w.hands[std::size_t(s)] = 0;
                for (int i = 0; i < n; ++i) {
                    const int c = unseen[k++];
                    if (voids[std::size_t(s)] >> suit(c) & 1) ok = false;
                    w.hands[std::size_t(s)] |= std::uint16_t(1u << c);
                }
            }
            if (ok || attempt == 999) return w;
        }
        return *this;
    }
};

using G = MiniTricks;

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
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardbot agents: all checks passed\n");
    return 0;
}
