#pragma once
// The framework's toy game for its tests: 4 players in two teams, 16 cards
// (4 suits x 4 ranks), 4 each, follow suit, the highest card of the led suit
// wins, the winner leads; payoff = own team's tricks minus the others'. And its
// encoder for the batched environment and the networks.
#include "cardbot/game.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <random>
#include <vector>

namespace cardbot_test {

struct MiniTricks {
    using Action = int;
    static constexpr int kPlayers = 4;
    std::array<std::uint16_t, 4> hands{};
    std::array<std::uint8_t, 4> voids{};  // bit s: the seat showed out of suit s
    std::vector<std::pair<int, int>> trick;  // (seat, card)
    std::vector<std::pair<int, int>> played;  // every card so far, in order
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
        played.push_back({to_move, c});
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

/** MiniTricks for BatchEnv and the networks: what the seat to move knows. */
struct MiniEncoder {
    static constexpr int kStateDim = 16 + 4 * 16 + 4 * 16 + 2;  // own hand, played per seat, trick per seat, tricks
    static constexpr int kPlyDim = 16 + 4;                      // card, relative seat
    static constexpr int kMaxPlies = 16;
    static constexpr int kActionDim = 16;
    static constexpr int kMaxLegal = 4;
    static constexpr int kPolicyDim = 16;                       // the fast policy scores cards

    MiniTricks newGame(std::mt19937_64& rng) const { return MiniTricks::deal(rng()); }
    bool learns(const MiniTricks&) const { return true; }
    int autoAction(const MiniTricks& g, std::mt19937_64&) const { return g.legalActions()[0]; }
    int encode(const MiniTricks& g, int seat, float* state, float* history) const {
        for (int c = 0; c < 16; ++c)
            if (g.hands[std::size_t(seat)] >> c & 1) state[c] = 1.0f;
        for (std::size_t i = 0; i < g.played.size(); ++i) {
            const int rel = (g.played[i].first - seat + 4) % 4;
            state[16 + rel * 16 + g.played[i].second] = 1.0f;
            history[i * kPlyDim + std::size_t(g.played[i].second)] = 1.0f;
            history[i * kPlyDim + 16 + std::size_t(rel)] = 1.0f;
        }
        for (const auto& [s, c] : g.trick) state[80 + ((s - seat + 4) % 4) * 16 + c] = 1.0f;
        state[144] = float(g.tricks[std::size_t(g.team(seat))]) / 4.0f;
        state[145] = float(g.tricks[std::size_t(1 - g.team(seat))]) / 4.0f;
        return int(g.played.size());
    }
    void encodeAction(const MiniTricks&, int, int card, float* out) const { out[card] = 1.0f; }
    int actionSlot(const MiniTricks&, int, int card) const { return card; }
    float reward(const MiniTricks& g, int team) const { return float(g.payoff(team)) / 4.0f; }
};

}  // namespace cardbot_test
