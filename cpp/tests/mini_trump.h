#pragma once
// The framework's toy game with an auction, for the bidding steps' tests:
// MiniTricks (16 cards, 4 each, follow suit) with trumps. From the first seat
// on, each seat passes or names a trump suit; the first to name one is the
// declarer, and the first seat leads. A trump wins the trick over any card of
// the led suit. The declarer's team needs 3 of the 4 tricks: it scores
// 2 * tricks - 5 (from -5 to +3), the others the opposite; four passes: 0.
#include "cardbot/game.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <random>
#include <vector>

namespace cardbot_test {

struct MiniTrump {
    using Action = int;  // 0..15 cards, 16 pass, 17 + s: name suit s trumps
    static constexpr int kPlayers = 4;
    static constexpr int kPass = 16, kName = 17;
    std::array<std::uint16_t, 4> hands{};
    std::array<std::uint8_t, 4> voids{};
    std::vector<std::pair<int, int>> trick, played;
    std::vector<int> auction;  // the auction's actions, in order
    int first = 0, to_move = 0, trump = -1, declarer = -1;
    bool bidding = true;
    std::array<int, 2> tricks{};

    static int suit(int c) { return c / 4; }
    static MiniTrump deal(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::array<int, 16> d{};
        for (int i = 0; i < 16; ++i) d[std::size_t(i)] = i;
        std::shuffle(d.begin(), d.end(), rng);
        MiniTrump g;
        for (int i = 0; i < 16; ++i) g.hands[std::size_t(i / 4)] |= std::uint16_t(1u << d[std::size_t(i)]);
        g.first = g.to_move = int(seed % 4);
        return g;
    }
    int toMove() const { return to_move; }
    bool passedOut() const { return !bidding && trump < 0; }
    bool finished() const { return passedOut() || tricks[0] + tricks[1] == 4; }
    int team(int seat) const { return seat % 2; }
    std::vector<int> legalActions() const {
        if (bidding) return {kPass, kName, kName + 1, kName + 2, kName + 3};
        std::vector<int> all, follow;
        for (int c = 0; c < 16; ++c)
            if (hands[std::size_t(to_move)] >> c & 1) {
                all.push_back(c);
                if (!trick.empty() && suit(c) == suit(trick[0].second)) follow.push_back(c);
            }
        return follow.empty() ? all : follow;
    }
    void apply(int a) {
        if (bidding) {
            auction.push_back(a);
            if (a == kPass) {
                if (auction.size() == 4) bidding = false;  // passed out
                else to_move = (to_move + 1) % 4;
                return;
            }
            trump = a - kName;
            declarer = to_move;
            bidding = false;
            to_move = first;
            return;
        }
        if (!trick.empty() && suit(a) != suit(trick[0].second)) voids[std::size_t(to_move)] |= std::uint8_t(1u << suit(trick[0].second));
        hands[std::size_t(to_move)] &= std::uint16_t(~(1u << a));
        trick.push_back({to_move, a});
        played.push_back({to_move, a});
        if (trick.size() < 4) { to_move = (to_move + 1) % 4; return; }
        auto beats = [&](int c, int best) {
            const bool ct = suit(c) == trump, bt = suit(best) == trump;
            if (ct != bt) return ct;
            if (ct) return c > best;
            return suit(c) == suit(trick[0].second) && c > best;
        };
        auto best = trick[0];
        for (auto& p : trick) if (beats(p.second, best.second)) best = p;
        ++tricks[std::size_t(team(best.first))];
        to_move = best.first;
        trick.clear();
    }
    double payoff(int seat) const {
        if (trump < 0) return 0.0;
        const int t = team(declarer);
        const double v = 2.0 * tricks[std::size_t(t)] - 5.0;
        return team(seat) == t ? v : -v;
    }
    MiniTrump determinize(int observer, std::mt19937_64& rng) const {
        std::vector<int> unseen;
        for (int s = 0; s < 4; ++s)
            if (s != observer) for (int c = 0; c < 16; ++c) if (hands[std::size_t(s)] >> c & 1) unseen.push_back(c);
        for (int attempt = 0; attempt < 1000; ++attempt) {
            std::shuffle(unseen.begin(), unseen.end(), rng);
            MiniTrump w = *this;
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

/** The step-2 bidder: name the longest suit (the highest on ties) when it has 3 or more cards. */
inline int baselineTrumpBid(const MiniTrump& g) {
    int best = -1, best_n = 3;
    for (int s = 0; s < 4; ++s) {
        const int n = int(std::bitset<16>(g.hands[std::size_t(g.to_move)] >> (4 * s) & 0xF).count());
        if (n >= best_n) { best_n = n; best = s; }
    }
    return best < 0 ? MiniTrump::kPass : MiniTrump::kName + best;
}

/** Card play for BatchEnv and the networks; the auction is played by autoAction (or the bid net). */
struct MiniTrumpEncoder {
    // own hand, played per relative seat, trick per relative seat, tricks, trump, declarer's relative seat
    static constexpr int kStateDim = 16 + 4 * 16 + 4 * 16 + 2 + 4 + 4;
    static constexpr int kPlyDim = 16 + 4;
    static constexpr int kMaxPlies = 16;
    static constexpr int kActionDim = 16;
    static constexpr int kMaxLegal = 4;
    static constexpr int kPolicyDim = 16;

    MiniTrump newGame(std::mt19937_64& rng) const { return MiniTrump::deal(rng()); }
    bool learns(const MiniTrump& g) const { return !g.bidding; }
    int autoAction(const MiniTrump& g, std::mt19937_64&) const { return baselineTrumpBid(g); }
    int encode(const MiniTrump& g, int seat, float* state, float* history) const {
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
        if (g.trump >= 0) {
            state[146 + g.trump] = 1.0f;
            state[150 + (g.declarer - seat + 4) % 4] = 1.0f;
        }
        return int(g.played.size());
    }
    void encodeAction(const MiniTrump&, int, int card, float* out) const { out[card] = 1.0f; }
    int actionSlot(const MiniTrump&, int, int card) const { return card; }
    float reward(const MiniTrump& g, int team) const { return float(g.payoff(team)) / 5.0f; }
};

/** The auction for BidEnv and the bid net. */
struct MiniTrumpBidEncoder {
    static constexpr int kFeatDim = 16 + 4;   // own hand, own position in the auction
    static constexpr int kHistDim = 5 + 4;    // the action (pass, a suit), its relative seat
    static constexpr int kMaxHist = 4;
    static constexpr int kNumBids = 5;        // pass, suit 0..3

    MiniTrump newGame(std::mt19937_64& rng) const { return MiniTrump::deal(rng()); }
    bool isBid(const MiniTrump& g) const { return g.bidding; }
    int bidSlot(const MiniTrump&, int, int a) const { return a - MiniTrump::kPass; }
    int encode(const MiniTrump& g, int seat, float* feat, float* hist) const {
        for (int c = 0; c < 16; ++c)
            if (g.hands[std::size_t(seat)] >> c & 1) feat[c] = 1.0f;
        feat[16 + g.auction.size()] = 1.0f;
        for (std::size_t i = 0; i < g.auction.size(); ++i) {
            const int who = (g.first + int(i)) % 4;
            hist[i * kHistDim + std::size_t(g.auction[i] - MiniTrump::kPass)] = 1.0f;
            hist[i * kHistDim + 5 + std::size_t((who - seat + 4) % 4)] = 1.0f;
        }
        return int(g.auction.size());
    }
    float reward(const MiniTrump& g, int team) const { return float(g.payoff(team)) / 5.0f; }
    int baselineBid(const MiniTrump& g, std::mt19937_64&) const { return baselineTrumpBid(g); }
};

}  // namespace cardbot_test
