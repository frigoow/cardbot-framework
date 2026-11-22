#pragma once
// Duplicate-deal evaluation (docs/DESIGN.md: evaluation at every step). Each
// deal is played twice with the same cards, the teams swapping agents, so the
// comparison is how two agents handled the same cards; the paired differences
// give the mean and its 95% interval.

#include "cardbot/game.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace cardbot {

struct DuelResult {
    int pairs = 0;
    double mean = 0;      // A's payoff per deal, averaged over both tables of a pair
    double ci95 = 0;      // half-width of the 95% interval of the mean
    double a_better = 0;  // share of pairs where A did better than B on the same cards
};

/**
 * A against B on `pairs` deals. deal(seed) makes a fresh game; the team of
 * seat 0 plays A at the first table and B at the second. The payoff counted is
 * that of a seat of A's team. Deals run on `threads` threads, each deal with
 * its own seeded generator, so a run is reproducible for a given seed.
 */
template <class G>
DuelResult duel(const std::function<G(std::uint64_t)>& deal, const Agent<G>& a, const Agent<G>& b, int pairs,
                std::uint64_t seed = 1, unsigned threads = 1) {
    std::vector<double> diff(std::size_t(pairs), 0.0);
    std::atomic<int> next{0};
    auto work = [&]() {
        for (int i; (i = next++) < pairs;) {
            const std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + std::uint64_t(i);
            double sum = 0;
            for (int table = 0; table < 2; ++table) {
                G g = deal(s);
                const int a_team = table == 0 ? g.team(0) : 1 - g.team(0);
                std::vector<Agent<G>> agents;
                int a_seat = -1;
                for (int seat = 0; seat < G::kPlayers; ++seat) {
                    const bool is_a = g.team(seat) == a_team;
                    agents.push_back(is_a ? a : b);
                    if (is_a && a_seat < 0) a_seat = seat;
                }
                std::mt19937_64 rng(s ^ (0xD1B54A32D192ED03ull * std::uint64_t(table + 1)));
                const G end = playOut(std::move(g), agents, rng);
                sum += end.payoff(a_seat);
            }
            diff[std::size_t(i)] = sum / 2.0;
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    DuelResult r;
    r.pairs = pairs;
    if (pairs == 0) return r;
    double s1 = 0, s2 = 0;
    int better = 0;
    for (double d : diff) {
        s1 += d;
        s2 += d * d;
        if (d > 0) ++better;
    }
    r.mean = s1 / pairs;
    const double var = pairs > 1 ? (s2 - s1 * s1 / pairs) / (pairs - 1) : 0;
    r.ci95 = 1.96 * std::sqrt(std::max(var, 0.0) / pairs);
    r.a_better = double(better) / pairs;
    return r;
}

}  // namespace cardbot
