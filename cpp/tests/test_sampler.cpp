// cardbot/sampler.h on small random deals: the count matches the enumeration,
// the marginals match the enumerated frequencies, samples respect the domains
// and hand sizes, and a placement mixture counts as the sum of its parts.
#include "cardbot/sampler.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <random>

using Deal = cardbot::HiddenDeal<8>;

namespace {

int failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

int popcount(std::uint64_t m) { return int(std::bitset<64>(m).count()); }

Deal randomDeal(std::mt19937_64& rng) {
    // 3 hidden seats holding 2..5 cards each, every card allowed to 1..3 seats
    std::uniform_int_distribution<int> size(2, 5);
    const std::array<int, 3> need{size(rng), size(rng), size(rng)};
    const int n = need[0] + need[1] + need[2];
    std::vector<int> cards;
    Deal::Domain dom;
    // deal a real hand first so at least one deal exists, then widen the domains
    std::vector<int> owner;
    for (int k = 0; k < 3; ++k) for (int i = 0; i < need[std::size_t(k)]; ++i) owner.push_back(k);
    std::shuffle(owner.begin(), owner.end(), rng);
    for (int i = 0; i < n; ++i) {
        cards.push_back(2 * i + 1);
        std::uint8_t d = std::uint8_t(1u << owner[std::size_t(i)]);
        for (int k = 0; k < 3; ++k) if (rng() % 2) d |= std::uint8_t(1u << k);
        dom.push_back(d);
    }
    return Deal(cards, dom, need);
}

void testCountMarginalsSamples() {
    std::mt19937_64 rng(5);
    for (int trial = 0; trial < 40; ++trial) {
        const Deal d = randomDeal(rng);
        const std::size_t n = d.cards().size();
        std::vector<std::array<double, 3>> freq(n, {0, 0, 0});
        double worlds = 0;
        d.enumerate([&](const Deal::Hands& h) {
            worlds += 1;
            for (int k = 0; k < 3; ++k) CHECK(popcount(h[std::size_t(k)]) == d.need()[std::size_t(k)]);
            for (std::size_t i = 0; i < n; ++i)
                for (int k = 0; k < 3; ++k)
                    if (h[std::size_t(k)] >> d.cards()[i] & 1) {
                        CHECK(d.domain()[i] >> k & 1);
                        freq[i][std::size_t(k)] += 1;
                    }
            return true;
        });
        CHECK(worlds == d.count());
        const auto marg = d.marginals();
        for (std::size_t i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) CHECK(std::fabs(marg[i][std::size_t(k)] - freq[i][std::size_t(k)] / worlds) < 1e-9);
        for (int s = 0; s < 20; ++s) {
            const auto h = d.sampleUnchecked(rng);
            CHECK(h.has_value());
            for (int k = 0; k < 3; ++k) CHECK(popcount((*h)[std::size_t(k)]) == d.need()[std::size_t(k)]);
        }
    }
}

void testPlacements() {
    std::mt19937_64 rng(9);
    Deal d = randomDeal(rng);
    // two placements: the first card forced to seat 0, or to seat 1
    Deal::Domain a = d.domain(), b = d.domain();
    a[0] = 1; b[0] = 2;
    const double ca = d.countFor(a), cb = d.countFor(b);
    if (ca > 0) d.addPlacement(a, ca);
    if (cb > 0) d.addPlacement(b, cb);
    CHECK(d.count() == ca + cb);
    const auto marg = d.marginals();
    if (ca + cb > 0) CHECK(std::fabs(marg[0][2]) < 1e-12);  // never with seat 2
}

void testUniformity() {
    // two hidden seats of 2 from 4 free cards: 6 deals, each about 1/6 of the draws
    Deal d({1, 3, 5, 7}, {3, 3, 3, 3}, {2, 2, 0});
    CHECK(d.twoSeat());
    CHECK(d.count() == 6);
    std::mt19937_64 rng(1);
    std::array<int, 16> seen{};
    const int draws = 60000;
    for (int i = 0; i < draws; ++i) {
        const auto h = d.sampleUnchecked(rng);
        int key = 0;
        for (int c = 0; c < 4; ++c) if ((*h)[0] >> (2 * c + 1) & 1) key |= 1 << c;
        ++seen[std::size_t(key)];
    }
    for (int key = 0; key < 16; ++key)
        if (popcount(std::uint64_t(key)) == 2) CHECK(std::abs(seen[std::size_t(key)] - draws / 6) < draws / 60);
}

void testTwoSeat52Card() {
    // 52-card Bridge post-dummy case: 2 hidden seats of up to 13 cards, plus 3-seat opening lead
    using Deal52 = cardbot::HiddenDeal<13>;
    std::mt19937_64 rng(42);
    for (int trial = 0; trial < 30; ++trial) {
        const int n0 = 3 + int(rng() % 5);
        const int n1 = 3 + int(rng() % 5);
        const int n = n0 + n1;
        std::vector<int> owner(std::size_t(n), 0);
        for (int i = n0; i < n; ++i) owner[std::size_t(i)] = 1;
        std::shuffle(owner.begin(), owner.end(), rng);
        std::vector<int> cards;
        Deal52::Domain dom;
        Deal52::Weights wts;
        for (int i = 0; i < n; ++i) {
            cards.push_back(51 - i * 2);  // exercises high bits > 31
            std::uint8_t mask = std::uint8_t(1u << owner[std::size_t(i)]);
            if (rng() % 3 != 0) mask |= 3u;
            dom.push_back(mask);
            wts.push_back({0.5 + double(rng() % 10) / 10.0, 0.5 + double(rng() % 10) / 10.0, 0.0});
        }
        Deal52 d(cards, dom, {n0, n1, 0});
        CHECK(d.twoSeat());
        // Check against full 3-seat buildWays
        const auto w3 = Deal52::buildWays(dom);
        CHECK(std::fabs(d.count() - Deal52::waysAt(w3, 0, n0, n1)) < 1e-9);

        std::vector<std::array<double, 3>> freq(std::size_t(n), {0, 0, 0});
        double worlds = 0;
        d.enumerate([&](const Deal52::Hands& h) {
            worlds += 1;
            CHECK(popcount(h[0]) == n0 && popcount(h[1]) == n1 && h[2] == 0);
            for (std::size_t i = 0; i < std::size_t(n); ++i) {
                for (int k = 0; k < 2; ++k) {
                    if (h[std::size_t(k)] >> cards[i] & 1) {
                        CHECK(dom[i] >> k & 1);
                        freq[i][std::size_t(k)] += 1;
                    }
                }
            }
            return true;
        });
        CHECK(worlds == d.count());
        const auto marg = d.marginals();
        for (std::size_t i = 0; i < std::size_t(n); ++i) {
            CHECK(std::fabs(marg[i][0] - freq[i][0] / worlds) < 1e-9);
            CHECK(std::fabs(marg[i][1] - freq[i][1] / worlds) < 1e-9);
            CHECK(marg[i][2] == 0.0);
        }
        for (int s = 0; s < 10; ++s) {
            const auto hu = d.sampleUnchecked(rng);
            const auto hw = d.sampleWeighted(rng, wts, {}, 4);
            CHECK(hu.has_value() && hw.has_value());
            CHECK(popcount((*hu)[0]) == n0 && popcount((*hu)[1]) == n1 && (*hu)[2] == 0);
            CHECK(popcount((*hw)[0]) == n0 && popcount((*hw)[1]) == n1 && (*hw)[2] == 0);
        }
    }
    // Full 26-card 2-seat Bridge hand (13 + 13): C(26, 13) = 10,400,600
    std::vector<int> c26(26);
    for (int i = 0; i < 26; ++i) c26[std::size_t(i)] = 26 + i;
    Deal52 full2(c26, Deal52::Domain(26, 3), {13, 13, 0});
    CHECK(std::fabs(full2.count() - 10400600.0) < 1e-6);
    const auto h26 = full2.sampleUnchecked(rng);
    CHECK(h26.has_value() && popcount((*h26)[0]) == 13 && popcount((*h26)[1]) == 13 && (*h26)[2] == 0);
    // Full 39-card 3-seat Bridge opening lead (13 + 13 + 13): C(39,13)*C(26,13) = 84,478,098,072,866,400
    std::vector<int> c39(39);
    for (int i = 0; i < 39; ++i) c39[std::size_t(i)] = 13 + i;
    Deal52 full3(c39, Deal52::Domain(39, 7), {13, 13, 13});
    CHECK(!full3.twoSeat());
    CHECK(std::fabs(full3.count() - 84478098072866400.0) / 84478098072866400.0 < 1e-12);
    const auto h39 = full3.sampleUnchecked(rng);
    CHECK(h39.has_value() && popcount((*h39)[0]) == 13 && popcount((*h39)[1]) == 13 && popcount((*h39)[2]) == 13);
}

}  // namespace

int main() {
    testCountMarginalsSamples();
    testPlacements();
    testUniformity();
    testTwoSeat52Card();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardbot sampler: all checks passed\n");
    return 0;
}
