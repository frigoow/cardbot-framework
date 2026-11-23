// bidding.h on the toy game with an auction (mini_trump.h): the counterfactual
// targets, their reproducibility, and duplicate-deal bidder evaluation.
#include "cardbot/bidding.h"
#include "mini_trump.h"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

using G = cardbot_test::MiniTrump;
using BE = cardbot_test::MiniTrumpBidEncoder;

}  // namespace

int main() {
    cardbot::BidEnv<G, BE> env1(1), env4(4);
    const auto a = env1.generate(200, 3, 0.2, 2);
    const auto b = env4.generate(200, 3, 0.2, 2);
    CHECK(a.n > 200);  // the first seat always decides, later seats after passes
    CHECK(a.n == b.n && a.targets == b.targets && a.feat == b.feat && a.n_hist == b.n_hist);
    int last_pass = 0;
    for (int i = 0; i < a.n; ++i) {
        float legal = 0;
        for (int k = 0; k < BE::kNumBids; ++k) legal += a.legal_mask[std::size_t(i * BE::kNumBids + k)];
        CHECK(legal == 5.0f);
        for (int k = 0; k < BE::kNumBids; ++k) CHECK(std::fabs(a.targets[std::size_t(i * BE::kNumBids + k)]) <= 1.0f);
        if (a.n_hist[std::size_t(i)] == 3) {  // the fourth seat after three passes: passing ends the deal at 0
            CHECK(a.targets[std::size_t(i * BE::kNumBids)] == 0.0f);
            ++last_pass;
        }
    }
    CHECK(last_pass > 0);
    // the baseline against itself: even
    const auto r = env4.evaluate(1, 1, 2000, 5);
    std::printf("baseline vs baseline bidder: %+.3f +- %.3f per deal\n", r.mean, r.ci95);
    CHECK(std::fabs(r.mean) < 3 * r.ci95 + 1e-9);
    // a bidder that never names trumps loses to the baseline
    const int pass_only = env4.addBidder([](const G&, std::mt19937_64&) { return G::kPass; });
    const auto r2 = env4.evaluate(1, pass_only, 2000, 5);
    std::printf("baseline vs always-pass bidder: %+.3f +- %.3f per deal\n", r2.mean, r2.ci95);
    CHECK(r2.mean - r2.ci95 > 0);
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardbot bidding: all checks passed\n");
    return 0;
}
