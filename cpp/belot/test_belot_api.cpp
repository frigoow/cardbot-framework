// belot_api end to end: whole games by the bots in every seat through the C
// interface only, illegal moves refused, a saved game continued identically.
// With BELOT_MODELS=<dir> set, also times the full-strength bot on the nets.
#include "belot_api.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed: %s\n", __FILE__, \
                         __LINE__, #cond, belot_last_error());               \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

std::string view(const BelotGame* g, int seat) {
    const int n = belot_view(g, seat, nullptr, 0);
    std::string s(std::size_t(n) + 1, '\0');
    belot_view(g, seat, s.data(), n + 1);
    s.resize(std::size_t(n));
    return s;
}

std::string save(const BelotGame* g) {
    const int n = belot_save(g, nullptr, 0);
    std::string s(std::size_t(n) + 1, '\0');
    belot_save(g, s.data(), n + 1);
    s.resize(std::size_t(n));
    return s;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

int toMove(const std::string& v) {
    const auto p = v.find("\"toMove\":");
    return std::atoi(v.c_str() + p + 9);
}

// Plays the bots until the game is over; returns the number of decisions.
int playOut(BelotGame* g, int level, std::uint64_t seed, double* max_ms = nullptr) {
    int decisions = 0;
    for (int guard = 0; guard < 5000; ++guard) {
        const std::string v = view(g, 0);
        if (has(v, "\"phase\":\"game_over\"")) return decisions;
        if (has(v, "\"phase\":\"deal_over\"")) {
            CHECK(belot_next_deal(g) == 0);
            continue;
        }
        const int seat = toMove(v);
        int action = -1;
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(belot_bot_action(g, seat, level, seed + std::uint64_t(guard), &action) == 0);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (max_ms && ms > *max_ms) *max_ms = ms;
        CHECK(belot_apply(g, seat, action) == 0);
        ++decisions;
    }
    CHECK(!"the game did not end");
    return decisions;
}

void testWholeGames() {
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        BelotGame* g = belot_new(seed, 1);
        CHECK(g != nullptr);
        playOut(g, 0, seed * 1000);
        const std::string v = view(g, 0);
        CHECK(has(v, "\"phase\":\"game_over\""));
        CHECK(!has(v, "\"winner\":-1"));
        CHECK(belot_next_deal(g) < 0);  // nothing after the game
        belot_free(g);
    }
    // a single deal
    BelotGame* g = belot_new(42, 0);
    playOut(g, 0, 7);
    CHECK(has(view(g, 2), "\"phase\":\"game_over\""));
    CHECK(has(view(g, 2), "\"result\":"));
    belot_free(g);
}

void testRefusesWhatIsNotAllowed() {
    BelotGame* g = belot_new(5, 1);
    const std::string v = view(g, 0);
    const int seat = toMove(v);
    CHECK(has(v, "\"phase\":\"bid\""));
    CHECK(belot_apply(g, (seat + 1) % 4, BELOT_ACTION_BID) < 0);  // not that seat's turn
    CHECK(belot_apply(g, seat, 3) < 0);                           // a card during the auction
    CHECK(belot_apply(g, seat, BELOT_ACTION_BID + 7) < 0);        // double with nothing to double
    // only the seat to move sees legal actions, and nobody sees another hand
    CHECK(has(view(g, (seat + 1) % 4), "\"legal\":[]"));
    CHECK(!has(view(g, seat), "\"legal\":[]"));
    belot_free(g);
}

void testSaveAndLoad() {
    BelotGame* g = belot_new(11, 1);
    // into the second deal, then a few decisions into the play
    int decisions = 0;
    while (decisions < 400) {
        const std::string v = view(g, 0);
        if (has(v, "\"dealNo\":2") && has(v, "\"phase\":\"play\"") && has(v, "\"winners\":[1") ) break;
        if (has(v, "\"phase\":\"deal_over\"")) { belot_next_deal(g); continue; }
        if (has(v, "\"phase\":\"game_over\"")) break;
        const int seat = toMove(v);
        int a = -1;
        belot_bot_action(g, seat, 0, 99 + std::uint64_t(decisions), &a);
        belot_apply(g, seat, a);
        ++decisions;
    }
    const std::string text = save(g);
    BelotGame* h = belot_load(text.c_str());
    CHECK(h != nullptr);
    if (h) {
        for (int s = 0; s < 4; ++s) CHECK(view(g, s) == view(h, s));
        CHECK(save(h) == text);
        // and both go on the same way
        playOut(g, 0, 5);
        playOut(h, 0, 5);
        CHECK(view(g, 0) == view(h, 0));
        belot_free(h);
    }
    CHECK(belot_load("nonsense") == nullptr);
    belot_free(g);
}

void timeFullStrength(const char* dir) {
    const int loaded = belot_models_load(dir);
    char info[512];
    belot_models_info(info, sizeof info);
    std::printf("models: %d loaded (%s)\n", loaded, info);
    CHECK(loaded == 10);
    for (int level = 1; level < BELOT_LEVELS; ++level) {
        BelotGame* g = belot_new(3, 0);
        double max_ms = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const int n = playOut(g, level, 1, &max_ms);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("level %d: %d decisions, %.1f ms mean, %.1f ms max\n", level, n, ms / n, max_ms);
        // a hint for the next human decision of a fresh deal
        belot_free(g);
    }
    BelotGame* g = belot_new(8, 0);
    int actions[64];
    double values[64];
    int pick = -1;
    const int seat = toMove(view(g, 0));
    const int n = belot_hint(g, seat, 2, 1, actions, values, 64, &pick);
    CHECK(n >= 1);
    CHECK(pick >= BELOT_ACTION_BID);
    belot_free(g);
}

}  // namespace

int main() {
    testWholeGames();
    testRefusesWhatIsNotAllowed();
    testSaveAndLoad();
    if (const char* dir = std::getenv("BELOT_MODELS")) timeFullStrength(dir);
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("belot_api: all checks passed\n");
    return 0;
}
