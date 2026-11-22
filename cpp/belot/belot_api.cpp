#include "belot_api.h"

#include "bg/agent.h"
#include "bg/bg.h"
#include "bg/bid_model.h"
#include "bg/neural.h"
#include "bg/rule_policy.h"
#include "bg/view.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace bg;

namespace {

thread_local std::string g_error;

int fail(const std::string& message) {
    g_error = message;
    return -1;
}

int writeText(const std::string& text, char* buf, int cap) {
    const int n = int(text.size());
    if (buf && cap > n) std::memcpy(buf, text.c_str(), std::size_t(n) + 1);
    return n;
}

// ---- the networks: loaded once, read-only afterwards (their forward passes are const)

struct Models {
    BgBidModel bid;
    std::array<BgFastPolicyModel, 3> fast;  // suit, no trumps, all trumps
    std::array<BgBeliefModel, 3> belief;
    std::array<BgQGRUModel, 3> q;
    std::string info = "none loaded";
};

Models& models() {
    static Models m;
    return m;
}

const char* const kFamilies[3] = {"suit", "nt", "at"};

// ---- dealing: portable (std::shuffle differs between standard libraries)

std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

std::array<int, kCards> shuffledDeck(std::uint64_t seed, int deal_no) {
    std::uint64_t s = seed ^ (0xD1B54A32D192ED03ull * std::uint64_t(deal_no + 1));
    std::array<int, kCards> deck{};
    for (int i = 0; i < kCards; ++i) deck[i] = i;
    for (int i = kCards - 1; i > 0; --i) std::swap(deck[i], deck[splitmix(s) % std::uint64_t(i + 1)]);
    return deck;
}

int bidIndexOfFlag(std::uint8_t b) { return bidToIndex(b); }

int contractIndex(std::uint8_t contract) {
    const std::uint8_t base = contract & 0x3F;
    for (int i = 6; i >= 1; --i)
        if (base & indexToBid(i)) return i;
    return 0;
}

int doubling(std::uint8_t contract) {
    if (contract & kReDouble) return 2;
    if (contract & kDouble) return 1;
    return 0;
}

}  // namespace

struct BelotGame {
    std::uint64_t seed = 1;
    bool match = true;
    int deal_no = 0;                // deals started, 1 for the first
    GameScore before;               // the score before the current deal
    GameScore score;                // the score now (the current deal added once it is over)
    std::array<int, kCards> deck{};
    std::optional<Round> round;
    std::vector<std::pair<int, int>> actions;  // this deal's choices, in order (save / replay)
    bool recorded = false;

    bool over() const { return round && round->finished() && (!match || score.winner >= 0); }

    void startDeal() {
        ++deal_no;
        before = score;
        deck = shuffledDeck(seed, deal_no);
        round.emplace(deck, score.first, score.hanging);
        actions.clear();
        recorded = false;
        settle();
    }

    // A deal can end inside any move (forced moves are made by the engine).
    void settle() {
        if (round->finished() && !recorded) {
            score.record(round->contract(), round->result());
            recorded = true;
        }
    }
};

namespace {

// ---- legal actions

std::vector<int> legalActions(const Round& r) {
    std::vector<int> out;
    switch (r.decision()) {
        case Round::Decision::kBid: {
            out.push_back(BELOT_ACTION_BID + 0);
            const std::uint8_t avail = r.availableBids();
            for (int i = 1; i <= 8; ++i)
                if (avail & indexToBid(i)) out.push_back(BELOT_ACTION_BID + i);
            break;
        }
        case Round::Decision::kAnnounce: {
            const int n = int(r.offeredAnnounces().size());
            for (int m = 0; m < (1 << n); ++m) out.push_back(BELOT_ACTION_DECLARE + m);
            break;
        }
        case Round::Decision::kPlay: {
            Mask legal = r.legalMoves();
            while (legal) {
                const int c = lowestCard(legal);
                out.push_back(c);
                legal &= legal - 1;
            }
            break;
        }
        case Round::Decision::kDone: break;
    }
    return out;
}

bool isLegal(const Round& r, int action) {
    switch (r.decision()) {
        case Round::Decision::kBid: {
            const int b = action - BELOT_ACTION_BID;
            if (b == 0) return true;
            return b >= 1 && b <= 8 && (r.availableBids() & indexToBid(b));
        }
        case Round::Decision::kAnnounce: {
            const int m = action - BELOT_ACTION_DECLARE;
            return m >= 0 && m < (1 << int(r.offeredAnnounces().size()));
        }
        case Round::Decision::kPlay: return action >= 0 && action < kCards && (r.legalMoves() & bit(action));
        case Round::Decision::kDone: return false;
    }
    return false;
}

void applyAction(Round& r, int action) {
    switch (r.decision()) {
        case Round::Decision::kBid: r.bid(indexToBid(action - BELOT_ACTION_BID)); break;
        case Round::Decision::kAnnounce: {
            const int m = action - BELOT_ACTION_DECLARE;
            std::vector<Announce> chosen;
            const auto& offered = r.offeredAnnounces();
            for (int i = 0; i < int(offered.size()); ++i)
                if (m & (1 << i)) chosen.push_back(offered[i]);
            r.declare(chosen);
            break;
        }
        case Round::Decision::kPlay: {
            const bool belote = beloteAllowed(r.hand(r.toMove()), r.contract(), r.trick(), action);
            r.play(action, belote);
            break;
        }
        case Round::Decision::kDone: break;
    }
}

// ---- the view as JSON

class Json {
public:
    Json& raw(const std::string& s) { comma(); out_ += s; return *this; }
    Json& key(const char* k) { comma(); out_ += '"'; out_ += k; out_ += "\":"; fresh_ = true; return *this; }
    Json& num(long long v) { comma(); out_ += std::to_string(v); return *this; }
    Json& boolean(bool v) { comma(); out_ += v ? "true" : "false"; return *this; }
    Json& str(const char* s) { comma(); out_ += '"'; out_ += s; out_ += '"'; return *this; }
    Json& open(char c) { comma(); out_ += c; fresh_ = true; return *this; }
    Json& close(char c) { out_ += c; fresh_ = false; return *this; }
    Json& cards(Mask m) {
        open('[');
        while (m) { num(lowestCard(m)); m &= m - 1; }
        return close(']');
    }
    const std::string& text() const { return out_; }

private:
    void comma() {
        if (!fresh_ && !out_.empty()) out_ += ',';
        fresh_ = false;
    }
    std::string out_;
    bool fresh_ = true;
};

const char* phaseName(const BelotGame& g) {
    const Round& r = *g.round;
    switch (r.decision()) {
        case Round::Decision::kBid: return "bid";
        case Round::Decision::kAnnounce: return "declare";
        case Round::Decision::kPlay: return "play";
        case Round::Decision::kDone: return g.over() ? "game_over" : "deal_over";
    }
    return "?";
}

int cardPoints(Mask won, std::uint8_t contract) {
    int p = 0;
    while (won) { p += cardValue(lowestCard(won), contract); won &= won - 1; }
    return p;
}

/*
 * {"me", "phase": bid | declare | play | deal_over | game_over, "toMove",
 *  "first" (bids and leads first this deal), "dealNo", "match",
 *  "hand": my cards, "handCounts": [4], "legal": my legal actions (my turn only),
 *  "contract": 0 none .. 6 all trumps, "contractSeat", "doubling": 0 / 1 / 2,
 *  "offered": [{type, cards}] my combinations to declare (declare phase only),
 *  "log": [{k: bid, s, b} | {k: decl, s, t} | {k: play, s, c, bel}] the public record of this deal,
 *  "winners": seat that took each finished trick, "trick": {cards, seats} on the table,
 *  "cardPoints": [2] card points taken so far, "score": [2] game points, "hanging",
 *  "result" (once the deal is over): {points [2], raw [2], capot, hanging,
 *      declared: [{s, t, scored, cards}]}, "winner": team or -1 }
 */
std::string viewJson(const BelotGame& g, int me) {
    const Round& r = *g.round;
    const bool done = r.finished();
    Json j;
    j.open('{');
    j.key("me").num(me).key("phase").str(phaseName(g)).key("toMove").num(done ? -1 : r.toMove());
    j.key("first").num(r.first()).key("dealNo").num(g.deal_no).key("match").boolean(g.match);
    j.key("hand").cards(r.hand(me));
    j.key("handCounts").open('[');
    for (int s = 0; s < kPlayers; ++s) j.num(popcount(r.hand(s)));
    j.close(']');
    j.key("legal").open('[');
    if (!done && r.toMove() == me)
        for (int a : legalActions(r)) j.num(a);
    j.close(']');
    j.key("contract").num(contractIndex(r.contract())).key("contractSeat").num(r.contractPlayer());
    j.key("doubling").num(doubling(r.contract()));
    j.key("offered").open('[');
    if (!done && r.decision() == Round::Decision::kAnnounce && r.toMove() == me)
        for (const Announce& a : r.offeredAnnounces()) {
            j.open('{').key("type").num(a.type).key("cards").cards(announceCards(a)).close('}');
        }
    j.close(']');

    // the public record: bids, then each card with the declarations its seat made just before it
    j.key("log").open('[');
    for (const auto& [seat, b] : r.bids()) {
        j.open('{').key("k").str("bid").key("s").num(seat).key("b").num(bidIndexOfFlag(b)).close('}');
    }
    std::array<bool, kPlayers> told{};
    auto declarationsOf = [&](int seat) {
        if (told[seat]) return;
        told[seat] = true;
        for (const Announce& a : r.announces())
            if (a.player == seat && a.type != kBelot)
                j.open('{').key("k").str("decl").key("s").num(seat).key("t").num(a.type).close('}');
    };
    const auto& plays = r.plays();
    for (std::size_t i = 0; i < plays.size(); ++i) {
        const int seat = plays[i].first;
        if (i < kPlayers) declarationsOf(seat);
        j.open('{').key("k").str("play").key("s").num(seat).key("c").num(plays[i].second.first);
        j.key("bel").num(plays[i].second.second ? 1 : 0).close('}');
    }
    // declared in trick 1, card not played yet
    if (plays.size() < kPlayers)
        for (const Announce& a : r.announces())
            if (a.type != kBelot) declarationsOf(a.player);
    j.close(']');

    j.key("winners").open('[');
    for (int w : r.trickWinners()) j.num(w);
    j.close(']');
    const Trick& t = r.trick();
    j.key("trick").open('{').key("cards").open('[');
    for (int i = 0; i < t.n; ++i) j.num(t.cards[i]);
    j.close(']').key("seats").open('[');
    for (int i = 0; i < t.n; ++i) j.num(t.players[i]);
    j.close(']').close('}');
    j.key("cardPoints").open('[').num(cardPoints(r.won(0), r.contract())).num(cardPoints(r.won(1), r.contract())).close(']');
    j.key("score").open('[').num(g.score.sn).num(g.score.ew).close(']');
    j.key("hanging").num(done ? g.score.hanging : r.hangingBefore());
    if (done) {
        const RoundResult& res = r.result();
        j.key("result").open('{');
        j.key("points").open('[').num(res.sn).num(res.ew).close(']');
        j.key("raw").open('[').num(res.sn_total).num(res.ew_total).close(']');
        j.key("capot").boolean(res.no_tricks).key("hanging").num(res.hanging);
        j.key("declared").open('[');
        for (const Announce& a : r.announces()) {
            j.open('{').key("s").num(a.player).key("t").num(a.type).key("scored").num(a.scored);
            j.key("cards").cards(announceCards(a)).close('}');
        }
        j.close(']').close('}');
    }
    j.key("winner").num(g.score.winner);
    j.close('}');
    return j.text();
}

// ---- the bots

SearchOptions searchOptions(int level, std::uint64_t seed) {
    const Models& m = models();
    SearchOptions o;
    // The configuration of the strongest measured player (runs/bg_bridge_e2e_bid_dbl_5mspec_1000p.log:
    // 54.35% +- 1.74 pp of 2000 games against Kostov's Master), with fewer worlds below the top level.
    o.worlds = level >= 2 ? 192 : 64;
    o.root_exact_trick = 4;
    o.endgame_tricks = 3;
    o.lik_eps = 1.0;
    o.auction_weight = 1.0;
    o.fast_lik_weight = 0.3;
    o.fast_lik_temp = 1.0;
    o.q_blend = 0.25;
    o.q_taper_trick = 5;
    o.race = true;
    o.race_min = 32;
    o.race_step = 32;
    o.race_z = 2.0;
    o.z = -1.0;
    o.seed = seed | 1;
    NeuralSuite* suites[3] = {&o.suit_models, &o.nt_models, &o.at_models};
    for (int f = 0; f < 3; ++f) {
        if (m.fast[f].loaded()) suites[f]->fast = &m.fast[f];
        if (m.belief[f].loaded()) suites[f]->belief = &m.belief[f];
        if (m.q[f].loaded()) suites[f]->q = &m.q[f];
    }
    return o;
}

int ruleCard(const Round& r, int seat, const GameScore& before, std::uint64_t seed) {
    std::mt19937_64 rng(seed | 1);
    DealSampler sampler(publicState(r, seat, before.sn, before.ew));
    if (auto hands = sampler.sample(rng))
        if (auto world = sampler.replay(*hands)) return rulePolicy(*world);
    return lowestCard(r.legalMoves());
}

int botBid(const Round& r, const GameScore& before) {
    const Models& m = models();
    std::uint8_t b = m.bid.loaded() ? m.bid.chooseBid(bidPublicState(r, before.sn, before.ew), true, 3.0)
                                    : naturalBid(r);
    if (b != kPass && !(r.availableBids() & b)) b = kPass;
    return BELOT_ACTION_BID + bidToIndex(b);
}

int botDeclare(const Round& r) {
    const auto& offered = r.offeredAnnounces();
    const auto chosen = chooseDeclarations(offered);
    int mask = 0;
    for (int i = 0; i < int(offered.size()); ++i)
        for (const Announce& c : chosen)
            if (c.type == offered[i].type && c.card == offered[i].card) mask |= 1 << i;
    return BELOT_ACTION_DECLARE + mask;
}

bool myTurn(const BelotGame* g, int seat) {
    return g && g->round && !g->round->finished() && g->round->toMove() == seat;
}

}  // namespace

extern "C" {

const char* belot_last_error(void) { return g_error.c_str(); }

int belot_models_load(const char* dir) {
    Models& m = models();
    const std::string base = std::string(dir ? dir : ".") + "/";
    int loaded = 0;
    std::string info;
    auto note = [&](bool ok, const std::string& name) {
        if (ok) ++loaded;
        info += (info.empty() ? "" : " ") + name + (ok ? "" : "(missing)");
    };
    note(m.bid.load(base + "belot_bid.bin"), "bid");
    for (int f = 0; f < 3; ++f) {
        const std::string fam = kFamilies[f];
        note(m.fast[f].load(base + "belot_" + fam + "_fast.bin"), fam + "_fast");
        note(m.belief[f].load(base + "belot_" + fam + "_belief.bin"), fam + "_belief");
        note(m.q[f].load(base + "belot_" + fam + "_q.bin"), fam + "_q");
    }
    m.info = info;
    return loaded;
}

int belot_models_info(char* buf, int cap) { return writeText(models().info, buf, cap); }

BelotGame* belot_new(uint64_t seed, int match) {
    try {
        auto* g = new BelotGame;
        g->seed = seed;
        g->match = match != 0;
        g->score.first = int(seed % 4);
        g->startDeal();
        return g;
    } catch (const std::exception& e) {
        fail(e.what());
        return nullptr;
    }
}

BelotGame* belot_clone(const BelotGame* game) {
    if (!game) { fail("no game"); return nullptr; }
    return new BelotGame(*game);
}

void belot_free(BelotGame* game) { delete game; }

int belot_next_deal(BelotGame* game) {
    if (!game || !game->round) return fail("no game");
    if (!game->round->finished()) return fail("the deal is not over");
    if (game->over()) return fail("the game is over");
    try {
        game->startDeal();
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    return 0;
}

int belot_view(const BelotGame* game, int seat, char* buf, int cap) {
    if (!game || !game->round) return fail("no game");
    if (seat < 0 || seat >= kPlayers) return fail("no such seat");
    return writeText(viewJson(*game, seat), buf, cap);
}

int belot_hand(const BelotGame* game, int seat, int64_t* out_mask) {
    if (!game || !game->round || seat < 0 || seat >= kPlayers) return fail("no such seat");
    *out_mask = std::int64_t(game->round->hand(seat));
    return 0;
}

int belot_apply(BelotGame* game, int seat, int action) {
    if (!myTurn(game, seat)) return fail("not this seat's turn");
    if (!isLegal(*game->round, action)) return fail("illegal action " + std::to_string(action));
    try {
        applyAction(*game->round, action);
        game->actions.push_back({seat, action});
        game->settle();
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    return 0;
}

int belot_bot_action(const BelotGame* game, int seat, int level, uint64_t seed, int* out_action) {
    if (!myTurn(game, seat)) return fail("not this seat's turn");
    const Round& r = *game->round;
    try {
        switch (r.decision()) {
            case Round::Decision::kBid: *out_action = botBid(r, game->before); break;
            case Round::Decision::kAnnounce: *out_action = botDeclare(r); break;
            case Round::Decision::kPlay:
                if (level <= 0) {
                    *out_action = ruleCard(r, seat, game->before, seed);
                } else {
                    *out_action = searchCard(publicState(r, seat, game->before.sn, game->before.ew),
                                             searchOptions(level, seed)).card;
                }
                break;
            case Round::Decision::kDone: return fail("the deal is over");
        }
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    if (!isLegal(r, *out_action)) {
        // never leave a seat stuck: the first legal action
        *out_action = legalActions(r).front();
    }
    return 0;
}

int belot_hint(const BelotGame* game, int seat, int level, uint64_t seed, int* actions, double* values, int cap,
               int* out_pick) {
    if (!myTurn(game, seat)) return fail("not this seat's turn");
    const Round& r = *game->round;
    int n = 0;
    try {
        if (r.decision() == Round::Decision::kBid) {
            const Models& m = models();
            if (!m.bid.loaded()) return 0;
            std::array<double, kBgNumBids> q{};
            const std::uint16_t legal = m.bid.evalBids(bidPublicState(r, game->before.sn, game->before.ew), true, q);
            int best = -1;
            for (int i = 0; i < kBgNumBids && n < cap; ++i) {
                const int a = BELOT_ACTION_BID + i;
                if (!(legal & (1u << i)) || !isLegal(r, a)) continue;
                actions[n] = a;
                values[n] = q[i];
                if (best < 0 || q[i] > values[best]) best = n;
                ++n;
            }
            *out_pick = best >= 0 ? actions[best] : BELOT_ACTION_BID;
        } else if (r.decision() == Round::Decision::kPlay) {
            const SearchResult res = searchCard(publicState(r, seat, game->before.sn, game->before.ew),
                                                searchOptions(level < 1 ? 1 : level, seed));
            for (const auto& [card, v] : res.values) {
                if (n >= cap) break;
                actions[n] = card;
                values[n] = v;
                ++n;
            }
            *out_pick = res.card;
        } else {
            *out_pick = -1;
        }
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    return n;
}

int belot_save(const BelotGame* game, char* buf, int cap) {
    if (!game || !game->round) return fail("no game");
    std::ostringstream out;
    const GameScore& b = game->before;
    out << "belot1 " << game->seed << ' ' << (game->match ? 1 : 0) << ' ' << game->deal_no << ' ' << b.sn << ' '
        << b.ew << ' ' << b.hanging << ' ' << b.first << ' ' << b.rounds << ' ' << b.winner << ' '
        << game->actions.size();
    for (const auto& [s, a] : game->actions) out << ' ' << s << ' ' << a;
    return writeText(out.str(), buf, cap);
}

BelotGame* belot_load(const char* text) {
    if (!text) { fail("no text"); return nullptr; }
    std::istringstream in(text);
    std::string tag;
    auto g = std::make_unique<BelotGame>();
    int match = 1, n = 0;
    GameScore b;
    in >> tag >> g->seed >> match >> g->deal_no >> b.sn >> b.ew >> b.hanging >> b.first >> b.rounds >> b.winner >> n;
    if (tag != "belot1" || !in || g->deal_no < 1 || n < 0) { fail("not a saved Belot game"); return nullptr; }
    g->match = match != 0;
    g->score = b;
    g->deal_no -= 1;  // startDeal counts it again
    try {
        g->startDeal();
        for (int i = 0; i < n; ++i) {
            int s = -1, a = -1;
            if (!(in >> s >> a)) { fail("the saved game is cut short"); return nullptr; }
            if (belot_apply(g.get(), s, a) < 0) return nullptr;
        }
    } catch (const std::exception& e) {
        fail(e.what());
        return nullptr;
    }
    return g.release();
}

}  // extern "C"
