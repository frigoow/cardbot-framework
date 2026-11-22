#pragma once
// The auction (docs/DESIGN.md, steps 2, 5 and 6), as the Belote bot trains its
// bidder: counterfactual bid values. At every auction decision of a learning
// seat, every legal bid is tried on the same deal; each branch's auction is
// finished by the table's bidders and its cards played by the play agent, and
// the deal's return for the bidder's team is that bid's target. The bid net
// (cardbot::BidModel, python/cardbot/train/bidding.py) is fitted to them, then
// bids for the learning seats from the next round on (self-play), and in step 6
// for the card-play trainer's auctions (BatchEnv::setAutoAgent).
//
// The game's BidEncoder decides everything game-specific:
//
//   struct MyBidEncoder {
//       static constexpr int kFeatDim, kHistDim, kMaxHist, kNumBids;
//       G newGame(std::mt19937_64&) const;                         // a fresh deal, at its auction
//       bool isBid(const G&) const;                                // a decision of the auction
//       int bidSlot(const G&, int seat, ActionOf<G>) const;        // the bid's value slot, [0, kNumBids)
//       int encode(const G&, int seat, float* feat, float* hist) const;   // returns the auction length
//       float reward(const G&, int team) const;                    // a finished deal's return for a team
//       ActionOf<G> baselineBid(const G&, std::mt19937_64&) const; // the step-2 bidder (a rule, or random)
//   };
//
// And everything else is a knob: the play agent (random, a rule, the fast
// policy, a search), the table (which bidder each seat uses on each deal:
// self-play, mixed partners, fixed opponents), the exploration and the number
// of play-outs per branch.

#include "cardbot/eval.h"
#include "cardbot/game.h"
#include "cardbot/models.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cardbot {

/** The bid net's value for each of g's legal bids (in legalActions() order). */
template <class G, class BidEnc>
std::vector<float> bidValues(const BidModel& model, const BidEnc& enc, const G& g, const std::vector<ActionOf<G>>& legal) {
    thread_local std::vector<float> feat, hist, values;
    feat.assign(std::size_t(BidEnc::kFeatDim), 0.0f);
    hist.assign(std::size_t(BidEnc::kMaxHist) * BidEnc::kHistDim, 0.0f);
    values.resize(std::size_t(model.numBids()));
    const int seat = g.toMove();
    const int n = enc.encode(g, seat, feat.data(), hist.data());
    model.forward(feat.data(), hist.data(), n, values.data());
    std::vector<float> out(legal.size());
    for (std::size_t k = 0; k < legal.size(); ++k) {
        const int slot = enc.bidSlot(g, seat, legal[k]);
        if (slot < 0 || slot >= model.numBids()) throw std::out_of_range("cardbot: bid slot outside the bid net's outputs");
        out[k] = values[std::size_t(slot)];
    }
    return out;
}

/** Bids the legal bid of highest value (auction decisions only). */
template <class G, class BidEnc>
Agent<G> bidModelAgent(std::shared_ptr<const BidModel> model, BidEnc enc = BidEnc()) {
    if (!model || !model->loaded()) throw std::invalid_argument("bidModelAgent: no loaded model");
    if (model->featDim() != BidEnc::kFeatDim || model->histDim() != BidEnc::kHistDim)
        throw std::invalid_argument("bidModelAgent: the model's sizes differ from the encoder's");
    return [model, enc](const G& g, std::mt19937_64&) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const auto v = bidValues(*model, enc, g, legal);
        return legal[std::size_t(std::max_element(v.begin(), v.end()) - v.begin())];
    };
}

/** One agent from two: `bidder` for the auction's decisions (isBid), `player` for the rest. */
template <class G>
Agent<G> phaseAgent(std::function<bool(const G&)> is_bid, Agent<G> bidder, Agent<G> player) {
    return [is_bid, bidder, player](const G& g, std::mt19937_64& rng) {
        return is_bid(g) ? bidder(g, rng) : player(g, rng);
    };
}

/** The counterfactual bid targets of a batch of deals; rows are auction decisions. */
struct BidBatch {
    int n = 0;
    std::vector<float> feat;            // [n, kFeatDim]
    std::vector<float> hist;            // [n, kMaxHist, kHistDim]
    std::vector<std::int32_t> n_hist;   // [n]
    std::vector<float> legal_mask;      // [n, kNumBids]
    std::vector<float> targets;         // [n, kNumBids]: the bid's return for the bidder's team (0 if illegal)
};

template <class G, class BidEnc>
class BidEnv {
public:
    using Action = ActionOf<G>;
    using Table = std::array<int, std::size_t(G::kPlayers)>;

    explicit BidEnv(int threads = 4, BidEnc enc = BidEnc()) : enc_(std::move(enc)), threads_(std::max(1, threads)) {
        play_ = randomAgent();
    }

    const BidEnc& encoder() const { return enc_; }

    /** The agent that plays every decision outside the auction (default: uniformly random). */
    void setPlayAgent(Agent<G> a) { play_ = std::move(a); }
    /** The learning seats' bid net (until then they bid by enc.baselineBid). */
    void setBidModel(std::shared_ptr<const BidModel> m) {
        bid_model_ = std::move(m);
        learner_ = bid_model_ ? bidModelAgent<G, BidEnc>(bid_model_, enc_) : Agent<G>();
    }
    bool hasBidModel() const { return bool(bid_model_); }
    /** Another bidder for the table; returns its index (the learner is 0, the baseline 1). */
    int addBidder(Agent<G> a) {
        others_.push_back(std::move(a));
        return int(others_.size()) + 1;
    }
    /** Which bidder each seat uses on deal i (default: the learner everywhere, i.e. self-play). */
    void setTable(std::function<Table(int deal)> t) { table_ = std::move(t); }

    /** Bidder k: 0 the learner (bid net, else the baseline), 1 the baseline, 2.. the added ones. */
    Agent<G> bidder(int k) const {
        if (k == 0 && learner_) return learner_;
        if (k <= 1) {
            const BidEnc enc = enc_;
            return [enc](const G& g, std::mt19937_64& rng) { return enc.baselineBid(g, rng); };
        }
        return others_.at(std::size_t(k - 2));
    }
    /** The full agent of bidder k: it bids as bidder k and plays as the play agent. */
    Agent<G> agent(int k) const {
        const BidEnc enc = enc_;
        return phaseAgent<G>([enc](const G& g) { return enc.isBid(g); }, bidder(k), play_);
    }

    /**
     * Counterfactual targets on num_deals fresh deals. The auction's actual
     * course follows the table's bidders, a random legal bid with probability
     * eps; at each decision of a learning seat (bidder 0) with two or more
     * legal bids, every legal bid is finished `samples` times (the same random
     * streams for every bid) and averaged. Reproducible for a seed whatever the
     * thread count.
     */
    BidBatch generate(int num_deals, std::uint64_t seed, double eps = 0.05, int samples = 1) const {
        struct Row {
            std::vector<float> feat, hist, mask, tgt;
            int n_hist = 0;
        };
        std::vector<std::vector<Row>> per_deal(std::size_t(std::max(0, num_deals)));
        std::vector<Agent<G>> bidders;
        for (int k = 0; k < int(others_.size()) + 2; ++k) bidders.push_back(bidder(k));
        auto run = [&](int d) {
            std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(d + 1) * 0xBF58476D1CE4E5B9ull);
            G g = enc_.newGame(rng);
            const Table table = tableFor(d);
            auto finish = [&](G x, std::mt19937_64& r) {
                while (!x.finished()) {
                    const Agent<G>& a = enc_.isBid(x) ? bidders[std::size_t(table[std::size_t(x.toMove())])] : play_;
                    x.apply(a(x, r));
                }
                return x;
            };
            while (!g.finished() && enc_.isBid(g)) {
                const int seat = g.toMove();
                const auto legal = g.legalActions();
                if (table[std::size_t(seat)] == 0 && legal.size() >= 2) {
                    Row row;
                    row.feat.assign(std::size_t(BidEnc::kFeatDim), 0.0f);
                    row.hist.assign(std::size_t(BidEnc::kMaxHist) * BidEnc::kHistDim, 0.0f);
                    row.mask.assign(std::size_t(BidEnc::kNumBids), 0.0f);
                    row.tgt.assign(std::size_t(BidEnc::kNumBids), 0.0f);
                    row.n_hist = enc_.encode(g, seat, row.feat.data(), row.hist.data());
                    std::vector<std::uint64_t> seeds(static_cast<std::size_t>(std::max(1, samples)));
                    for (auto& s : seeds) s = rng();
                    const int team = g.team(seat);
                    for (const Action& a : legal) {
                        const int slot = enc_.bidSlot(g, seat, a);
                        if (slot < 0 || slot >= BidEnc::kNumBids) throw std::out_of_range("cardbot::BidEnv: bid slot");
                        double sum = 0.0;
                        for (std::uint64_t s : seeds) {
                            G b = g;
                            b.apply(a);
                            std::mt19937_64 r(s);
                            sum += enc_.reward(finish(std::move(b), r), team);
                        }
                        row.mask[std::size_t(slot)] = 1.0f;
                        row.tgt[std::size_t(slot)] = float(sum / double(seeds.size()));
                    }
                    per_deal[std::size_t(d)].push_back(std::move(row));
                }
                const bool explore = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < eps;
                g.apply(explore ? legal[std::uniform_int_distribution<std::size_t>(0, legal.size() - 1)(rng)]
                                : bidders[std::size_t(table[std::size_t(seat)])](g, rng));
            }
        };
        parallel(num_deals, run);
        BidBatch out;
        for (const auto& rows : per_deal)
            for (const Row& r : rows) {
                out.feat.insert(out.feat.end(), r.feat.begin(), r.feat.end());
                out.hist.insert(out.hist.end(), r.hist.begin(), r.hist.end());
                out.n_hist.push_back(r.n_hist);
                out.legal_mask.insert(out.legal_mask.end(), r.mask.begin(), r.mask.end());
                out.targets.insert(out.targets.end(), r.tgt.begin(), r.tgt.end());
                ++out.n;
            }
        return out;
    }

    /** Bidder a against bidder b (indices as in bidder()), both playing with the play agent, on duplicate deals. */
    DuelResult evaluate(int a, int b, int pairs, std::uint64_t seed = 1) const {
        const BidEnc enc = enc_;
        const std::function<G(std::uint64_t)> deal = [enc](std::uint64_t s) {
            std::mt19937_64 r(s);
            return enc.newGame(r);
        };
        return duel<G>(deal, agent(a), agent(b), pairs, seed, unsigned(threads_));
    }

private:
    static Agent<G> randomAgent() {
        return [](const G& g, std::mt19937_64& rng) {
            const auto legal = g.legalActions();
            return legal[std::uniform_int_distribution<std::size_t>(0, legal.size() - 1)(rng)];
        };
    }
    Table tableFor(int d) const {
        if (table_) return table_(d);
        Table t{};
        return t;
    }
    template <class F>
    void parallel(int n, F&& f) const {
        if (threads_ <= 1 || n < 2) {
            for (int i = 0; i < n; ++i) f(i);
            return;
        }
        // an exception inside a thread would end the process: the first one is rethrown here
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(threads_));
        std::vector<std::thread> pool;
        for (int t = 0; t < threads_; ++t)
            pool.emplace_back([&, t] {
                try {
                    for (int i = t; i < n; i += threads_) f(i);
                } catch (...) {
                    errors[std::size_t(t)] = std::current_exception();
                }
            });
        for (auto& th : pool) th.join();
        for (auto& e : errors)
            if (e) std::rethrow_exception(e);
    }

    BidEnc enc_;
    int threads_ = 1;
    Agent<G> play_, learner_;
    std::shared_ptr<const BidModel> bid_model_;
    std::vector<Agent<G>> others_;
    std::function<Table(int)> table_;
};

}  // namespace cardbot
