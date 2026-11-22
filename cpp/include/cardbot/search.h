#pragma once
// The search bot's game-independent parts (docs/DESIGN.md, step 3):
//
//  1. drawWorlds: the determinizations to search on. Every consistent world
//     when few remain (enumerated), otherwise a pool drawn by the game's
//     sampler (e.g. belief-weighted), weighted by how likely each world makes
//     the public record (the game's own log-weight: past plays, the auction),
//     resampled (systematic), deduplicated (a world drawn twice counts twice
//     but is searched once) and shuffled for candidate racing.
//  2. raceMoves: every candidate move evaluated on the same worlds (common
//     random numbers), with candidate racing that drops moves a paired z-test
//     shows to be worse, an optional blend with a learned Q estimate (and its
//     taper), and the decision whether the search overrules the prior move.
//
// The game supplies the parts that know the game -- its sampler and replay,
// its world weights, its play-out or exact solver per world, its prior move
// and Q values -- as template parameters, so any of them can be replaced
// without a cost in speed. Header-only: Bulgarian Belot's search compiles to
// exactly what it was (same arithmetic, same draws from the generator).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cardbot {

// =================================================================== worlds

struct WorldOptions {
    int worlds = 48;   // worlds to search on (and the enumeration cap)
    int pool = 3;      // when weighted: draw pool * worlds, then resample
    bool shuffle = false;  // shuffle the deduplicated worlds (for racing)
};

template <class State>
struct WorldSet {
    std::vector<State> states;
    std::vector<double> weights;  // a world's multiplicity times its likelihood
    bool enumerated = false;
};

/**
 * The worlds for one decision. Source is the game's (all hands H are the
 * game's own type, ordered by operator<):
 *
 *   double count() const;                        // consistent worlds, or more
 *   std::vector<H> enumerate(std::uint64_t cap) const;
 *   std::optional<State> replay(const H&) const; // the game in that world, or nothing
 *   std::optional<H> draw(std::mt19937_64&) const;      // one world (e.g. belief-weighted)
 *   std::optional<H> drawHard(std::mt19937_64&) const;  // the last resort when draw() finds none
 *   bool weighted() const;                       // whether logWeight is used
 *   double logWeight(const H&) const;            // likelihood of the public record in that world
 *   bool hasPrior() const;                       // whether addLogPrior is used
 *   void addLogPrior(const H&, double& lw) const; // a prior on enumerated worlds (e.g. the belief net's),
 *                                                // added onto lw term by term
 */
template <class State, class Source>
WorldSet<State> drawWorlds(const Source& src, const WorldOptions& o, std::mt19937_64& rng) {
    using H = typename decltype(src.enumerate(std::uint64_t(1)))::value_type;
    WorldSet<State> out;
    const bool weighted = src.weighted();
    const bool prior = src.hasPrior();

    const double total_worlds = src.count();
    if (total_worlds > 0.0 && total_worlds <= double(std::max(1, o.worlds))) {
        const auto all = src.enumerate(std::uint64_t(std::max(1, o.worlds)));
        if (!all.empty()) {
            std::vector<double> lws;
            lws.reserve(all.size());
            out.states.reserve(all.size());
            for (const H& h : all) {
                auto r = src.replay(h);
                if (!r) continue;
                double lw = weighted ? src.logWeight(h) : 0.0;
                if (prior) src.addLogPrior(h, lw);
                out.states.push_back(std::move(*r));
                lws.push_back(lw);
            }
            if (!out.states.empty()) {
                out.enumerated = true;
                const double top = *std::max_element(lws.begin(), lws.end());
                out.weights.resize(out.states.size());
                for (std::size_t i = 0; i < out.states.size(); ++i)
                    out.weights[i] = (weighted || prior) ? std::exp(std::max(-40.0, lws[i] - top)) : 1.0;
                return out;
            }
        }
    }

    const int n_draw = weighted ? o.worlds * std::max(1, o.pool) : o.worlds;
    std::vector<H> drawn;
    std::vector<State> pool;
    std::vector<double> lws;
    drawn.reserve(std::size_t(n_draw));
    pool.reserve(std::size_t(n_draw));
    lws.reserve(std::size_t(n_draw));
    for (int i = 0; i < n_draw; ++i) {
        auto h = src.draw(rng);
        if (!h) break;
        auto r = src.replay(*h);
        if (!r) continue;
        drawn.push_back(*h);
        pool.push_back(std::move(*r));
        lws.push_back(weighted ? src.logWeight(*h) : 0.0);
    }
    if (pool.empty()) {
        // the rare case the draws find nothing consistent: the last resort, unweighted
        for (int i = 0; i < std::max(8, o.worlds / 4); ++i) {
            auto h = src.drawHard(rng);
            if (!h) break;
            auto r = src.replay(*h);
            if (!r) continue;
            drawn.push_back(*h);
            pool.push_back(std::move(*r));
            lws.push_back(0.0);
        }
    }
    if (pool.empty()) throw std::runtime_error("cardbot::drawWorlds: no consistent world");

    std::vector<std::size_t> picked;
    if (weighted) {
        // systematic resampling in proportion to the likelihoods
        const int count = std::min(o.worlds, int(pool.size()));
        const double top = *std::max_element(lws.begin(), lws.end());
        std::vector<double> pw(pool.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < pool.size(); ++i) {
            pw[i] = std::exp(std::max(-40.0, lws[i] - top));
            sum += pw[i];
        }
        const double step = sum / double(count);
        double u = std::uniform_real_distribution<double>(0.0, step)(rng);
        double cum = pw[0];
        std::size_t j = 0;
        picked.reserve(std::size_t(count));
        for (int k = 0; k < count; ++k, u += step) {
            while (u > cum && j + 1 < pw.size()) cum += pw[++j];
            picked.push_back(j);
        }
    } else {
        picked.resize(pool.size());
        for (std::size_t i = 0; i < pool.size(); ++i) picked[i] = i;
    }

    // each distinct world searched once, weighted by how often it was drawn
    std::sort(picked.begin(), picked.end(), [&](std::size_t a, std::size_t b) { return drawn[a] < drawn[b]; });
    out.states.reserve(picked.size());
    out.weights.reserve(picked.size());
    for (std::size_t k = 0; k < picked.size(); ++k) {
        const std::size_t idx = picked[k];
        if (k > 0 && drawn[idx] == drawn[picked[k - 1]]) {
            out.weights.back() += 1.0;
        } else {
            out.states.push_back(std::move(pool[idx]));
            out.weights.push_back(1.0);
        }
    }
    // no order bias for the racing that follows
    if (o.shuffle && out.states.size() > 1) {
        std::vector<std::size_t> perm(out.states.size());
        for (std::size_t i = 0; i < perm.size(); ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<State> ss;
        std::vector<double> sw;
        ss.reserve(out.states.size());
        sw.reserve(out.weights.size());
        for (std::size_t idx : perm) {
            ss.push_back(std::move(out.states[idx]));
            sw.push_back(out.weights[idx]);
        }
        out.states.swap(ss);
        out.weights.swap(sw);
    }
    return out;
}

// ==================================================================== moves

struct RaceOptions {
    bool race = false;     // drop clearly worse moves early
    int race_min = 32;     // worlds before the first test
    int race_step = 32;    // worlds between tests
    double race_z = 2.0;   // a move is dropped when the leader beats it by race_z standard errors
    double z = 1.0;        // the search overrules the prior move only by z standard errors (z < 0: always argmax)
    double margin = 0.1;   // ... and by more than this margin
};

/**
 * A learned estimate blended into the search's values: score = (1 - weight)
 * * mean + weight * q. With center, q is first put on the search's scale:
 * average of the alive moves' means + slope * (q - average of their q).
 */
struct QBlend {
    double weight = 0.0;
    std::vector<double> q;  // per move, in moves order (empty: no blend)
    bool center = false;
    double slope = 1.0;
};

template <class Move>
struct RaceResult {
    Move move{};
    bool overruled = false;
    int worlds = 0;                              // the evaluated worlds' total weight, rounded
    std::vector<std::pair<Move, double>> values;  // the moves still in the race, with their scores
};

/**
 * Evaluates moves on the worlds and chooses. evaluate(w, alive, val) fills
 * val[i * nw + w] for every move i with alive[i] (the game's play-out after
 * move i, or its exact solver), where nw = weights.size(). With exact (or no
 * racing, few worlds, two moves) every world is evaluated for every move;
 * otherwise worlds are evaluated in order and moves are dropped by racing.
 * prior is the move kept unless the search beats it clearly (z, margin);
 * enumerated worlds always decide by the best score.
 */
template <class Move, class Evaluate>
RaceResult<Move> raceMoves(const std::vector<Move>& moves, const std::vector<double>& wts, bool enumerated,
                           bool exact, const Move& prior, Evaluate&& evaluate, const QBlend& qb,
                           const RaceOptions& o) {
    RaceResult<Move> res;
    res.move = prior;
    const std::size_t nm = moves.size();
    const std::size_t nw = wts.size();
    std::vector<double> val(nm * nw, 0.0);
    std::vector<char> alive(nm, 1);
    std::size_t w_eval = 0;

    if (exact || !o.race || nw <= std::size_t(std::max(1, o.race_min)) || nm <= 2) {
        for (std::size_t w = 0; w < nw; ++w) evaluate(w, alive, val);
        w_eval = nw;
    } else {
        std::size_t next_check = std::size_t(std::max(8, o.race_min));
        const std::size_t step = std::size_t(std::max(8, o.race_step));
        std::size_t n_alive = nm;
        for (std::size_t w = 0; w < nw; ++w) {
            evaluate(w, alive, val);
            w_eval = w + 1;
            if (w_eval >= next_check && w_eval < nw && n_alive > 1) {
                next_check = w_eval + step;
                double sw = 0.0, sw2 = 0.0;
                for (std::size_t k = 0; k < w_eval; ++k) {
                    sw += wts[k];
                    sw2 += wts[k] * wts[k];
                }
                const double ess = (sw * sw) / std::max(sw2, 1e-12);
                std::vector<double> cur_m(nm, -1e30);
                std::size_t leader = 0;
                for (std::size_t i = 0; i < nm; ++i) {
                    if (!alive[i]) continue;
                    double acc = 0.0;
                    for (std::size_t k = 0; k < w_eval; ++k) acc += wts[k] * val[i * nw + k];
                    cur_m[i] = acc / sw;
                    if (!alive[leader] || cur_m[i] > cur_m[leader]) leader = i;
                }
                for (std::size_t i = 0; i < nm; ++i) {
                    if (!alive[i] || i == leader) continue;
                    const double diff = cur_m[leader] - cur_m[i];
                    if (diff <= 0.0) continue;
                    double var = 0.0;
                    for (std::size_t k = 0; k < w_eval; ++k) {
                        const double d = (val[leader * nw + k] - val[i * nw + k]) - diff;
                        var += wts[k] * d * d;
                    }
                    const double se = std::sqrt(std::max(var / sw, 1e-12) / std::max(ess, 1.0));
                    if (diff > o.race_z * se) {
                        alive[i] = 0;
                        --n_alive;
                    }
                }
                if (n_alive <= 1) break;
            }
        }
    }

    double sum_w = 0.0, sum_w2 = 0.0;
    for (std::size_t w = 0; w < w_eval; ++w) {
        sum_w += wts[w];
        sum_w2 += wts[w] * wts[w];
    }
    res.worlds = int(std::lround(sum_w));

    std::vector<double> mean(nm, -1e30);
    double alive_search_avg = 0.0;
    int alive_cnt = 0;
    for (std::size_t i = 0; i < nm; ++i) {
        if (!alive[i]) continue;
        double acc = 0.0;
        for (std::size_t w = 0; w < w_eval; ++w) acc += wts[w] * val[i * nw + w];
        mean[i] = acc / sum_w;
        alive_search_avg += mean[i];
        ++alive_cnt;
    }
    if (alive_cnt > 0) alive_search_avg /= double(alive_cnt);

    std::vector<double> score = mean;
    const double w_q = qb.weight;
    if (w_q > 0.0 && !qb.q.empty()) {
        double q_avg = 0.0;
        for (std::size_t i = 0; i < nm; ++i)
            if (alive[i]) q_avg += qb.q[i];
        if (alive_cnt > 0) q_avg /= double(alive_cnt);
        for (std::size_t i = 0; i < nm; ++i) {
            if (!alive[i]) continue;
            const double q_aligned = qb.center ? (alive_search_avg + qb.slope * (qb.q[i] - q_avg)) : qb.q[i];
            score[i] = (1.0 - w_q) * mean[i] + w_q * q_aligned;
        }
    }

    std::size_t best = 0, prior_i = 0;
    bool best_init = false;
    for (std::size_t i = 0; i < nm; ++i) {
        if (moves[i] == prior) prior_i = i;
        if (alive[i]) {
            res.values.push_back({moves[i], score[i]});
            if (!best_init || score[i] > score[best]) {
                best = i;
                best_init = true;
            }
        }
    }
    if (best == prior_i) return res;
    if (o.z < 0 || enumerated || !alive[prior_i]) {
        res.move = moves[best];
        res.overruled = true;
        return res;
    }
    const double m = score[best] - score[prior_i];
    const double dm = mean[best] - mean[prior_i];
    double var = 0.0;
    for (std::size_t w = 0; w < w_eval; ++w) {
        const double d = (val[best * nw + w] - val[prior_i * nw + w]) - dm;
        var += wts[w] * d * d;
    }
    const double ess = (sum_w * sum_w) / std::max(sum_w2, 1e-12);
    const double se = (1.0 - w_q) * std::sqrt(std::max(var / sum_w, 1e-12) / std::max(ess, 1.0));
    if (m > o.margin && m > o.z * se) {
        res.move = moves[best];
        res.overruled = true;
    }
    return res;
}

}  // namespace cardbot
