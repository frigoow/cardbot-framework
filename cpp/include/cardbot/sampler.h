#pragma once
// The world sampler's counting core: deals a trick-taking game's hidden cards
// to the three seats the observer cannot see, exactly uniformly (or weighted
// per card and seat, e.g. by a belief network) over every deal that respects
//   - each card's domain: the seats that may hold it (voids, pinned cards...),
//   - each seat's hand size,
//   - optionally, a mixture of alternative domains with weights ("placements":
//     e.g. every way an opponent's announced combination can sit in their
//     hand, weighted by the deals that complete it).
// It counts the deals with a dynamic programme over (card, cards left for
// seat 0, cards left for seat 1), samples by walking the counts, enumerates
// them when few remain, and gives each card's exact marginal per seat.
//
// The game turns its public record into the domains, decides which sampled
// deals are consistent (accept), and maps the three hidden seats to its own.
// Header-only: the arithmetic must compile exactly as the game's own sampler
// did (the same sums in the same order, the same draws from the generator).

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace cardbot {

/**
 * The hidden cards and what may hold them. MaxHand is the largest hand a
 * hidden seat can have (8 for a 32-card game for four).
 */
template <int MaxHand = 8>
class HiddenDeal {
public:
    static constexpr int kSide = MaxHand + 1;
    static constexpr int kPlane = kSide * kSide;

    /** The cards each hidden seat gets: bit c = card c. */
    using Hands = std::array<std::uint64_t, 3>;
    /** Per hidden card: bit k = hidden seat k may hold it. */
    using Domain = std::vector<std::uint8_t>;
    /** Per hidden card (in cards() order), a weight for each hidden seat. */
    using Weights = std::vector<std::array<double, 3>>;
    using Accept = std::function<bool(const Hands&)>;

    struct Placement {
        Domain domain;
        double weight;
    };

    HiddenDeal() = default;
    HiddenDeal(std::vector<int> cards, Domain domain, std::array<int, 3> need)
        : cards_(std::move(cards)), domain_(std::move(domain)), need_(need), two_seat_(need[2] == 0) {
        if (two_seat_) ways2_ = buildWays2(domain_);
        else ways_ = buildWays(domain_);
    }

    const std::vector<int>& cards() const { return cards_; }
    const Domain& domain() const { return domain_; }
    const std::array<int, 3>& need() const { return need_; }
    bool twoSeat() const { return two_seat_; }

    // ---- the counting tables

    /** 2-seat fast table (when need[2] == 0): w[i][a] = ways for cards i.. with a left for seat 0 (rest to seat 1). */
    static std::vector<double> buildWays2(const Domain& dom) {
        const std::size_t n = dom.size();
        std::vector<double> w((n + 1) * kSide, 0.0);
        auto at = [&](std::size_t i, int a) -> double& { return w[i * kSide + std::size_t(a)]; };
        for (std::size_t i = n + 1; i-- > 0;) {
            const int rem = int(n - i);
            const int a_lo = std::max(0, rem - MaxHand);
            const int a_hi = std::min(MaxHand, rem);
            for (int a = a_lo; a <= a_hi; ++a) {
                const int b = rem - a;
                double x = 0.0;
                if (i == n) {
                    x = (a == 0 && b == 0) ? 1.0 : 0.0;
                } else {
                    const std::uint8_t d = dom[i];
                    if ((d & 1) && a > 0) x += at(i + 1, a - 1);
                    if ((d & 2) && b > 0) x += at(i + 1, a);
                }
                at(i, a) = x;
            }
        }
        return w;
    }

    static std::vector<double> buildWaysWeighted2(const Domain& dom, const Weights& weights) {
        const std::size_t n = dom.size();
        std::vector<double> w((n + 1) * kSide, 0.0);
        auto at = [&](std::size_t i, int a) -> double& { return w[i * kSide + std::size_t(a)]; };
        for (std::size_t i = n + 1; i-- > 0;) {
            const int rem = int(n - i);
            const int a_lo = std::max(0, rem - MaxHand);
            const int a_hi = std::min(MaxHand, rem);
            for (int a = a_lo; a <= a_hi; ++a) {
                const int b = rem - a;
                double x = 0.0;
                if (i == n) {
                    x = (a == 0 && b == 0) ? 1.0 : 0.0;
                } else {
                    const std::uint8_t d = dom[i];
                    const auto& cw = weights[i];
                    if ((d & 1) && a > 0) x += cw[0] * at(i + 1, a - 1);
                    if ((d & 2) && b > 0) x += cw[1] * at(i + 1, a);
                }
                at(i, a) = x;
            }
        }
        return w;
    }

    static double ways2At(const std::vector<double>& w, std::size_t i, int a) {
        return w[i * kSide + std::size_t(a)];
    }

    /** w[i][a][b]: the ways to deal cards i.. with a and b left for seats 0 and 1 (the rest to seat 2). */
    static std::vector<double> buildWays(const Domain& dom) {
        const std::size_t n = dom.size();
        std::vector<double> w((n + 1) * kPlane, 0.0);
        auto at = [&](std::size_t i, int a, int b) -> double& { return w[i * kPlane + std::size_t(a * kSide + b)]; };
        for (std::size_t i = n + 1; i-- > 0;) {
            for (int a = 0; a <= MaxHand; ++a) {
                for (int b = 0; b <= MaxHand; ++b) {
                    const int cc = int(n - i) - a - b;
                    double x = 0;
                    if (cc >= 0 && cc <= MaxHand) {
                        if (i == n) {
                            x = (a == 0 && b == 0 && cc == 0) ? 1.0 : 0.0;
                        } else {
                            const std::uint8_t d = dom[i];
                            if ((d & 1) && a > 0) x += at(i + 1, a - 1, b);
                            if ((d & 2) && b > 0) x += at(i + 1, a, b - 1);
                            if ((d & 4) && cc > 0) x += at(i + 1, a, b);
                        }
                    }
                    at(i, a, b) = x;
                }
            }
        }
        return w;
    }

    static std::vector<double> buildWaysWeighted(const Domain& dom, const Weights& weights) {
        const std::size_t n = dom.size();
        std::vector<double> w((n + 1) * kPlane, 0.0);
        auto at = [&](std::size_t i, int a, int b) -> double& { return w[i * kPlane + std::size_t(a * kSide + b)]; };
        for (std::size_t i = n + 1; i-- > 0;) {
            for (int a = 0; a <= MaxHand; ++a) {
                for (int b = 0; b <= MaxHand; ++b) {
                    const int cc = int(n - i) - a - b;
                    double x = 0;
                    if (cc >= 0 && cc <= MaxHand) {
                        if (i == n) {
                            x = (a == 0 && b == 0 && cc == 0) ? 1.0 : 0.0;
                        } else {
                            const std::uint8_t d = dom[i];
                            const auto& cw = weights[i];
                            if ((d & 1) && a > 0) x += cw[0] * at(i + 1, a - 1, b);
                            if ((d & 2) && b > 0) x += cw[1] * at(i + 1, a, b - 1);
                            if ((d & 4) && cc > 0) x += cw[2] * at(i + 1, a, b);
                        }
                    }
                    at(i, a, b) = x;
                }
            }
        }
        return w;
    }

    static double waysAt(const std::vector<double>& w, std::size_t i, int a, int b) {
        return w[i * kPlane + std::size_t(a * kSide + b)];
    }

    /** The deals a domain admits. */
    double countFor(const Domain& dom) const {
        if (two_seat_) return ways2At(buildWays2(dom), 0, need_[0]);
        return waysAt(buildWays(dom), 0, need_[0], need_[1]);
    }

    // ---- placements: a mixture of alternative domains

    /** Adds an alternative domain with its weight (usually countFor(domain)). */
    void addPlacement(Domain dom, double weight) {
        placements_.push_back({std::move(dom), weight});
        placement_total_ += weight;
    }
    void clearPlacements() { placements_.clear(); placement_total_ = 0; }
    const std::vector<Placement>& placements() const { return placements_; }

    // ---- counting, enumeration, marginals, sampling

    double ways(std::size_t i, int a, int b) const {
        if (two_seat_) return (int(cards_.size() - i) - a - b == 0) ? ways2At(ways2_, i, a) : 0.0;
        return waysAt(ways_, i, a, b);
    }

    double count() const {
        if (!placements_.empty()) return placement_total_;
        if (two_seat_) return ways2At(ways2_, 0, need_[0]);
        return waysAt(ways_, 0, need_[0], need_[1]);
    }

    /**
     * Visits every deal (each placement's in turn) in a fixed order; [visit]
     * returns false to stop. Deals that two placements share are visited twice.
     */
    void enumerate(const std::function<bool(const Hands&)>& visit) const {
        bool stop = false;
        auto enumDom = [&](const Domain& dom) {
            if (two_seat_) {
                const auto tab = buildWays2(dom);
                Hands cur{};
                std::function<void(std::size_t, int)> dfs = [&](std::size_t i, int a) {
                    if (stop) return;
                    if (i == cards_.size()) {
                        if (!visit(cur)) stop = true;
                        return;
                    }
                    const int b = int(cards_.size() - i) - a;
                    const std::uint8_t d = dom[i];
                    const std::uint64_t bit_c = std::uint64_t(1) << cards_[i];
                    if ((d & 1) && a > 0 && ways2At(tab, i + 1, a - 1) > 0) {
                        cur[0] |= bit_c;
                        dfs(i + 1, a - 1);
                        cur[0] &= ~bit_c;
                    }
                    if ((d & 2) && b > 0 && ways2At(tab, i + 1, a) > 0) {
                        cur[1] |= bit_c;
                        dfs(i + 1, a);
                        cur[1] &= ~bit_c;
                    }
                };
                if (ways2At(tab, 0, need_[0]) > 0) dfs(0, need_[0]);
                return;
            }
            const auto tab = buildWays(dom);
            Hands cur{};
            std::function<void(std::size_t, int, int)> dfs = [&](std::size_t i, int a, int b) {
                if (stop) return;
                if (i == cards_.size()) {
                    if (!visit(cur)) stop = true;
                    return;
                }
                const int cc = int(cards_.size() - i) - a - b;
                const std::uint8_t d = dom[i];
                const std::uint64_t bit_c = std::uint64_t(1) << cards_[i];
                if ((d & 1) && a > 0 && waysAt(tab, i + 1, a - 1, b) > 0) {
                    cur[0] |= bit_c;
                    dfs(i + 1, a - 1, b);
                    cur[0] &= ~bit_c;
                }
                if ((d & 2) && b > 0 && waysAt(tab, i + 1, a, b - 1) > 0) {
                    cur[1] |= bit_c;
                    dfs(i + 1, a, b - 1);
                    cur[1] &= ~bit_c;
                }
                if ((d & 4) && cc > 0 && waysAt(tab, i + 1, a, b) > 0) {
                    cur[2] |= bit_c;
                    dfs(i + 1, a, b);
                    cur[2] &= ~bit_c;
                }
            };
            if (waysAt(tab, 0, need_[0], need_[1]) > 0) dfs(0, need_[0], need_[1]);
        };
        if (!placements_.empty()) {
            for (const Placement& pl : placements_) {
                if (stop) break;
                enumDom(pl.domain);
            }
        } else {
            enumDom(domain_);
        }
    }

    /** Each hidden card's exact probability of being with each hidden seat (in cards() order). */
    std::vector<std::array<double, 3>> marginals() const {
        const std::size_t n = cards_.size();
        std::vector<std::array<double, 3>> marg(n, std::array<double, 3>{0.0, 0.0, 0.0});
        if (n == 0) return marg;
        auto accumDom = [&](const Domain& dom, double scale) {
            if (two_seat_) {
                const auto bwd = buildWays2(dom);
                const double tot = ways2At(bwd, 0, need_[0]);
                if (tot <= 0.0) return;
                std::vector<double> fwd((n + 1) * kSide, 0.0);
                auto fat = [&](std::size_t i, int a) -> double& { return fwd[i * kSide + std::size_t(a)]; };
                fat(0, need_[0]) = 1.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const std::uint8_t d = dom[i];
                    double s0 = 0.0, s1 = 0.0;
                    for (int a = 0; a <= need_[0]; ++a) {
                        const double fv = fat(i, a);
                        if (fv == 0.0) continue;
                        const int b = int(n - i) - a;
                        if ((d & 1) && a > 0) {
                            const double bv = ways2At(bwd, i + 1, a - 1);
                            if (bv > 0.0) {
                                fat(i + 1, a - 1) += fv;
                                s0 += fv * bv;
                            }
                        }
                        if ((d & 2) && b > 0) {
                            const double bv = ways2At(bwd, i + 1, a);
                            if (bv > 0.0) {
                                fat(i + 1, a) += fv;
                                s1 += fv * bv;
                            }
                        }
                    }
                    const double inv = scale / tot;
                    marg[i][0] += s0 * inv;
                    marg[i][1] += s1 * inv;
                }
                return;
            }
            const auto bwd = buildWays(dom);
            const double tot = waysAt(bwd, 0, need_[0], need_[1]);
            if (tot <= 0.0) return;
            // forward: fwd[i][a][b] = ways to deal cards 0..i-1 leaving capacities (a, b, cc)
            std::vector<double> fwd((n + 1) * kPlane, 0.0);
            auto fat = [&](std::size_t i, int a, int b) -> double& { return fwd[i * kPlane + std::size_t(a * kSide + b)]; };
            fat(0, need_[0], need_[1]) = 1.0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::uint8_t d = dom[i];
                double s0 = 0.0, s1 = 0.0, s2 = 0.0;
                for (int a = 0; a <= need_[0]; ++a) {
                    for (int b = 0; b <= need_[1]; ++b) {
                        const double fv = fat(i, a, b);
                        if (fv == 0.0) continue;
                        const int cc = int(n - i) - a - b;
                        if ((d & 1) && a > 0) {
                            const double bv = waysAt(bwd, i + 1, a - 1, b);
                            if (bv > 0.0) {
                                fat(i + 1, a - 1, b) += fv;
                                s0 += fv * bv;
                            }
                        }
                        if ((d & 2) && b > 0) {
                            const double bv = waysAt(bwd, i + 1, a, b - 1);
                            if (bv > 0.0) {
                                fat(i + 1, a, b - 1) += fv;
                                s1 += fv * bv;
                            }
                        }
                        if ((d & 4) && cc > 0) {
                            const double bv = waysAt(bwd, i + 1, a, b);
                            if (bv > 0.0) {
                                fat(i + 1, a, b) += fv;
                                s2 += fv * bv;
                            }
                        }
                    }
                }
                const double inv = scale / tot;
                marg[i][0] += s0 * inv;
                marg[i][1] += s1 * inv;
                marg[i][2] += s2 * inv;
            }
        };
        if (!placements_.empty() && placement_total_ > 0.0) {
            for (const Placement& pl : placements_) accumDom(pl.domain, pl.weight / placement_total_);
        } else {
            accumDom(domain_, 1.0);
        }
        return marg;
    }

    /** One deal, uniformly over the domain (or the placements); nothing when there is none. */
    std::optional<Hands> sampleUnchecked(std::mt19937_64& rng) const {
        if (two_seat_) {
            const Domain* dom = &domain_;
            const std::vector<double>* table = &ways2_;
            std::vector<double> local;
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (!placements_.empty()) {
                double x = u(rng) * placement_total_;
                std::size_t j = 0;
                while (j + 1 < placements_.size() && x >= placements_[j].weight) x -= placements_[j++].weight;
                dom = &placements_[j].domain;
                local = buildWays2(*dom);
                table = &local;
            }
            int a = need_[0];
            if (ways2At(*table, 0, a) <= 0.0) return std::nullopt;
            Hands h{};
            for (std::size_t i = 0; i < cards_.size(); ++i) {
                const int b = int(cards_.size() - i) - a;
                const std::uint8_t d = (*dom)[i];
                const double w0 = (d & 1) && a > 0 ? ways2At(*table, i + 1, a - 1) : 0.0;
                const double w1 = (d & 2) && b > 0 ? ways2At(*table, i + 1, a) : 0.0;
                const double x = u(rng) * (w0 + w1);
                const int k = (x < w0 || w1 <= 0.0) ? 0 : 1;
                h[std::size_t(k)] |= std::uint64_t(1) << cards_[i];
                if (k == 0) --a;
            }
            return h;
        }
        const Domain* dom = &domain_;
        const std::vector<double>* table = &ways_;
        std::vector<double> local;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (!placements_.empty()) {
            double x = u(rng) * placement_total_;
            std::size_t j = 0;
            while (j + 1 < placements_.size() && x >= placements_[j].weight) x -= placements_[j++].weight;
            dom = &placements_[j].domain;
            local = buildWays(*dom);
            table = &local;
        }
        int a = need_[0], b = need_[1];
        if (waysAt(*table, 0, a, b) <= 0) return std::nullopt;
        Hands h{};
        for (std::size_t i = 0; i < cards_.size(); ++i) {
            const int cc = int(cards_.size() - i) - a - b;
            const std::uint8_t d = (*dom)[i];
            const double w0 = (d & 1) && a > 0 ? waysAt(*table, i + 1, a - 1, b) : 0.0;
            const double w1 = (d & 2) && b > 0 ? waysAt(*table, i + 1, a, b - 1) : 0.0;
            const double w2 = (d & 4) && cc > 0 ? waysAt(*table, i + 1, a, b) : 0.0;
            const double x = u(rng) * (w0 + w1 + w2);
            int k = x < w0 ? 0 : x < w0 + w1 ? 1 : 2;
            if (k == 2 && w2 <= 0) k = w1 > 0 ? 1 : 0;
            h[std::size_t(k)] |= std::uint64_t(1) << cards_[i];
            if (k == 0) --a;
            else if (k == 1) --b;
        }
        return h;
    }

    /**
     * Uniform deals until one is accepted (at most max_tries); the last one
     * drawn when none is, nothing when there is no deal at all.
     */
    std::optional<Hands> sample(std::mt19937_64& rng, const Accept& accept, int max_tries) const {
        std::optional<Hands> last;
        for (int i = 0; i < max_tries; ++i) {
            last = sampleUnchecked(rng);
            if (!last) return std::nullopt;
            if (!accept || accept(*last)) return last;
        }
        return last;  // gave up: an unchecked deal
    }

    /**
     * Deals drawn in proportion to the product of their cards' seat weights
     * (e.g. a belief network's), until one is accepted; falls back to
     * sample() when the weights leave no deal.
     */
    std::optional<Hands> sampleWeighted(std::mt19937_64& rng, const Weights& weights, const Accept& accept,
                                        int max_tries) const {
        if (two_seat_) {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            std::vector<double> pl_weights;
            double pl_tot = 0.0;
            std::vector<double> base_tab;
            if (!placements_.empty()) {
                pl_weights.reserve(placements_.size());
                for (const Placement& pl : placements_) {
                    const auto tab = buildWaysWeighted2(pl.domain, weights);
                    const double w = ways2At(tab, 0, need_[0]);
                    pl_weights.push_back(w);
                    pl_tot += w;
                }
                if (pl_tot <= 0.0) return sample(rng, accept, max_tries);
            } else {
                base_tab = buildWaysWeighted2(domain_, weights);
                if (ways2At(base_tab, 0, need_[0]) <= 0.0) return sample(rng, accept, max_tries);
            }

            std::optional<Hands> last;
            for (int t = 0; t < max_tries; ++t) {
                const Domain* dom = &domain_;
                const std::vector<double>* tab = &base_tab;
                std::vector<double> local;
                if (!placements_.empty()) {
                    double x = u(rng) * pl_tot;
                    std::size_t j = 0;
                    while (j + 1 < placements_.size() && x >= pl_weights[j]) x -= pl_weights[j++];
                    dom = &placements_[j].domain;
                    local = buildWaysWeighted2(*dom, weights);
                    tab = &local;
                }
                int a = need_[0];
                Hands h{};
                for (std::size_t i = 0; i < cards_.size(); ++i) {
                    const int b = int(cards_.size() - i) - a;
                    const std::uint8_t d = (*dom)[i];
                    const auto& cw = weights[i];
                    const double w0 = (d & 1) && a > 0 ? cw[0] * ways2At(*tab, i + 1, a - 1) : 0.0;
                    const double w1 = (d & 2) && b > 0 ? cw[1] * ways2At(*tab, i + 1, a) : 0.0;
                    const double x = u(rng) * (w0 + w1);
                    const int k = (x < w0 || w1 <= 0.0) ? 0 : 1;
                    h[std::size_t(k)] |= std::uint64_t(1) << cards_[i];
                    if (k == 0) --a;
                }
                last = h;
                if (!accept || accept(h)) return h;
            }
            return last;
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> pl_weights;
        double pl_tot = 0.0;
        std::vector<double> base_tab;
        if (!placements_.empty()) {
            pl_weights.reserve(placements_.size());
            for (const Placement& pl : placements_) {
                const auto tab = buildWaysWeighted(pl.domain, weights);
                const double w = waysAt(tab, 0, need_[0], need_[1]);
                pl_weights.push_back(w);
                pl_tot += w;
            }
            if (pl_tot <= 0.0) return sample(rng, accept, max_tries);
        } else {
            base_tab = buildWaysWeighted(domain_, weights);
            if (waysAt(base_tab, 0, need_[0], need_[1]) <= 0.0) return sample(rng, accept, max_tries);
        }

        std::optional<Hands> last;
        for (int t = 0; t < max_tries; ++t) {
            const Domain* dom = &domain_;
            const std::vector<double>* tab = &base_tab;
            std::vector<double> local;
            if (!placements_.empty()) {
                double x = u(rng) * pl_tot;
                std::size_t j = 0;
                while (j + 1 < placements_.size() && x >= pl_weights[j]) x -= pl_weights[j++];
                dom = &placements_[j].domain;
                local = buildWaysWeighted(*dom, weights);
                tab = &local;
            }
            int a = need_[0], b = need_[1];
            Hands h{};
            for (std::size_t i = 0; i < cards_.size(); ++i) {
                const int cc = int(cards_.size() - i) - a - b;
                const std::uint8_t d = (*dom)[i];
                const auto& cw = weights[i];
                const double w0 = (d & 1) && a > 0 ? cw[0] * waysAt(*tab, i + 1, a - 1, b) : 0.0;
                const double w1 = (d & 2) && b > 0 ? cw[1] * waysAt(*tab, i + 1, a, b - 1) : 0.0;
                const double w2 = (d & 4) && cc > 0 ? cw[2] * waysAt(*tab, i + 1, a, b) : 0.0;
                const double x = u(rng) * (w0 + w1 + w2);
                int k = x < w0 ? 0 : x < w0 + w1 ? 1 : 2;
                if (k == 2 && w2 <= 0) k = w1 > 0 ? 1 : 0;
                h[std::size_t(k)] |= std::uint64_t(1) << cards_[i];
                if (k == 0) --a;
                else if (k == 1) --b;
            }
            last = h;
            if (!accept || accept(h)) return h;
        }
        return last;
    }

private:
    std::vector<int> cards_;
    Domain domain_;
    std::array<int, 3> need_{};
    bool two_seat_ = false;
    std::vector<double> ways_;
    std::vector<double> ways2_;
    std::vector<Placement> placements_;
    double placement_total_ = 0;
};

}  // namespace cardbot
