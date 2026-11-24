#pragma once
// pybind11 bindings a game's Python module uses (docs/DESIGN.md, step 4):
//
//   PYBIND11_MODULE(mygame_cpp, m) {
//       cardbot::bindBatchEnv<MyGame, MyEncoder, MyBidEncoder>(m, "BatchEnv");  // no auction: leave out MyBidEncoder
//       cardbot::bindBidEnv<MyGame, MyBidEncoder, MyEncoder>(m, "BidEnv");      // steps 5 and 6
//       cardbot::bindQGru(m);
//       cardbot::bindFastPolicy(m);
//       cardbot::bindBidModel(m);
//   }
//
// BatchEnv(num_envs, seed, threads) in Python: .observe() -> dict of numpy
// arrays, .step(action_idx, auto_reset) -> (done, reward[B, 2]), .reset_all(seed),
// and the sizes as attributes (STATE_DIM, PLY_DIM, MAX_PLIES, ACTION_DIM, MAX_LEGAL,
// POLICY_DIM). observe()'s legal_slots are the legal actions' fast-policy slots.

#include "cardbot/bidding.h"
#include "cardbot/env.h"
#include "cardbot/models.h"
#include "cardbot/policy.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cardbot {

namespace py = pybind11;

template <class Model>
std::shared_ptr<const Model> loadModel(const std::string& path, const std::string& tag) {
    if (tag.size() != 4) throw std::invalid_argument("tag: 4 bytes");
    auto model = std::make_shared<Model>();
    if (!model->load(path, tag.c_str())) throw std::runtime_error("cannot load " + path);
    return model;
}

/**
 * BatchEnv<G, Enc> as `name`. With a BidEncoder, also use_bid_model(path, tag)
 * (step 6: the auction's decisions by the bid net instead of
 * Enc::autoAction) and use_auto_action() (back to autoAction).
 */
template <class G, class Enc, class BidEnc = void>
void bindBatchEnv(py::module_& m, const char* name) {
    using Env = BatchEnv<G, Enc>;
    auto cls = py::class_<Env>(m, name);
    if constexpr (!std::is_void_v<BidEnc>) {
        cls.def("use_bid_model", [](Env& env, const std::string& path, const std::string& tag) {
            const Enc enc = env.encoder();
            const BidEnc benc;
            const Agent<G> bid = bidModelAgent<G, BidEnc>(loadModel<BidModel>(path, tag), benc);
            env.setAutoAgent([enc, benc, bid](const G& g, std::mt19937_64& rng) {
                return benc.isBid(g) ? bid(g, rng) : enc.autoAction(g, rng);
            });
        }, py::arg("path"), py::arg("tag") = "CBB1");
        cls.def("use_auto_action", [](Env& env) { env.setAutoAgent({}); });
    }
    if constexpr (detail::HasBelief<Enc>::value) {
        cls.def_property_readonly_static("BELIEF_CARDS", [](py::object) { return Enc::kBeliefCards; })
           .def_property_readonly_static("BELIEF_SEATS", [](py::object) { return Enc::kBeliefSeats; })
           .def("set_collect_belief", &Env::setCollectBelief, py::arg("on"))
           .def_property_readonly("collect_belief", &Env::collectBelief);
    }
    cls
        .def(py::init([](int num_envs, std::uint64_t seed, int threads) { return new Env(num_envs, seed, threads); }),
             py::arg("num_envs"), py::arg("seed") = 1, py::arg("threads") = 4)
        .def_property_readonly("num_envs", &Env::numEnvs)
        .def_property_readonly_static("STATE_DIM", [](py::object) { return Enc::kStateDim; })
        .def_property_readonly_static("PLY_DIM", [](py::object) { return Enc::kPlyDim; })
        .def_property_readonly_static("MAX_PLIES", [](py::object) { return Enc::kMaxPlies; })
        .def_property_readonly_static("ACTION_DIM", [](py::object) { return Enc::kActionDim; })
        .def_property_readonly_static("MAX_LEGAL", [](py::object) { return Enc::kMaxLegal; })
        .def_property_readonly_static("POLICY_DIM", [](py::object) { return policyDim<Enc>(); })
        .def("reset_all", &Env::resetAll, py::arg("seed"))
        .def("observe", [](const Env& env) {
            const py::ssize_t b = env.numEnvs();
            py::array_t<float> state({b, py::ssize_t(Enc::kStateDim)});
            py::array_t<float> history({b, py::ssize_t(Enc::kMaxPlies), py::ssize_t(Enc::kPlyDim)});
            py::array_t<std::int32_t> n_plies(b);
            py::array_t<float> legal({b, py::ssize_t(Enc::kMaxLegal), py::ssize_t(Enc::kActionDim)});
            py::array_t<std::int32_t> n_legal(b);
            py::array_t<std::int32_t> team(b);
            py::array_t<std::int32_t> slots({b, py::ssize_t(Enc::kMaxLegal)});
            {
                py::gil_scoped_release release;
                env.observe(state.mutable_data(), history.mutable_data(), n_plies.mutable_data(), legal.mutable_data(),
                            n_legal.mutable_data(), team.mutable_data());
                env.legalSlots(slots.mutable_data());
            }
            py::dict d;
            d["state"] = state;
            d["history"] = history;
            d["n_plies"] = n_plies;
            d["legal_actions"] = legal;
            d["n_legal"] = n_legal;
            d["team"] = team;
            d["legal_slots"] = slots;
            if constexpr (detail::HasBelief<Enc>::value) {
                if (env.collectBelief()) {
                    py::array_t<std::int8_t> bt({b, py::ssize_t(Enc::kBeliefCards)});
                    py::array_t<float> bm({b, py::ssize_t(Enc::kBeliefCards), py::ssize_t(Enc::kBeliefSeats)});
                    {
                        py::gil_scoped_release release;
                        env.observeBelief(bt.mutable_data(), bm.mutable_data());
                    }
                    d["belief_targets"] = bt;
                    d["belief_marginals"] = bm;
                }
            }
            return d;
        })
        .def("step", [](Env& env, py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> action_idx,
                        bool auto_reset) {
            if (action_idx.size() != env.numEnvs()) throw std::invalid_argument("step: one action index per env");
            const py::ssize_t b = env.numEnvs();
            py::array_t<std::uint8_t> done(b);
            py::array_t<float> reward({b, py::ssize_t(2)});
            {
                py::gil_scoped_release release;
                env.step(action_idx.data(), done.mutable_data(), reward.mutable_data(), auto_reset);
            }
            return py::make_tuple(done, reward);
        }, py::arg("action_idx"), py::arg("auto_reset") = true);
}

/**
 * BidEnv<G, BidEnc> as `name` (steps 5 and 6), PlayEnc being the card-play
 * encoder for the fast policy: generate(num_deals, seed, eps, samples) -> dict,
 * use_bid_model(path, tag), use_play_policy(path, tag, temperature),
 * evaluate(a, b, pairs, seed) -> (mean, ci95) with bidders 0 (the bid net, or
 * the baseline before one) and 1 (the baseline), and the sizes as attributes.
 * Tables and other bidders are set in C++ (the game's own binding).
 */
template <class G, class BidEnc, class PlayEnc>
py::class_<BidEnv<G, BidEnc>> bindBidEnv(py::module_& m, const char* name) {
    using Env = BidEnv<G, BidEnc>;
    py::class_<Env> cls(m, name);
    cls.def(py::init([](int threads) { return new Env(threads); }), py::arg("threads") = 4)
        .def_property_readonly_static("FEAT_DIM", [](py::object) { return BidEnc::kFeatDim; })
        .def_property_readonly_static("HIST_DIM", [](py::object) { return BidEnc::kHistDim; })
        .def_property_readonly_static("MAX_HIST", [](py::object) { return BidEnc::kMaxHist; })
        .def_property_readonly_static("NUM_BIDS", [](py::object) { return BidEnc::kNumBids; })
        .def("use_bid_model", [](Env& env, const std::string& path, const std::string& tag) {
            env.setBidModel(loadModel<BidModel>(path, tag));
        }, py::arg("path"), py::arg("tag") = "CBB1")
        .def("use_play_policy", [](Env& env, const std::string& path, const std::string& tag, double temperature) {
            env.setPlayAgent(fastPolicyAgent<G, PlayEnc>(loadModel<FastPolicyModel>(path, tag), PlayEnc(), temperature));
        }, py::arg("path"), py::arg("tag") = "BGP1", py::arg("temperature") = 0.0)
        .def("generate", [](const Env& env, int num_deals, std::uint64_t seed, double eps, int samples) {
            BidBatch b;
            {
                py::gil_scoped_release release;
                b = env.generate(num_deals, seed, eps, samples);
            }
            const py::ssize_t n = b.n;
            auto arr = [](const auto& v, std::vector<py::ssize_t> shape) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                py::array_t<T> a(shape);
                std::copy(v.begin(), v.end(), a.mutable_data());
                return a;
            };
            py::dict d;
            d["feat"] = arr(b.feat, {n, py::ssize_t(BidEnc::kFeatDim)});
            d["hist"] = arr(b.hist, {n, py::ssize_t(BidEnc::kMaxHist), py::ssize_t(BidEnc::kHistDim)});
            d["n_hist"] = arr(b.n_hist, {n});
            d["legal_mask"] = arr(b.legal_mask, {n, py::ssize_t(BidEnc::kNumBids)});
            d["targets"] = arr(b.targets, {n, py::ssize_t(BidEnc::kNumBids)});
            return d;
        }, py::arg("num_deals"), py::arg("seed"), py::arg("eps") = 0.05, py::arg("samples") = 1)
        .def("evaluate", [](const Env& env, int a, int b, int pairs, std::uint64_t seed) {
            DuelResult r;
            {
                py::gil_scoped_release release;
                r = env.evaluate(a, b, pairs, seed);
            }
            return py::make_tuple(r.mean, r.ci95);
        }, py::arg("a"), py::arg("b"), py::arg("pairs"), py::arg("seed") = 1);
    return cls;
}

/** BidModel for Python: load an exported bid net, run the C++ forward pass. */
inline void bindBidModel(py::module_& m) {
    py::class_<BidModel>(m, "BidModel")
        .def(py::init<>())
        .def("load", [](BidModel& b, const std::string& path, const std::string& tag) {
            if (tag.size() != 4) throw std::invalid_argument("tag: 4 bytes");
            return b.load(path, tag.c_str());
        }, py::arg("path"), py::arg("tag") = "CBB1")
        .def("forward", [](const BidModel& b, py::array_t<float, py::array::c_style | py::array::forcecast> feat,
                           py::array_t<float, py::array::c_style | py::array::forcecast> hist, int n_hist) {
            if (feat.size() != b.featDim() || hist.size() < py::ssize_t(n_hist) * b.histDim())
                throw std::invalid_argument("forward: feature or history size");
            py::array_t<float> out(b.numBids());
            b.forward(feat.data(), hist.data(), n_hist, out.mutable_data());
            return out;
        });
}

/** QGruModel for Python: load an exported file and run the C++ forward pass (checks, and C++-side agents). */
inline void bindQGru(py::module_& m) {
    py::class_<QGruModel>(m, "QGruModel")
        .def(py::init<>())
        .def("load", [](QGruModel& q, const std::string& path, const std::string& tag) {
            if (tag.size() != 4) throw std::invalid_argument("tag: 4 bytes");
            return q.load(path, tag.c_str());
        }, py::arg("path"), py::arg("tag") = "BGQ1")
        .def("forward_q", [](const QGruModel& q, py::array_t<float, py::array::c_style | py::array::forcecast> state,
                             py::array_t<float, py::array::c_style | py::array::forcecast> history, int n_plies,
                             py::array_t<float, py::array::c_style | py::array::forcecast> legal, int n_legal) {
            py::array_t<float> out(n_legal);
            q.forwardQ(state.data(), history.data(), n_plies, legal.data(), n_legal, out.mutable_data());
            return out;
        });
}

/** FastPolicyModel for Python: load an exported file, run the C++ forward pass. */
inline void bindFastPolicy(py::module_& m) {
    py::class_<FastPolicyModel>(m, "FastPolicyModel")
        .def(py::init<>())
        .def("load", [](FastPolicyModel& p, const std::string& path, const std::string& tag) {
            if (tag.size() != 4) throw std::invalid_argument("tag: 4 bytes");
            return p.load(path, tag.c_str());
        }, py::arg("path"), py::arg("tag") = "BGP1")
        .def_property_readonly("state_dim", &FastPolicyModel::stateDim)
        .def_property_readonly("out_dim", &FastPolicyModel::outDim)
        .def("forward", [](const FastPolicyModel& p, py::array_t<float, py::array::c_style | py::array::forcecast> state) {
            if (state.size() != p.stateDim()) throw std::invalid_argument("forward: state of stateDim floats");
            py::array_t<float> out(p.outDim());
            p.forward(state.data(), out.mutable_data());
            return out;
        });
}

/** BeliefModel for Python: load an exported BeliefNet and run the C++ forward pass. */
inline void bindBeliefModel(py::module_& m) {
    py::class_<BeliefModel>(m, "BeliefModel")
        .def(py::init<>())
        .def("load", [](BeliefModel& b, const std::string& path, int n_cards, int n_seats, const std::string& tag) {
            if (tag.size() != 4) throw std::invalid_argument("tag: 4 bytes");
            return b.load(path, n_cards, n_seats, tag.c_str());
        }, py::arg("path"), py::arg("n_cards"), py::arg("n_seats") = 3, py::arg("tag") = "BGB1")
        .def_property_readonly("state_dim", &BeliefModel::stateDim)
        .def_property_readonly("ply_dim", &BeliefModel::plyDim)
        .def_property_readonly("num_cards", &BeliefModel::numCards)
        .def_property_readonly("num_seats", &BeliefModel::numSeats)
        .def("forward_logits", [](const BeliefModel& b, py::array_t<float, py::array::c_style | py::array::forcecast> state,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> history, int n_plies,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> marginals) {
            const py::ssize_t nc = b.numCards(), ns = b.numSeats();
            if (state.size() != b.stateDim() || history.size() < py::ssize_t(n_plies) * b.plyDim() ||
                marginals.size() != nc * ns)
                throw std::invalid_argument("forward_logits: input size mismatch");
            py::array_t<float> out({nc, ns});
            b.forwardLogits(state.data(), history.data(), n_plies, marginals.data(), out.mutable_data());
            return out;
        })
        .def("forward_probs", [](const BeliefModel& b, py::array_t<float, py::array::c_style | py::array::forcecast> state,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> history, int n_plies,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> marginals, float temperature) {
            const py::ssize_t nc = b.numCards(), ns = b.numSeats();
            if (state.size() != b.stateDim() || history.size() < py::ssize_t(n_plies) * b.plyDim() ||
                marginals.size() != nc * ns)
                throw std::invalid_argument("forward_probs: input size mismatch");
            py::array_t<float> out({nc, ns});
            b.forwardProbs(state.data(), history.data(), n_plies, marginals.data(), out.mutable_data(), temperature);
            return out;
        }, py::arg("state"), py::arg("history"), py::arg("n_plies"), py::arg("marginals"), py::arg("temperature") = 1.0f);
}

}  // namespace cardbot
