// The toy game's Python module for the framework's Python tests.
#include "cardbot/eval.h"
#include "cardbot/policy.h"
#include "cardbot/python.h"
#include "../mini_tricks.h"

#include <memory>
#include <string>

namespace {

using G = cardbot_test::MiniTricks;
using Enc = cardbot_test::MiniEncoder;

std::shared_ptr<const cardbot::FastPolicyModel> loadPolicy(const std::string& path) {
    auto m = std::make_shared<cardbot::FastPolicyModel>();
    if (!m->load(path, "TEST", -1, Enc::kStateDim, cardbot::policyDim<Enc>())) throw std::runtime_error("cannot load " + path);
    return m;
}

// "random", "ismcts:N", "policy:PATH", "rollout:W" (random play-outs), "rollout:W:PATH" (fast-policy play-outs)
cardbot::Agent<G> makeAgent(const std::string& spec) {
    if (spec == "random") return cardbot::randomAgent<G>();
    if (spec.rfind("ismcts:", 0) == 0) {
        cardbot::IsmctsOptions o;
        o.iterations = std::stoul(spec.substr(7));
        o.scale = 0.25;
        return cardbot::ismctsAgent<G>(o);
    }
    if (spec.rfind("policy:", 0) == 0) return cardbot::fastPolicyAgent<G, Enc>(loadPolicy(spec.substr(7)));
    if (spec.rfind("rollout:", 0) == 0) {
        const std::string rest = spec.substr(8);
        const auto colon = rest.find(':');
        cardbot::RolloutOptions o;
        o.worlds = std::stoi(rest.substr(0, colon));
        const auto playout = colon == std::string::npos ? cardbot::randomAgent<G>()
                                                        : cardbot::fastPolicyAgent<G, Enc>(loadPolicy(rest.substr(colon + 1)));
        return cardbot::rolloutAgent<G>(playout, o);
    }
    throw std::invalid_argument("unknown agent " + spec);
}

}  // namespace

PYBIND11_MODULE(mini_tricks_cpp, m) {
    cardbot::bindBatchEnv<G, Enc>(m, "BatchEnv");
    cardbot::bindQGru(m);
    cardbot::bindFastPolicy(m);
    // a duplicate-deal duel between two agents given by spec: (mean, ci95) in tricks per deal for A
    m.def("duel", [](const std::string& a, const std::string& b, int pairs, std::uint64_t seed, int threads) {
        const auto agent_a = makeAgent(a), agent_b = makeAgent(b);
        pybind11::gil_scoped_release release;
        const auto r = cardbot::duel<G>([](std::uint64_t s) { return G::deal(s); }, agent_a, agent_b, pairs, seed,
                                        threads);
        pybind11::gil_scoped_acquire acquire;
        return pybind11::make_tuple(r.mean, r.ci95);
    }, pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("pairs"), pybind11::arg("seed") = 1,
       pybind11::arg("threads") = 4);
}
