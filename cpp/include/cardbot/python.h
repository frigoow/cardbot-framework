#pragma once
// pybind11 bindings a game's Python module uses (docs/DESIGN.md, step 4):
//
//   PYBIND11_MODULE(mygame_cpp, m) {
//       cardbot::bindBatchEnv<MyGame, MyEncoder>(m, "BatchEnv");
//       cardbot::bindQGru(m);
//   }
//
// BatchEnv(num_envs, seed, threads) in Python: .observe() -> dict of numpy
// arrays, .step(action_idx, auto_reset) -> (done, reward[B, 2]), .reset_all(seed),
// and the sizes as attributes (STATE_DIM, PLY_DIM, MAX_PLIES, ACTION_DIM, MAX_LEGAL).

#include "cardbot/env.h"
#include "cardbot/models.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace cardbot {

namespace py = pybind11;

template <class G, class Enc>
void bindBatchEnv(py::module_& m, const char* name) {
    using Env = BatchEnv<G, Enc>;
    py::class_<Env>(m, name)
        .def(py::init([](int num_envs, std::uint64_t seed, int threads) { return new Env(num_envs, seed, threads); }),
             py::arg("num_envs"), py::arg("seed") = 1, py::arg("threads") = 4)
        .def_property_readonly("num_envs", &Env::numEnvs)
        .def_property_readonly_static("STATE_DIM", [](py::object) { return Enc::kStateDim; })
        .def_property_readonly_static("PLY_DIM", [](py::object) { return Enc::kPlyDim; })
        .def_property_readonly_static("MAX_PLIES", [](py::object) { return Enc::kMaxPlies; })
        .def_property_readonly_static("ACTION_DIM", [](py::object) { return Enc::kActionDim; })
        .def_property_readonly_static("MAX_LEGAL", [](py::object) { return Enc::kMaxLegal; })
        .def("reset_all", &Env::resetAll, py::arg("seed"))
        .def("observe", [](const Env& env) {
            const py::ssize_t b = env.numEnvs();
            py::array_t<float> state({b, py::ssize_t(Enc::kStateDim)});
            py::array_t<float> history({b, py::ssize_t(Enc::kMaxPlies), py::ssize_t(Enc::kPlyDim)});
            py::array_t<std::int32_t> n_plies(b);
            py::array_t<float> legal({b, py::ssize_t(Enc::kMaxLegal), py::ssize_t(Enc::kActionDim)});
            py::array_t<std::int32_t> n_legal(b);
            py::array_t<std::int32_t> team(b);
            {
                py::gil_scoped_release release;
                env.observe(state.mutable_data(), history.mutable_data(), n_plies.mutable_data(), legal.mutable_data(),
                            n_legal.mutable_data(), team.mutable_data());
            }
            py::dict d;
            d["state"] = state;
            d["history"] = history;
            d["n_plies"] = n_plies;
            d["legal_actions"] = legal;
            d["n_legal"] = n_legal;
            d["team"] = team;
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

}  // namespace cardbot
