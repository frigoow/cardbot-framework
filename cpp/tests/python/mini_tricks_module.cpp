// The toy game's Python module for the framework's Python tests.
#include "cardbot/python.h"
#include "../mini_tricks.h"

PYBIND11_MODULE(mini_tricks_cpp, m) {
    cardbot::bindBatchEnv<cardbot_test::MiniTricks, cardbot_test::MiniEncoder>(m, "BatchEnv");
    cardbot::bindQGru(m);
}
