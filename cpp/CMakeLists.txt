cmake_minimum_required(VERSION 3.22)
project(cardbot LANGUAGES CXX)

# The game-independent C++ of the bots (docs/DESIGN.md), for a game's engine:
#   add_subdirectory(path/to/cardbot-framework/cpp cardbot)
#   target_link_libraries(my_engine PUBLIC cardbot)
# Header-only so far: the network kernels must inline into each model's code
# to keep results bit-identical with the per-game copies they replaced.
#
# Its own tests:  cmake -B build -S cpp && cmake --build build && ctest --test-dir build

add_library(cardbot INTERFACE)
target_include_directories(cardbot INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ismcsolver/include)
find_package(Threads REQUIRED)
target_link_libraries(cardbot INTERFACE Threads::Threads)
target_compile_features(cardbot INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_executable(cardbot_test_nn tests/test_nn.cpp)
    target_link_libraries(cardbot_test_nn PRIVATE cardbot)
    add_test(NAME nn COMMAND cardbot_test_nn)
    add_executable(cardbot_test_sampler tests/test_sampler.cpp)
    target_link_libraries(cardbot_test_sampler PRIVATE cardbot)
    add_test(NAME sampler COMMAND cardbot_test_sampler)
    add_executable(cardbot_test_agents tests/test_agents.cpp)
    target_link_libraries(cardbot_test_agents PRIVATE cardbot)
    add_test(NAME agents COMMAND cardbot_test_agents)
    add_executable(cardbot_test_bidding tests/test_bidding.cpp)
    target_link_libraries(cardbot_test_bidding PRIVATE cardbot)
    add_test(NAME bidding COMMAND cardbot_test_bidding)
    add_executable(cardbot_test_endgame tests/test_endgame.cpp)
    target_link_libraries(cardbot_test_endgame PRIVATE cardbot)
    add_test(NAME endgame COMMAND cardbot_test_endgame)
endif()

# The Python test module (the toy game's BatchEnv and the Q-GRU loader):
#   cmake -B build -S cpp -DCARDBOT_PYTHON=ON -Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
option(CARDBOT_PYTHON "Build the toy game's Python module for python/tests" OFF)
if(CARDBOT_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(mini_tricks_cpp tests/python/mini_tricks_module.cpp)
    target_link_libraries(mini_tricks_cpp PRIVATE cardbot)
    target_include_directories(mini_tricks_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

