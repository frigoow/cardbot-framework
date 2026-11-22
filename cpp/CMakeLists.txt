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
target_include_directories(cardbot INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cardbot INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_executable(cardbot_test_nn tests/test_nn.cpp)
    target_link_libraries(cardbot_test_nn PRIVATE cardbot)
    add_test(NAME nn COMMAND cardbot_test_nn)
endif()
