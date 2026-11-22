cmake_minimum_required(VERSION 3.22)
project(cardtable_engine LANGUAGES CXX)

# The game-independent C++ of the bots, for any game's engine to build on:
#   add_subdirectory(path/to/cardtable/engine cardtable_engine)
#   target_link_libraries(my_engine PUBLIC cardtable_engine)
# Header-only for now (the network kernels must inline into each model's
# code to keep results bit-identical with the per-game copies they replace).
#
# Its own tests:  cmake -B build -S engine && cmake --build build && ctest --test-dir build

add_library(cardtable_engine INTERFACE)
target_include_directories(cardtable_engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(cardtable_engine INTERFACE cxx_std_17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_executable(cardtable_test_nn tests/test_nn.cpp)
    target_link_libraries(cardtable_test_nn PRIVATE cardtable_engine)
    add_test(NAME nn COMMAND cardtable_test_nn)
endif()
