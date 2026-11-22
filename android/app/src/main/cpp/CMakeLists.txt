cmake_minimum_required(VERSION 3.22.1)
project(manille_android LANGUAGES CXX)

add_library(manille_mobile SHARED
    ../../../../../engines/manille/cpp/jni/jni_bridge.cpp
    ../../../../../engines/manille/cpp/mobile/manille_mobile.cpp
    ../../../../../engines/manille/cpp/manille_game.cpp
    ../../../../../engines/manille/cpp/encoding.cpp
    ../../../../../engines/manille/cpp/rule_agent.cpp
    ../../../../../engines/manille/cpp/rule_playout.cpp
    ../../../../../engines/manille/cpp/fast_policy.cpp
    ../../../../../engines/manille/cpp/prior/trump_rule.cpp
    ../../../../../engines/manille/cpp/sampler/uniform.cpp
    ../../../../../engines/manille/cpp/sampler/trump_constrained.cpp
)

target_compile_features(manille_mobile PRIVATE cxx_std_17)
target_compile_options(manille_mobile PRIVATE -O3)
target_include_directories(manille_mobile PRIVATE
    ../../../../../engines/manille/cpp
    ../../../../../engines/manille/third_party/ismcsolver/include
)

find_library(log-lib log)
target_link_libraries(manille_mobile PRIVATE ${log-lib})
