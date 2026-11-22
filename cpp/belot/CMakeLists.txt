cmake_minimum_required(VERSION 3.22)
project(belot_app LANGUAGES CXX)

# Bulgarian Belot for the apps: the engine and bots of engines/belote (cpp/bg
# and the cpp/belote core it builds on) behind belot_api.h.
#
#   cmake -B build -S cpp/belot && cmake --build build -j && ctest --test-dir build
#
# BELOT_JNI=ON adds the JNI library (belot_mobile) the Android and desktop
# apps load; it needs jni.h (JNI_INCLUDE_DIRS, or the NDK on Android). The
# desktop build (android/buildSrc CMakeJniLibrary) collects it from
# BELOT_JNI_OUT.

option(BELOT_JNI "Build the JNI library belot_mobile" OFF)
option(BELOT_TESTS "Build the C++ tests" ON)
set(JNI_INCLUDE_DIRS "" CACHE STRING "Directories with jni.h (desktop builds)")
set(BELOT_JNI_OUT "" CACHE PATH "Where to put belot_mobile (desktop builds)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ENGINE ${CMAKE_CURRENT_SOURCE_DIR}/../../engines/belote/cpp)

add_library(belot_core STATIC
    ${ENGINE}/belote/card.cpp
    ${ENGINE}/belote/toon.cpp
    ${ENGINE}/belote/deal.cpp
    ${ENGINE}/belote/match.cpp
    ${ENGINE}/belote/observation.cpp
    ${ENGINE}/belote/sampler.cpp
    ${ENGINE}/belote/encoder.cpp
    ${ENGINE}/bg/bg.cpp
    ${ENGINE}/bg/view.cpp
    ${ENGINE}/bg/rule_policy.cpp
    ${ENGINE}/bg/agent.cpp
    ${ENGINE}/bg/encoder.cpp
    ${ENGINE}/bg/batch_env.cpp
    ${ENGINE}/bg/neural.cpp
    ${ENGINE}/bg/match_equity.cpp
    ${ENGINE}/bg/bid_model.cpp
    belot_api.cpp
)
target_include_directories(belot_core PUBLIC ${ENGINE} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(belot_core PUBLIC Threads::Threads)
if(MSVC)
    target_compile_options(belot_core PRIVATE /O2 /EHsc "/FI${CMAKE_CURRENT_SOURCE_DIR}/msvc_compat.h")
else()
    target_compile_options(belot_core PRIVATE -O3)
endif()

if(BELOT_JNI)
    add_library(belot_mobile SHARED jni_belot.cpp)
    target_link_libraries(belot_mobile PRIVATE belot_core)
    if(JNI_INCLUDE_DIRS)
        target_include_directories(belot_mobile PRIVATE ${JNI_INCLUDE_DIRS})
    endif()
    if(BELOT_JNI_OUT)
        # one directory for every generator and config (a Windows DLL is a RUNTIME output)
        foreach(kind LIBRARY RUNTIME)
            set_target_properties(belot_mobile PROPERTIES ${kind}_OUTPUT_DIRECTORY ${BELOT_JNI_OUT})
            foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
                set_target_properties(belot_mobile PROPERTIES ${kind}_OUTPUT_DIRECTORY_${config} ${BELOT_JNI_OUT})
            endforeach()
        endforeach()
    endif()
    if(ANDROID)
        find_library(log-lib log)
        target_link_libraries(belot_mobile PRIVATE ${log-lib})
    endif()
endif()

if(BELOT_TESTS AND NOT ANDROID)
    enable_testing()
    add_executable(belot_test_api test_belot_api.cpp)
    target_link_libraries(belot_test_api PRIVATE belot_core)
    add_test(NAME belot_api COMMAND belot_test_api)
endif()
