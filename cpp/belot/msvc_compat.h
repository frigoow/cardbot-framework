#pragma once
// MSVC lacks the GCC/Clang builtins the engine uses outside belote/card.h's
// own MSVC branches (engines/belote cpp/bg/bg.cpp: __builtin_ctz). Force-included
// into the engine's sources on MSVC only (CMakeLists.txt), so the pinned
// engine builds unchanged.
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int __builtin_ctz(unsigned x) {
    unsigned long index;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
}
#endif
