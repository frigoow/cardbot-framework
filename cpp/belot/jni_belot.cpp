// JNI for com.cardtable.belot.jvm.BelotNative: the C interface of
// belot_api.h, one static native per function. Games are passed as jlong
// handles; 0 / negative results mean an error (BelotNative.lastError()).
#include "belot_api.h"

#include <jni.h>

#include <cstdint>

#include <string>
#include <vector>

namespace {

std::string text(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

template <typename F>
jstring textResult(JNIEnv* env, F call) {
    const int n = call(nullptr, 0);
    if (n < 0) return nullptr;
    std::vector<char> buf(std::size_t(n) + 1);
    call(buf.data(), n + 1);
    return env->NewStringUTF(buf.data());
}

BelotGame* game(jlong h) { return reinterpret_cast<BelotGame*>(h); }

}  // namespace

#define BELOT_JNI(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_cardtable_belot_jvm_BelotNative_##name

BELOT_JNI(jstring, lastError)(JNIEnv* env, jclass) { return env->NewStringUTF(belot_last_error()); }

BELOT_JNI(jint, modelsLoad)(JNIEnv* env, jclass, jstring dir) { return belot_models_load(text(env, dir).c_str()); }

BELOT_JNI(jstring, modelsInfo)(JNIEnv* env, jclass) {
    return textResult(env, [](char* b, int c) { return belot_models_info(b, c); });
}

BELOT_JNI(jlong, create)(JNIEnv*, jclass, jlong seed, jboolean match) {
    return reinterpret_cast<jlong>(belot_new(std::uint64_t(seed), match ? 1 : 0));
}

BELOT_JNI(jlong, copy)(JNIEnv*, jclass, jlong h) { return reinterpret_cast<jlong>(belot_clone(game(h))); }

BELOT_JNI(void, free)(JNIEnv*, jclass, jlong h) { belot_free(game(h)); }

BELOT_JNI(jint, nextDeal)(JNIEnv*, jclass, jlong h) { return belot_next_deal(game(h)); }

BELOT_JNI(jstring, view)(JNIEnv* env, jclass, jlong h, jint seat) {
    return textResult(env, [&](char* b, int c) { return belot_view(game(h), seat, b, c); });
}

BELOT_JNI(jlong, hand)(JNIEnv*, jclass, jlong h, jint seat) {
    int64_t mask = 0;
    return belot_hand(game(h), seat, &mask) < 0 ? -1 : jlong(mask);
}

BELOT_JNI(jint, apply)(JNIEnv*, jclass, jlong h, jint seat, jint action) { return belot_apply(game(h), seat, action); }

BELOT_JNI(jint, botAction)(JNIEnv*, jclass, jlong h, jint seat, jint level, jlong seed) {
    int action = -1;
    return belot_bot_action(game(h), seat, level, std::uint64_t(seed), &action) < 0 ? -1 : action;
}

// [action0, value0, action1, value1, ..., pick], or null on an error.
BELOT_JNI(jdoubleArray, hint)(JNIEnv* env, jclass, jlong h, jint seat, jint level, jlong seed) {
    int actions[64];
    double values[64];
    int pick = -1;
    const int n = belot_hint(game(h), seat, level, std::uint64_t(seed), actions, values, 64, &pick);
    if (n < 0) return nullptr;
    std::vector<double> out;
    for (int i = 0; i < n; ++i) {
        out.push_back(actions[i]);
        out.push_back(values[i]);
    }
    out.push_back(pick);
    jdoubleArray arr = env->NewDoubleArray(jsize(out.size()));
    env->SetDoubleArrayRegion(arr, 0, jsize(out.size()), out.data());
    return arr;
}

BELOT_JNI(jstring, save)(JNIEnv* env, jclass, jlong h) {
    return textResult(env, [&](char* b, int c) { return belot_save(game(h), b, c); });
}

BELOT_JNI(jlong, load)(JNIEnv* env, jclass, jstring s) {
    return reinterpret_cast<jlong>(belot_load(text(env, s).c_str()));
}
