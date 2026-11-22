#pragma once
// The building blocks of the bots' networks in plain C++: linear layers,
// LayerNorm, residual blocks, a GRU, the activations and the dot product /
// axpy kernels (AVX2 + FMA when compiled for it, plain loops otherwise), and
// the readers of the binary model files' float32 little-endian layers.
//
// Header-only and inline on purpose: each model's forward pass inlines these
// exactly as the per-game copies they replace did, so results stay
// bit-identical. Weights are row-major [out, in], as torch.nn.Linear stores
// them; the GRU follows torch.nn.GRU's gate order (r, z, n).

#include <cmath>
#include <cstddef>
#include <cstring>
#include <istream>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace cardtable::nn {

struct Linear {
    int in_dim = 0, out_dim = 0;
    std::vector<float> w;  // [out_dim, in_dim]
    std::vector<float> b;  // [out_dim]
};

struct LayerNorm {
    int dim = 0;
    std::vector<float> w;
    std::vector<float> b;
};

/** out = silu(x + ln2(fc2(silu(ln1(fc1(x)))))) (applyResBlock). */
struct ResBlock {
    Linear fc1, fc2;
    LayerNorm ln1, ln2;
};

struct GruLayer {
    int in_dim = 0, hidden = 0;
    std::vector<float> w_ih;       // [3*hidden, in_dim]
    std::vector<float> w_hh;       // [3*hidden, hidden]
    std::vector<float> b_ih;       // [3*hidden]
    std::vector<float> b_hh;       // [3*hidden]
    std::vector<float> empty_emb;  // [hidden]: the state of an empty sequence
};


inline float dotProduct(const float* a, const float* b, int n) {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    float sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]);
#else
    float sum = 0.0f;
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline void axpy(float alpha, const float* x, float* y, int n) {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 va = _mm256_set1_ps(alpha);
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
        _mm256_storeu_ps(y + i + 8, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; ++i) y[i] += alpha * x[i];
}

inline bool readBytes(std::istream& in, void* dst, std::size_t n) {
    in.read(reinterpret_cast<char*>(dst), std::streamsize(n));
    return bool(in);
}

inline bool readLinear(std::istream& in, Linear& l, int in_dim, int out_dim) {
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    l.w.resize(std::size_t(out_dim) * std::size_t(in_dim));
    l.b.resize(std::size_t(out_dim));
    return readBytes(in, l.w.data(), l.w.size() * sizeof(float)) &&
           readBytes(in, l.b.data(), l.b.size() * sizeof(float));
}

inline bool readLN(std::istream& in, LayerNorm& ln, int dim) {
    ln.dim = dim;
    ln.w.resize(std::size_t(dim));
    ln.b.resize(std::size_t(dim));
    return readBytes(in, ln.w.data(), ln.w.size() * sizeof(float)) &&
           readBytes(in, ln.b.data(), ln.b.size() * sizeof(float));
}

inline bool readResBlock(std::istream& in, ResBlock& rb, int dim) {
    return readLinear(in, rb.fc1, dim, dim) && readLN(in, rb.ln1, dim) &&
           readLinear(in, rb.fc2, dim, dim) && readLN(in, rb.ln2, dim);
}

inline bool readGru(std::istream& in, GruLayer& g, int in_dim, int hidden) {
    g.in_dim = in_dim;
    g.hidden = hidden;
    g.w_ih.resize(std::size_t(3 * hidden) * std::size_t(in_dim));
    g.w_hh.resize(std::size_t(3 * hidden) * std::size_t(hidden));
    g.b_ih.resize(std::size_t(3 * hidden));
    g.b_hh.resize(std::size_t(3 * hidden));
    g.empty_emb.resize(std::size_t(hidden));
    return readBytes(in, g.w_ih.data(), g.w_ih.size() * sizeof(float)) &&
           readBytes(in, g.w_hh.data(), g.w_hh.size() * sizeof(float)) &&
           readBytes(in, g.b_ih.data(), g.b_ih.size() * sizeof(float)) &&
           readBytes(in, g.b_hh.data(), g.b_hh.size() * sizeof(float)) &&
           readBytes(in, g.empty_emb.data(), g.empty_emb.size() * sizeof(float));
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline void applyLinear(const Linear& l, const float* x, float* y) {
    const int in_d = l.in_dim;
    const int out_d = l.out_dim;
    const float* w = l.w.data();
    const float* b = l.b.data();
    for (int o = 0; o < out_d; ++o) {
        const float* row = w + std::size_t(o) * std::size_t(in_d);
        y[o] = b[o] + dotProduct(row, x, in_d);
    }
}

inline void applyLN(const LayerNorm& ln, float* x) {
    const int d = ln.dim;
    float mean = 0.0f;
    for (int i = 0; i < d; ++i) mean += x[i];
    mean /= float(d);
    float var = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= float(d);
    const float inv = 1.0f / std::sqrt(var + 1e-5f);
    const float* w = ln.w.data();
    const float* b = ln.b.data();
    for (int i = 0; i < d; ++i) x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

inline void applyResBlock(const ResBlock& rb, const float* x, float* out, std::vector<float>& scratch) {
    const int d = rb.fc1.out_dim;
    scratch.resize(std::size_t(d));
    applyLinear(rb.fc1, x, scratch.data());
    applyLN(rb.ln1, scratch.data());
    for (int i = 0; i < d; ++i) scratch[std::size_t(i)] = silu(scratch[std::size_t(i)]);
    applyLinear(rb.fc2, scratch.data(), out);
    applyLN(rb.ln2, out);
    for (int i = 0; i < d; ++i) out[i] = silu(x[i] + out[i]);
}

inline void applyGru(const GruLayer& g, const float* history, int n_plies, float* h_out) {
    const int H = g.hidden;
    const int I = g.in_dim;
    if (n_plies <= 0) {
        std::memcpy(h_out, g.empty_emb.data(), std::size_t(H) * sizeof(float));
        return;
    }
    std::memset(h_out, 0, std::size_t(H) * sizeof(float));
    thread_local std::vector<float> gi;
    thread_local std::vector<float> gh;
    gi.resize(std::size_t(3 * H));
    gh.resize(std::size_t(3 * H));

    for (int t = 0; t < n_plies; ++t) {
        const float* x = history + std::size_t(t * I);
        for (int o = 0; o < 3 * H; ++o) {
            const float* wir = g.w_ih.data() + std::size_t(o) * std::size_t(I);
            gi[std::size_t(o)] = g.b_ih[std::size_t(o)] + dotProduct(wir, x, I);

            const float* whr = g.w_hh.data() + std::size_t(o) * std::size_t(H);
            gh[std::size_t(o)] = g.b_hh[std::size_t(o)] + dotProduct(whr, h_out, H);
        }
        for (int j = 0; j < H; ++j) {
            const float r = sigmoid(gi[std::size_t(j)] + gh[std::size_t(j)]);
            const float z = sigmoid(gi[std::size_t(H + j)] + gh[std::size_t(H + j)]);
            const float n = std::tanh(gi[std::size_t(2 * H + j)] + r * gh[std::size_t(2 * H + j)]);
            h_out[j] = (1.0f - z) * n + z * h_out[j];
        }
    }
}

}  // namespace cardtable::nn
