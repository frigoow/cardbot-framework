// cardtable/nn.h against a plain double-precision reference of each layer,
// and the binary readers on a layer written the way the exporters write it.
#include "cardtable/nn.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <vector>

using namespace cardtable::nn;

namespace {

int failures = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; } \
    } while (0)

std::mt19937 rng(7);
std::vector<float> randoms(std::size_t n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}
Linear randomLinear(int in, int out) { return {in, out, randoms(std::size_t(in) * std::size_t(out), .3f), randoms(std::size_t(out), .1f)}; }
LayerNorm randomLN(int d) {
    LayerNorm ln{d, randoms(std::size_t(d), .2f), randoms(std::size_t(d), .1f)};
    for (float& w : ln.w) w += 1.0f;
    return ln;
}
bool near(double a, double b, double tol = 1e-4) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

std::vector<double> refLinear(const Linear& l, const std::vector<double>& x) {
    std::vector<double> y(std::size_t(l.out_dim));
    for (int o = 0; o < l.out_dim; ++o) {
        double s = l.b[std::size_t(o)];
        for (int i = 0; i < l.in_dim; ++i) s += double(l.w[std::size_t(o) * std::size_t(l.in_dim) + std::size_t(i)]) * x[std::size_t(i)];
        y[std::size_t(o)] = s;
    }
    return y;
}
void refLN(const LayerNorm& ln, std::vector<double>& x) {
    double mean = 0, var = 0;
    for (double v : x) mean += v;
    mean /= double(x.size());
    for (double v : x) var += (v - mean) * (v - mean);
    var /= double(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = (x[i] - mean) / std::sqrt(var + 1e-5) * ln.w[i] + ln.b[i];
}
double refSilu(double v) { return v / (1 + std::exp(-v)); }
double refSigmoid(double v) { return 1 / (1 + std::exp(-v)); }

void testLinearAndKernels() {
    for (int in : {1, 7, 8, 16, 33, 611}) {
        const Linear l = randomLinear(in, 5);
        const std::vector<float> x = randoms(std::size_t(in));
        std::vector<float> y(5);
        applyLinear(l, x.data(), y.data());
        const std::vector<double> ref = refLinear(l, std::vector<double>(x.begin(), x.end()));
        for (int o = 0; o < 5; ++o) CHECK(near(y[std::size_t(o)], ref[std::size_t(o)]));
        std::vector<float> acc(std::size_t(in), 1.0f);
        axpy(.5f, x.data(), acc.data(), in);
        for (int i = 0; i < in; ++i) CHECK(near(acc[std::size_t(i)], 1.0 + .5 * x[std::size_t(i)]));
    }
}

void testLayerNormAndResBlock() {
    const int d = 24;
    const ResBlock rb{randomLinear(d, d), randomLinear(d, d), randomLN(d), randomLN(d)};
    const std::vector<float> x = randoms(std::size_t(d));
    std::vector<float> out(static_cast<std::size_t>(d)), scratch;
    applyResBlock(rb, x.data(), out.data(), scratch);
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> h = refLinear(rb.fc1, xd);
    refLN(rb.ln1, h);
    for (double& v : h) v = refSilu(v);
    std::vector<double> o = refLinear(rb.fc2, h);
    refLN(rb.ln2, o);
    for (int i = 0; i < d; ++i) CHECK(near(out[std::size_t(i)], refSilu(xd[std::size_t(i)] + o[std::size_t(i)]), 1e-3));
}

void testGru() {
    const int I = 9, H = 12, T = 5;
    GruLayer g{I, H, randoms(std::size_t(3 * H * I), .3f), randoms(std::size_t(3 * H * H), .3f), randoms(std::size_t(3 * H), .1f),
               randoms(std::size_t(3 * H), .1f), randoms(std::size_t(H))};
    const std::vector<float> seq = randoms(std::size_t(T * I));
    std::vector<float> h(static_cast<std::size_t>(H));
    applyGru(g, seq.data(), T, h.data());
    std::vector<double> r(std::size_t(H), 0.0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> next(static_cast<std::size_t>(H));
        for (int j = 0; j < H; ++j) {
            auto gate = [&](int k) {
                double a = g.b_ih[std::size_t(k * H + j)], b = g.b_hh[std::size_t(k * H + j)];
                for (int i = 0; i < I; ++i) a += double(g.w_ih[std::size_t((k * H + j) * I + i)]) * seq[std::size_t(t * I + i)];
                for (int i = 0; i < H; ++i) b += double(g.w_hh[std::size_t((k * H + j) * H + i)]) * r[std::size_t(i)];
                return std::make_pair(a, b);
            };
            const auto [ri, rh] = gate(0);
            const auto [zi, zh] = gate(1);
            const auto [ni, nh] = gate(2);
            const double rr = refSigmoid(ri + rh), z = refSigmoid(zi + zh), n = std::tanh(ni + rr * nh);
            next[std::size_t(j)] = (1 - z) * n + z * r[std::size_t(j)];
        }
        r = next;
    }
    for (int j = 0; j < H; ++j) CHECK(near(h[std::size_t(j)], r[std::size_t(j)], 1e-3));
    // an empty sequence is the learned empty embedding
    applyGru(g, seq.data(), 0, h.data());
    for (int j = 0; j < H; ++j) CHECK(h[std::size_t(j)] == g.empty_emb[std::size_t(j)]);
}

void testReaders() {
    // weights then bias, float32, as the exporters write a Linear
    const Linear l = randomLinear(3, 2);
    std::string bytes(reinterpret_cast<const char*>(l.w.data()), l.w.size() * sizeof(float));
    bytes.append(reinterpret_cast<const char*>(l.b.data()), l.b.size() * sizeof(float));
    std::istringstream in(bytes);
    Linear back;
    CHECK(readLinear(in, back, 3, 2));
    CHECK(back.w == l.w && back.b == l.b && back.in_dim == 3 && back.out_dim == 2);
    std::istringstream cut(bytes.substr(0, 10));
    CHECK(!readLinear(cut, back, 3, 2));
}

}  // namespace

int main() {
    testLinearAndKernels();
    testLayerNormAndResBlock();
    testGru();
    testReaders();
    if (failures) { std::fprintf(stderr, "%d check(s) failed\n", failures); return 1; }
    std::printf("cardtable nn: all checks passed\n");
    return 0;
}
