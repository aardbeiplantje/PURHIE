// Audio in the engine (audio.h): the WAV decode and the band-limited resampler: a windowed sinc (Kaiser window, the constants
// below), each output's taps normalized to unit sum (unit gain at DC), the edges repeating the first and last sample.
// Against librosa 0.11's soxr_hq (tests/audio_verify.cpp, the reference in /data/audio-ref/resample): the largest difference
// is about 5e-4 of the peak, the rest is soxr's own polyphase design.
#include "audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_error;

// the filter: the passband to 0.913 of the Nyquist rate and the stop band at it, as soxr's high quality (librosa's default)
// is designed: the cutoff at 0.9565 of the lower Nyquist, 90 zero crossings, a Kaiser beta of 12.4 (about 120 dB)
constexpr double kCutoff = 0.9565, kZeroCrossings = 90.0, kBeta = 12.4;
constexpr int kWinTab = 8192;

// the modified Bessel function I0 (its power series, enough terms for the Kaiser window's arguments)
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 200; ++k) {
        term *= q / ((double)k * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

}  // namespace

int audio_resample(const float* x, size_t n, int from_rate, int to_rate, float** out, size_t* m) {
    if (!out || !m || (n && !x) || from_rate <= 0 || to_rate <= 0) { g_error = "audio_resample: bad arguments"; return -1; }
    *out = nullptr;
    *m = 0;
    const size_t len = (size_t)std::ceil((double)n * to_rate / from_rate);
    float* buf = static_cast<float*>(std::malloc(sizeof(float) * std::max<size_t>(len, 1)));
    if (!buf) { g_error = "audio_resample: no memory"; return -1; }
    if (from_rate == to_rate) {
        std::memcpy(buf, x, sizeof(float) * n);
        *out = buf;
        *m = n;
        return 0;
    }
    const double ratio = (double)to_rate / from_rate;              // output samples per input sample
    const double fc = kCutoff * 0.5 * std::min(1.0, ratio);         // cutoff, in cycles per input sample
    const double hw_d = kZeroCrossings / (2.0 * fc);                // half width, input samples
    const int hw = (int)std::ceil(hw_d);
    // the Kaiser window over u in [-1, 1], tabulated (linear interpolation: the error is far below the float32 rounding)
    std::vector<double> win_tab(kWinTab + 1);
    const double i0b = bessel_i0(kBeta);
    for (int i = 0; i <= kWinTab; ++i) {
        const double u = (double)i / kWinTab;
        win_tab[i] = bessel_i0(kBeta * std::sqrt(std::max(0.0, 1.0 - u * u))) / i0b;
    }
    auto window = [&](double u) {
        const double a = std::fabs(u) * kWinTab;
        const int i = (int)a;
        if (i >= kWinTab) return 0.0;
        const double f = a - i;
        return win_tab[i] + (win_tab[i + 1] - win_tab[i]) * f;
    };
    const long nn = (long)n;
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < (long)len; ++j) {
        const double pos = (double)j / ratio;                       // the output's position in input samples
        const long i0 = (long)std::floor(pos);
        double sum = 0.0, acc = 0.0;
        for (int k = 0; k < 2 * hw; ++k) {
            const long idx = i0 - hw + 1 + k;
            const double t = (double)idx - pos;                     // the tap's distance to the output position
            const double u = t / hw_d;
            if (std::fabs(u) >= 1.0) continue;
            const double z = 2.0 * fc * t;
            const double s = std::fabs(z) < 1e-12 ? 1.0 : std::sin(M_PI * z) / (M_PI * z);
            const double w = 2.0 * fc * s * window(u);
            sum += w;
            acc += w * x[std::min<long>(std::max<long>(idx, 0), nn - 1)];  // edges repeat the end sample
        }
        buf[j] = (float)(sum != 0.0 ? acc / sum : 0.0);
    }
    *out = buf;
    *m = len;
    return 0;
}

int audio_decode_wav(const unsigned char* d, size_t n, int to_rate, float** samples_out, size_t* count_out) {
    if (!d || !samples_out || !count_out || to_rate <= 0) { g_error = "audio_decode_wav: bad arguments"; return -1; }
    *samples_out = nullptr;
    *count_out = 0;
    auto u16 = [&](size_t p) { return (unsigned)(d[p] | (d[p + 1] << 8)); };
    auto u32 = [&](size_t p) { return (unsigned long)(d[p] | (d[p + 1] << 8) | (d[p + 2] << 16)) | ((unsigned long)d[p + 3] << 24); };
    if (n < 12 || std::memcmp(d, "RIFF", 4) != 0 || std::memcmp(d + 8, "WAVE", 4) != 0) {
        g_error = "not a WAV file (RIFF WAVE)";
        return -1;
    }
    bool have_fmt = false;
    unsigned fmt_tag = 0, channels = 0, bits = 0;
    unsigned long rate = 0;
    size_t data_at = 0, data_n = 0;
    bool have_data = false;
    size_t p = 12;
    while (p + 8 <= n) {
        const unsigned long size = u32(p + 4);
        const size_t body = p + 8;
        const size_t avail = std::min<size_t>(size, n - body);
        if (std::memcmp(d + p, "fmt ", 4) == 0 && avail >= 16) {
            have_fmt = true;
            fmt_tag = u16(body);
            channels = u16(body + 2);
            rate = u32(body + 4);
            bits = u16(body + 14);
            if (fmt_tag == 0xFFFE && avail >= 26) fmt_tag = u16(body + 24);  // WAVE_FORMAT_EXTENSIBLE: the sub-format
        } else if (std::memcmp(d + p, "data", 4) == 0) {
            have_data = true;
            data_at = body;
            data_n = avail;
            break;  // (the samples are the last thing that matters)
        }
        p = body + size + (size & 1);
    }
    if (!have_fmt || !have_data) { g_error = "WAV: no fmt or data chunk"; return -1; }
    const bool pcm = fmt_tag == 1, flt = fmt_tag == 3;
    if (!(pcm && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) && !(flt && bits == 32)) {
        g_error = "WAV: only PCM 8, 16, 24, 32-bit or 32-bit float";
        return -1;
    }
    if (channels == 0 || rate == 0) { g_error = "WAV: no channels or rate"; return -1; }
    const size_t bps = bits / 8, frame = bps * channels;
    const size_t frames = data_n / frame;
    std::vector<float> mono(frames);
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0.0;
        for (unsigned c = 0; c < channels; ++c) {
            const unsigned char* s = d + data_at + i * frame + c * bps;
            double v;
            if (flt) {
                float f;
                std::memcpy(&f, s, 4);
                v = f;
            } else if (bits == 8) {
                v = ((int)s[0] - 128) / 128.0;
            } else if (bits == 16) {
                v = (int16_t)(s[0] | (s[1] << 8)) / 32768.0;
            } else if (bits == 24) {
                const int32_t x = (int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24);
                v = (x >> 8) / 8388608.0;
            } else {
                const int32_t x = (int32_t)((uint32_t)s[0] | (uint32_t)s[1] << 8 | (uint32_t)s[2] << 16 | (uint32_t)s[3] << 24);
                v = x / 2147483648.0;
            }
            acc += v;
        }
        mono[i] = (float)(acc / channels);
    }
    float* out = nullptr;
    size_t m = 0;
    if (audio_resample(mono.data(), frames, (int)rate, to_rate, &out, &m) != 0) return -1;
    *samples_out = out;
    *count_out = m;
    return 0;
}

void audio_free(float* samples) { std::free(samples); }

const char* audio_last_error(void) { return g_error.c_str(); }

// ---- FLAC (audio_flac16): the 16-bit mono stream of float samples (clipped to [-1, 1], rounded to nearest even, the same
// samples as the WAV writer). Frames of 4096 samples: a fixed predictor of order 0 to 3 (the cheapest one for the frame)
// with Rice-coded residuals in one partition. The stream is lossless: a FLAC decoder returns the same 16-bit samples.
namespace {

struct BitWriter {
    std::vector<uint8_t>* out;
    uint64_t acc = 0;
    int nacc = 0;
    explicit BitWriter(std::vector<uint8_t>* o) : out(o) {}
    void put(uint64_t v, int n) {  // the low n bits of v, n <= 32
        if (n == 0) return;
        acc = (acc << n) | (v & ((1ull << n) - 1));
        nacc += n;
        while (nacc >= 8) {
            nacc -= 8;
            out->push_back((uint8_t)(acc >> nacc));
        }
        acc &= (1ull << nacc) - 1;
    }
    void zeros_then_one(uint32_t q) {  // q zero bits, then a one (the Rice quotient)
        while (q >= 31) { put(0, 31); q -= 31; }
        put(1, (int)q + 1);
    }
    void align() {
        if (nacc) put(0, 8 - nacc);
    }
};

uint8_t flac_crc8(const uint8_t* d, size_t n) {
    uint8_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c ^= d[i];
        for (int b = 0; b < 8; ++b) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
    }
    return c;
}

uint16_t flac_crc16(const uint8_t* d, size_t n) {
    uint16_t c = 0;
    for (size_t i = 0; i < n; ++i) {
        c ^= (uint16_t)((uint16_t)d[i] << 8);
        for (int b = 0; b < 8; ++b) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x8005) : (uint16_t)(c << 1);
    }
    return c;
}

// the frame number as UTF-8 (up to 36 bits, the FLAC frame header's coding)
void put_utf8_number(BitWriter& bw, uint64_t v) {
    auto cap = [](int nb) { return nb == 1 ? 7 : (7 - nb) + 6 * (nb - 1); };  // the value bits of an nb-byte code
    int nb = 1;
    while (nb < 7 && v >= (1ull << cap(nb))) ++nb;
    if (nb == 1) { bw.put(v, 8); return; }
    bw.put(((0xFFu << (8 - nb)) & 0xFFu) | (uint32_t)(v >> (6 * (nb - 1))), 8);
    for (int i = nb - 2; i >= 0; --i) bw.put(0x80 | ((v >> (6 * i)) & 0x3F), 8);
}

// the Rice cost of residuals u (zigzagged) at parameter k, and the best k in 0..14 (4-bit parameter)
uint64_t rice_cost(const std::vector<uint32_t>& u, int k) {
    uint64_t bits = 0;
    for (uint32_t v : u) bits += (v >> k) + 1 + (uint64_t)k;
    return bits;
}

}  // namespace

int audio_flac16(const float* x, size_t n, int rate, unsigned char** out_bytes, size_t* out_n) {
    if (!out_bytes || !out_n || (!x && n)) { g_error = "audio_flac16: NULL argument"; return -1; }
    if (rate <= 0 || rate >= (1 << 20)) { g_error = "audio_flac16: the sample rate must be 1 to 2^20 - 1"; return -1; }
    std::vector<int32_t> s(n);
    for (size_t i = 0; i < n; ++i) {
        const double v = x[i] > 1.0 ? 1.0 : x[i] < -1.0 ? -1.0 : (double)x[i];  // (double, as the WAV writer's %.0f)
        s[i] = (int32_t)std::nearbyint(v * 32767.0);
    }
    const size_t kBlock = 4096;
    std::vector<uint8_t> out;
    out.reserve(n * 2 + 1024);
    // "fLaC", then the STREAMINFO block (the last metadata block)
    out.insert(out.end(), {'f', 'L', 'a', 'C'});
    const size_t last_block = n == 0 ? kBlock : (n % kBlock ? n % kBlock : kBlock);
    const size_t min_block = n > kBlock ? std::min(kBlock, last_block) : (n ? n : kBlock);
    {
        std::vector<uint8_t> si;
        BitWriter bw(&si);
        bw.put(1, 1);  // last
        bw.put(0, 7);  // STREAMINFO
        bw.put(34, 24);
        bw.put(min_block < 16 ? 16 : min_block, 16);
        bw.put(kBlock, 16);
        bw.put(0, 24);  // min frame size: unknown
        bw.put(0, 24);  // max frame size: unknown
        bw.put((uint64_t)rate, 20);
        bw.put(0, 3);   // channels - 1
        bw.put(15, 5);  // bits per sample - 1
        bw.put((uint64_t)n, 36);
        for (int i = 0; i < 16; ++i) bw.put(0, 8);  // MD5: not computed (the decoders accept zeros)
        out.insert(out.end(), si.begin(), si.end());
    }
    for (size_t start = 0, f = 0; start < n; start += kBlock, ++f) {
        const size_t len = std::min(kBlock, n - start);
        const int32_t* b = s.data() + start;
        // the fixed predictors: the residual of order p, and the cheapest one
        int best_p = 0;
        uint64_t best_bits = ~0ull;
        std::vector<uint32_t> best_u;
        int best_k = 0;
        for (int p = 0; p <= 3; ++p) {
            if ((size_t)p > len) break;
            std::vector<uint32_t> u;
            u.reserve(len);
            for (size_t j = p; j < len; ++j) {
                int64_t pred = 0;
                if (p == 1) pred = b[j - 1];
                else if (p == 2) pred = 2 * (int64_t)b[j - 1] - b[j - 2];
                else if (p == 3) pred = 3 * (int64_t)b[j - 1] - 3 * (int64_t)b[j - 2] + b[j - 3];
                const int64_t r = (int64_t)b[j] - pred;
                u.push_back((uint32_t)((r << 1) ^ (r >> 63)));
            }
            int k_best = 0;
            uint64_t c_best = ~0ull;
            for (int k = 0; k <= 14; ++k) {
                const uint64_t c = rice_cost(u, k);
                if (c < c_best) { c_best = c; k_best = k; }
            }
            const uint64_t bits = c_best + (uint64_t)p * 16;
            if (bits < best_bits) { best_bits = bits; best_p = p; best_u = std::move(u); best_k = k_best; }
        }
        // the frame header
        std::vector<uint8_t> fr;
        BitWriter bw(&fr);
        bw.put(0x3FFE, 14);
        bw.put(0, 1);  // reserved
        bw.put(0, 1);  // fixed block size
        const bool full = len == kBlock;
        bw.put(full ? 0xC : 0x7, 4);  // block size: 4096, or 16 bits after the header (len - 1)
        bw.put(0, 4);                 // sample rate: from STREAMINFO
        bw.put(0, 4);                 // mono
        bw.put(4, 3);                 // 16 bits per sample
        bw.put(0, 1);
        put_utf8_number(bw, f);
        if (!full) bw.put(len - 1, 16);
        bw.align();
        fr.push_back(flac_crc8(fr.data(), fr.size()));
        // the subframe: a FIXED one of order best_p, warm-up samples verbatim, Rice residuals (one partition)
        bw.put(0, 1);
        bw.put((uint64_t)(8 + best_p), 6);
        bw.put(0, 1);
        for (int j = 0; j < best_p; ++j) bw.put((uint64_t)(uint16_t)(int16_t)b[j], 16);
        bw.put(0, 2);  // residual coding: RICE (4-bit parameter)
        bw.put(0, 4);  // partition order 0
        bw.put((uint64_t)best_k, 4);
        for (uint32_t v : best_u) {
            bw.zeros_then_one(v >> best_k);
            bw.put(v & ((1u << best_k) - 1), best_k);
        }
        bw.align();
        const uint16_t crc = flac_crc16(fr.data(), fr.size());
        fr.push_back((uint8_t)(crc >> 8));
        fr.push_back((uint8_t)(crc & 0xFF));
        out.insert(out.end(), fr.begin(), fr.end());
    }
    unsigned char* mem = (unsigned char*)std::malloc(out.size());
    if (!mem) { g_error = "audio_flac16: out of memory"; return -1; }
    std::memcpy(mem, out.data(), out.size());
    *out_bytes = mem;
    *out_n = out.size();
    return 0;
}
