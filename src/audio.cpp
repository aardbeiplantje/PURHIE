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
