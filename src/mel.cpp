// Whisper's log-mel features (mel.h), the order of operations of transformers' WhisperFeatureExtractor (5.5.1): the
// frames' DFT and power, the filters as mel_filter_bank builds them, the log10 with its floor, the clamp, the scale.
// Everything is in double (the reference, transformers' torch path, runs float32: about 1e-5 from the double result).
#include "mel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Slaney's mel scale: linear below 1 kHz (3 mel per 200 Hz), logarithmic above (15 mel at 1 kHz)
double hz_to_mel(double f) {
    const double min_log_hz = 1000.0, min_log_mel = 15.0;
    const double logstep = 27.0 / std::log(6.4);
    if (f >= min_log_hz) return min_log_mel + std::log(f / min_log_hz) * logstep;
    return 3.0 * f / 200.0;
}

double mel_to_hz(double m) {
    const double min_log_hz = 1000.0, min_log_mel = 15.0;
    const double logstep = std::log(6.4) / 27.0;
    if (m >= min_log_mel) return min_log_hz * std::exp((m - min_log_mel) * logstep);
    return 200.0 * m / 3.0;
}

// The periodic Hann window of n points: numpy's hanning(n + 1) without its last point
void hann_periodic(int n, std::vector<double>& w) {
    w.resize(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n);
}

// Reflect-padded sample at index i (numpy's 'reflect': the edge is not repeated), for a signal of len samples
inline float reflect_at(const float* x, long long len, long long i) {
    if (i < 0) i = -i;
    if (i >= len) i = 2 * (len - 1) - i;
    return x[i];
}

}  // namespace

extern "C" {

int mel_filter_bank(int n_mels, int n_fft, int sample_rate, double fmin, double fmax, float* out) {
    if (n_mels <= 0 || n_fft < 2 || sample_rate <= 0 || !out || fmin > fmax) return -1;
    const int bins = n_fft / 2 + 1;
    // the filters' centres: n_mels + 2 points, evenly spaced on the mel scale
    const double mel_min = hz_to_mel(fmin), mel_max = hz_to_mel(fmax);
    std::vector<double> f(n_mels + 2);
    for (int j = 0; j < n_mels + 2; ++j) f[j] = mel_to_hz(mel_min + (mel_max - mel_min) * j / (n_mels + 1));
    const double nyquist = sample_rate / 2.0;
    for (int m = 0; m < n_mels; ++m) {
        const double lo = f[m], mid = f[m + 1], hi = f[m + 2];
        const double enorm = 2.0 / (hi - lo);  // Slaney normalization: constant energy per filter
        for (int k = 0; k < bins; ++k) {
            const double freq = nyquist * k / (bins - 1);  // linspace(0, sample_rate / 2, bins)
            const double down = (freq - lo) / (mid - lo), up = (hi - freq) / (hi - mid);
            out[(size_t)m * bins + k] = (float)(std::max(0.0, std::min(down, up)) * enorm);
        }
    }
    return 0;
}

int whisper_log_mel(const float* samples, int n, int n_mels, float* out) {
    if ((n > 0 && !samples) || n < 0 || n_mels <= 0 || !out) return -1;
    const int bins = WHISPER_N_FFT / 2 + 1, T = WHISPER_N_FRAMES;
    const long long L = WHISPER_CHUNK_SAMPLES;
    // the chunk: truncated to 30 s, zero-padded to it (before the STFT, as the extractor does)
    std::vector<float> chunk((size_t)L, 0.0f);
    std::copy(samples, samples + std::min<long long>(n, L), chunk.begin());

    std::vector<double> filt((size_t)n_mels * bins);
    {
        std::vector<float> f32((size_t)n_mels * bins);
        if (mel_filter_bank(n_mels, WHISPER_N_FFT, 16000, 0.0, 8000.0, f32.data()) != 0) return -1;
        std::copy(f32.begin(), f32.end(), filt.begin());
    }
    std::vector<double> win;
    hann_periodic(WHISPER_N_FFT, win);
    // cos and sin of 2 pi j / N for the DFT's twiddles (the index k n mod N)
    std::vector<double> cs(WHISPER_N_FFT), sn(WHISPER_N_FFT);
    for (int j = 0; j < WHISPER_N_FFT; ++j) {
        cs[j] = std::cos(2.0 * M_PI * j / WHISPER_N_FFT);
        sn[j] = std::sin(2.0 * M_PI * j / WHISPER_N_FFT);
    }

    // the log10 mel values [n_mels][T] (the frames' centres at t * hop, the centred frames' reflect padding)
    std::vector<double> logmel((size_t)n_mels * T);
    const int N = WHISPER_N_FFT, H = WHISPER_HOP;
#pragma omp parallel
    {
        std::vector<double> buf(N), power(bins);
#pragma omp for schedule(static)
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < N; ++j) {
                const long long i = (long long)t * H + j - N / 2;  // the sample under window point j
                buf[j] = (double)reflect_at(chunk.data(), L, i) * win[j];
            }
            for (int k = 0; k < bins; ++k) {
                double re = 0.0, im = 0.0;
                for (int j = 0; j < N; ++j) {
                    const int idx = (k * j) % N;
                    re += buf[j] * cs[idx];
                    im -= buf[j] * sn[idx];
                }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < n_mels; ++m) {
                double s = 0.0;
                const double* fr = &filt[(size_t)m * bins];
                for (int k = 0; k < bins; ++k) s += fr[k] * power[k];
                logmel[(size_t)m * T + t] = std::log10(std::max(1e-10, s));
            }
        }
    }
    // the clamp to the maximum less 8, then (x + 4) / 4
    double mx = logmel[0];
    for (double v : logmel) mx = std::max(mx, v);
    for (size_t i = 0; i < logmel.size(); ++i) {
        const double v = std::max(logmel[i], mx - 8.0);
        out[i] = (float)((v + 4.0) / 4.0);
    }
    return 0;
}

}  // extern "C"
