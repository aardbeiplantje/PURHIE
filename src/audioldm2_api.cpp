#include "audioldm2_api.h"

#include "audioldm2_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {
std::string g_error;
}

struct aud_pipe {
    audioldm2::Pipeline pipe;
    explicit aud_pipe(const std::string& dir) : pipe(dir) {}
};

extern "C" {

aud_pipe_t* aud_open(const char* dir, const char** clap_vocab, size_t n_clap_vocab, const char** clap_merges, size_t n_clap_merges,
                     const char** t5_pieces, const double* t5_scores, size_t n_t5) {
    if (!dir || !clap_vocab || !clap_merges || !t5_pieces || !t5_scores) { g_error = "aud_open: missing arguments"; return nullptr; }
    aud_pipe_t* p = new aud_pipe(dir);
    if (!p->pipe.ok()) {
        g_error = "aud_open: " + p->pipe.error();
        delete p;
        return nullptr;
    }
    std::vector<std::string> cv(clap_vocab, clap_vocab + n_clap_vocab), cm(clap_merges, clap_merges + n_clap_merges);
    std::vector<std::string> tp(t5_pieces, t5_pieces + n_t5);
    std::vector<double> ts(t5_scores, t5_scores + n_t5);
    if (!p->pipe.load_tokenizers(cv, cm, tp, ts)) {
        g_error = "aud_open: the tokenizers";
        delete p;
        return nullptr;
    }
    g_error.clear();
    return p;
}

void aud_close(aud_pipe_t* p) {
    delete p;
}

const char* aud_last_error(void) {
    return g_error.c_str();
}

int aud_generate(aud_pipe_t* p, const char* prompt, const char* negative, float seconds, int steps, float guidance,
                 unsigned long long seed, float** out, size_t* n) {
    if (!p || !prompt || !out || !n) { g_error = "aud_generate: missing arguments"; return -1; }
    std::vector<float> wav;
    if (!p->pipe.generate_text(prompt, negative ? negative : "", seconds, steps, guidance, (uint64_t)seed, wav)) {
        g_error = "aud_generate: the pipeline failed (see the log)";
        return -1;
    }
    float* buf = static_cast<float*>(std::malloc(sizeof(float) * (wav.empty() ? 1 : wav.size())));
    if (!buf) { g_error = "aud_generate: out of memory"; return -1; }
    if (!wav.empty()) std::memcpy(buf, wav.data(), sizeof(float) * wav.size());
    *out = buf;
    *n = wav.size();
    g_error.clear();
    return 0;
}

void aud_free(float* samples) {
    std::free(samples);
}

}  // extern "C"
