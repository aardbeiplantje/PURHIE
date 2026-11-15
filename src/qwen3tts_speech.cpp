// Speech (qwen3tts_speech.h): the chat ids of the text and the reference transcript, the sampled frames, then the codec.
#include "qwen3tts_speech.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string g_error;
}

int qwen3tts_speak(qwen3tts_model_t* model, qwen3tts_codec_t* codec, const char* text, const char* ref_text,
                   const int* ref_codes, int R, const float* spk, int language_id, const qwen3tts_sampling_t* s,
                   int max_frames, float** wav, size_t* n_out, int* n_frames) {
    if (!model || !codec || !text || !ref_text || !ref_codes || !spk || !s || !wav || !n_out || !n_frames || R < 1 ||
        max_frames <= 0) {
        g_error = "qwen3tts_speak: bad arguments";
        return -1;
    }
    *wav = nullptr;
    *n_out = 0;
    *n_frames = 0;
    try {
        const size_t tlen = std::strlen(text), rlen = std::strlen(ref_text);
        std::vector<int> text_ids(tlen + 64), ref_ids(rlen + 64);
        const int nt = qwen3tts_tokenize_chat(model, text, text_ids.data(), (int)text_ids.size());
        int nr = qwen3tts_tokenize_chat(model, ref_text, ref_ids.data(), (int)ref_ids.size());
        if (nt < 0 || nr < 0) throw std::runtime_error(qwen3tts_last_error());
        // the reference's ids as the prefill takes them: the template without its last three ids (the assistant header)
        if (nr < 3 + 8) throw std::runtime_error("qwen3tts_speak: the reference transcript is too short");
        nr -= 3;
        if (nt < 9) throw std::runtime_error("qwen3tts_speak: the text is too short");
        std::vector<int> codes((size_t)max_frames * 16);
        const int T = qwen3tts_generate(model, text_ids.data(), nt, ref_ids.data(), nr, ref_codes, R, spk, language_id, s,
                                        max_frames, codes.data());
        if (T < 0) throw std::runtime_error(qwen3tts_last_error());
        *n_frames = T;
        if (T == 0) return 0;
        // the reference's last frames as the left context, then the talker's, decoded together; the context's share of the
        // samples cut off. 25 frames: qwen-tts's chunked decode's left context (the whole reference, the same output as its
        // speak, at R + T frames: QWEN3TTS_CODEC_CONTEXT=0). A voice's reference is 74 to 204 frames: decoding all of it
        // for every request was most of a request's time.
        const int ncb = qwen3tts_n_codebooks(model);
        if (ncb != 16) throw std::runtime_error("qwen3tts_speak: the codec frames are not 16 codebooks");
        const char* cv = std::getenv("QWEN3TTS_CODEC_CONTEXT");
        const int ctx_max = cv ? std::atoi(cv) : 0;
        const int ctx = ctx_max <= 0 ? R : std::min(R, ctx_max);
        std::vector<int> all((size_t)(ctx + T) * ncb);
        std::memcpy(all.data(), ref_codes + (size_t)(R - ctx) * ncb, sizeof(int) * (size_t)ctx * ncb);
        std::memcpy(all.data() + (size_t)ctx * ncb, codes.data(), sizeof(int) * (size_t)T * ncb);
        float* full = nullptr;
        size_t n = 0;
        if (qwen3tts_codec_decode(codec, all.data(), ctx + T, &full, &n) != 0) throw std::runtime_error(qwen3tts_codec_last_error());
        const size_t cut = (size_t)((double)ctx / (double)(ctx + T) * (double)n);
        const size_t keep = n - cut;
        float* out = (float*)std::malloc(sizeof(float) * (keep ? keep : 1));
        if (!out) {
            qwen3tts_codec_free_wav(full);
            throw std::runtime_error("qwen3tts_speak: out of memory");
        }
        std::memcpy(out, full + cut, sizeof(float) * keep);
        qwen3tts_codec_free_wav(full);
        *wav = out;
        *n_out = keep;
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

void qwen3tts_speech_free_wav(float* wav) { std::free(wav); }

int qwen3tts_make_voice(qwen3tts_model_t* model, qwen3tts_codec_t* codec, const float* samples, size_t n, int** codes_out,
                        int* R_out, float* spk) {
    if (!model || !codec || !samples || !codes_out || !R_out || !spk) {
        g_error = "qwen3tts_make_voice: bad arguments";
        return -1;
    }
    if (qwen3tts_codec_encode(codec, samples, n, codes_out, R_out) != 0) {
        g_error = std::string("qwen3tts_make_voice: ") + qwen3tts_codec_last_error();
        return -1;
    }
    if (qwen3tts_speaker_embedding(model, samples, n, spk) != 0) {
        qwen3tts_codec_free_codes(*codes_out);
        *codes_out = nullptr;
        g_error = std::string("qwen3tts_make_voice: ") + qwen3tts_last_error();
        return -1;
    }
    return 0;
}
