#include "audioldm2_pipeline.h"

#include "audioldm2_clap.h"
#include "audioldm2_gpt2.h"
#include "audioldm2_projection.h"
#include "audioldm2_t5.h"
#include "audioldm2_unet.h"
#include "audioldm2_vae.h"
#include "audioldm2_vocoder.h"
#include "sd_device.h"
#include "torch_randn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace audioldm2 {
namespace {

constexpr int kLatentCh = 8, kLatentW = 16;   /* the latent: 8 channels, 64 mel bins / 4 */
constexpr int kTrainSteps = 1000, kStepsOffset = 1, kGptNew = 8;
constexpr int kSampleRate = 16000, kMelHop = 160;  /* the vocoder: 160 samples per mel frame */

}  // namespace

struct Pipeline::Files {
    zimage::SafeTensors clap, t5, proj, lm, unet, vae, voc;
    Files(const std::string& d)
        : clap(d + "/text_encoder/model.safetensors"), t5(d + "/text_encoder_2/model.safetensors"),
          proj(d + "/projection_model/diffusion_pytorch_model.safetensors"), lm(d + "/language_model/model.safetensors"),
          unet(d + "/unet/diffusion_pytorch_model.safetensors"), vae(d + "/vae/diffusion_pytorch_model.safetensors"),
          voc(d + "/vocoder/model.safetensors") {}
};

Pipeline::Pipeline(const std::string& dir) {
    f_.reset(new Files(dir));
    zimage::SafeTensors* all[] = {&f_->clap, &f_->t5, &f_->proj, &f_->lm, &f_->unet, &f_->vae, &f_->voc};
    const char* names[] = {"text_encoder", "text_encoder_2", "projection_model", "language_model", "unet", "vae", "vocoder"};
    for (int i = 0; i < 7; ++i) {
        if (!all[i]->ok()) { error_ = std::string(names[i]) + ": " + all[i]->error(); return; }
        if (!all[i]->register_device()) { error_ = std::string(names[i]) + ": the device registration failed"; return; }
    }
}

Pipeline::~Pipeline() = default;

bool Pipeline::generate(const Tokens& pos, const Tokens& neg, float seconds, int steps, float guidance, uint64_t seed, std::vector<float>& wav) {
    if (!ok()) { std::fprintf(stderr, "audioldm2: %s\n", error_.c_str()); return false; }
    if (steps < 1 || steps > kTrainSteps || seconds <= 0.0f) { std::fprintf(stderr, "audioldm2: steps 1..1000, seconds > 0\n"); return false; }
    /* the spectrogram height (frames): int(seconds / 0.01), rounded up to the VAE's factor 4; the latent is height / 4 by 16 */
    int height = (int)(seconds / 0.01f);
    if (height % 4 != 0) height = ((height + 3) / 4) * 4;
    const int Hl = height / 4, Wl = kLatentW;
    const int original = (int)(seconds * kSampleRate);
    /* the text: the negative (uncond) first, then the prompt, as the pipeline batches them. The negative prompt's T5 tokens are
     * padded to the prompt's length (encode_prompt: max_length = the prompt's length), with the padding masked */
    const int Lp = (int)pos.t5_ids.size();
    struct Text { std::vector<float> ctx1, ctx2; std::vector<int32_t> mask2; int T2 = 0; };
    Text texts[2];
    const Tokens* toks[2] = {&neg, &pos};
    for (int k = 0; k < 2; ++k) {
        const Tokens& t = *toks[k];
        std::vector<int32_t> ids = t.t5_ids, msk(ids.size(), 1);
        if (k == 0 && (int)ids.size() < Lp) {
            ids.resize(Lp, 0);
            msk.resize(Lp, 0);
        }
        const int L = (int)ids.size();
        std::vector<float> feat(512), t5h((size_t)L * 1024), pj((size_t)(L + 5) * 768);
        std::vector<int32_t> pmask((size_t)L + 5);
        if (!clap_text_forward(f_->clap, t.clap_ids.data(), t.clap_mask.data(), (int)t.clap_ids.size(), nullptr, nullptr, feat.data())) return false;
        if (!t5_encoder_forward(f_->t5, ids.data(), msk.data(), L, t5h.data())) return false;
        if (!projection_forward(f_->proj, feat.data(), 1, t5h.data(), L, msk.data(), pj.data(), pmask.data())) return false;
        texts[k].ctx1.assign((size_t)kGptNew * 768, 0.0f);
        if (!gpt2_generate(f_->lm, pj.data(), pmask.data(), L + 5, kGptNew, texts[k].ctx1.data())) return false;
        texts[k].ctx2 = std::move(t5h);
        texts[k].mask2 = msk;
        texts[k].T2 = L;
    }
    /* the initial latent: torch's CPU randn for the seed, [8][Hl][Wl] */
    const size_t nl = (size_t)kLatentCh * Hl * Wl;
    std::vector<float> x(nl);
    torch_randn(seed, x.data(), nl);
    /* the DDIM schedule (scaled-linear betas, leading spacing with offset 1, epsilon prediction, eta 0) */
    std::vector<double> acp(kTrainSteps);
    {
        const double b0 = std::sqrt(0.0015), b1 = std::sqrt(0.0195);
        double prod = 1.0;
        for (int i = 0; i < kTrainSteps; ++i) {
            const double b = b0 + (b1 - b0) * i / (kTrainSteps - 1);
            prod *= 1.0 - b * b;
            acp[i] = prod;
        }
    }
    const int ratio = kTrainSteps / steps;
    std::vector<float> eu(nl), ec(nl), eps(nl);
    for (int k = 0; k < steps; ++k) {
        const int t = (steps - 1 - k) * ratio + kStepsOffset;
        const int prev = t - ratio;
        const double a_t = acp[t];
        const double a_prev = prev >= 0 ? acp[prev] : acp[0];  /* set_alpha_to_one is false: the final alpha is the first one */
        /* classifier-free guidance: the unconditional and the conditional predictions */
        if (!unet_forward(f_->unet, x.data(), Hl, Wl, (float)t, texts[0].ctx1.data(), kGptNew, texts[0].ctx2.data(), texts[0].T2, texts[0].mask2.data(), eu.data())) return false;
        if (!unet_forward(f_->unet, x.data(), Hl, Wl, (float)t, texts[1].ctx1.data(), kGptNew, texts[1].ctx2.data(), texts[1].T2, texts[1].mask2.data(), ec.data())) return false;
        for (size_t i = 0; i < nl; ++i) eps[i] = eu[i] + guidance * (ec[i] - eu[i]);
        /* DDIM: x0 = (x - sqrt(1 - a_t) eps) / sqrt(a_t); x_prev = sqrt(a_prev) x0 + sqrt(1 - a_prev) eps */
        const double sa = std::sqrt(a_t), sb = std::sqrt(1.0 - a_t), sp = std::sqrt(a_prev), sq = std::sqrt(1.0 - a_prev);
        for (size_t i = 0; i < nl; ++i) {
            const double x0 = (x[i] - sb * eps[i]) / sa;
            x[i] = (float)(sp * x0 + sq * eps[i]);
        }
        /* debugging (ZI_AUDIO_DUMP=dir): the latent after each step, as the pipeline's callback gives it */
        if (const char* dump = std::getenv("ZI_AUDIO_DUMP")) {
            char name[512];
            std::snprintf(name, sizeof name, "%s/engine_step_%03d.f32", dump, k);
            if (FILE* f = std::fopen(name, "wb")) { std::fwrite(x.data(), sizeof(float), nl, f); std::fclose(f); }
        }
    }
    /* the VAE (the latent divided by the scaling factor inside), the mel [4 Hl][4 Wl] -> the vocoder (frames of 64 bins) */
    std::vector<float> mel((size_t)height * 64);
    if (!vae_decode(f_->vae, x.data(), Hl, Wl, mel.data())) return false;
    std::vector<float> full;
    if (!vocoder_forward(f_->voc, mel.data(), height, full)) return false;
    /* the audio length: the waveform cut to the seconds asked (the vocoder's output runs a few samples over) */
    if ((int)full.size() > original) full.resize((size_t)original);
    wav = std::move(full);
    (void)kMelHop;
    (void)kSampleRate;
    return true;
}

}  // namespace audioldm2
