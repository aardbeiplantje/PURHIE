// zimage_dit_verify: the Z-Image DiT's first stages (src/zimage_dit.cpp, host path) against the reference's stage outputs.
//
//   zimage_dit_verify <dit.safetensors> <raw dir>
//
// raw dir: the oracle as raw float32 files (export from /data/zimage-ref/dit: x.f32 [16][32][32] the latent, cap.f32 [15][2560]
// the caption's embeddings, t.f32 the step (0.5), t_embedder.f32 [256], x_embedder.f32 [256][3840], cap_embedder.f32
// [32][3840]). Each stage prints its max abs error and cosine against the reference. The host stages always run; the device stages run
// when the GPU is present (the mapping is registered for it). Build (the image device library, as tests/mtp_verify):
//   clang++ -O3 -I/opt/rocm/include -Isrc -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__ -I. tests/zimage_dit_verify.cpp \
//     src/zimage_dit.cpp -Wl,--start-group build/libqwen.a build/libinference.a -Wl,--end-group -L/opt/rocm/lib \
//     -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -fopenmp -lz -o zimage_dit_verify
#include "zimage_dit.h"
#include "sd_device.h"
#include "zimage_vae.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_raw(const std::string& path, std::vector<float>& v) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(float));
    const size_t got = fread(v.data(), 1, (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

static int compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol,
                   double cos_min = 0.9999) {
    if (got.size() != want.size()) {
        printf("[FAIL] %s: %zu values, the reference has %zu\n", what, got.size(), want.size());
        return 1;
    }
    double maxabs = 0, dot = 0, ng = 0, nw = 0, maxref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = want[i];
        maxabs = std::fmax(maxabs, std::fabs(a - b));
        maxref = std::fmax(maxref, std::fabs(b));
        dot += a * b; ng += a * a; nw += b * b;
    }
    const double cos = dot / std::sqrt(ng * nw + 1e-30);
    const bool ok = maxabs <= tol * std::fmax(maxref, 1e-12) && cos > cos_min;
    printf("[%s] %s: max abs %.3g (ref max %.3g, relative %.2g), cosine %.7f\n", ok ? "PASS" : "FAIL", what, maxabs,
           maxref, maxabs / std::fmax(maxref, 1e-12), cos);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s dit.safetensors raw_dir [ae.safetensors]\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    zimage::SafeTensors st(argv[1]);
    if (!st.ok()) { fprintf(stderr, "%s\n", st.error().c_str()); return 1; }
    std::vector<float> x, cap, t, want;
    if (!read_raw(dir + "/x.f32", x) || !read_raw(dir + "/cap.f32", cap) || !read_raw(dir + "/t.f32", t)) {
        fprintf(stderr, "cannot read the inputs in %s\n", dir.c_str());
        return 1;
    }
    int fails = 0;
    /* the reference's bf16 rounding of the timestep embedding is the expected difference: 2e-2 relative */
    std::vector<float> temb(256);
    zimage::timestep_embed(st, t[0], temb.data());
    if (read_raw(dir + "/t_embedder.f32", want)) fails += compare("t_embedder", temb, want, 2e-2);

    std::vector<float> tokens;
    zimage::patchify(x.data(), 16, 32, 32, 2, tokens);
    std::vector<float> xe;
    zimage::patch_embed(st, tokens, (int)(tokens.size() / 64), xe);
    if (read_raw(dir + "/x_embedder.f32", want)) fails += compare("x_embedder (patchify + patch embedding)", xe, want, 2e-2);

    std::vector<float> ce;
    int Lpad = 0;
    zimage::caption_embed(st, cap.data(), (int)(cap.size() / 2560), ce, Lpad);
    if (read_raw(dir + "/cap_embedder.f32", want)) fails += compare("cap_embedder (RMSNorm + linear, padded)", ce, want, 2e-2);

    /* the device path: the mapping registered (read in place), the activations in shared memory */
    /* the autoencoder (the VAE's file) for the decoder: its own mapping */
    zimage::SafeTensors vae(argc > 3 ? argv[3] : "/comfyui/vae/ae.safetensors");
    if (!vae.ok()) { fprintf(stderr, "%s\n", vae.error().c_str()); return 1; }
    if (sddev::enabled() && st.register_device() && vae.register_device()) {
        float* td = static_cast<float*>(sddev::alloc(sizeof(float) * 256, false));
        if (zimage::dev_timestep_embed(st, t[0], td) && read_raw(dir + "/t_embedder.f32", want)) {
            std::vector<float> tv(td, td + 256);
            fails += compare("device t_embedder", tv, want, 2e-2);
        } else fails++;
        float* xd = nullptr;
        int N = 0;
        if (zimage::dev_patch_embed(st, x.data(), 16, 32, 32, &xd, N) && read_raw(dir + "/x_embedder.f32", want)) {
            std::vector<float> xv(xd, xd + (size_t)N * 3840);
            fails += compare("device x_embedder (patchify + GEMM)", xv, want, 2e-2);
        } else fails++;
        /* the first noise refiner block on the device, from the embeddings above (in place in xd) */
        float* fr = static_cast<float*>(sddev::alloc(sizeof(float) * 256 * 64 * 2, false));
        zimage::rope_table_image(16, 16, 33, fr);
        if (zimage::dev_refiner_block(st, "noise_refiner.0", xd, N, td, fr) && read_raw(dir + "/noise_refiner0.f32", want)) {
            std::vector<float> rv(xd, xd + (size_t)N * 3840);
            fails += compare("device noise_refiner.0 (AdaLN, joint RoPE attention, SwiGLU)", rv, want, 2e-2);
        } else fails++;
        float* cd = nullptr;
        int Lc = 0;
        if (zimage::dev_caption_embed(st, cap.data(), (int)(cap.size() / 2560), &cd, Lc) && read_raw(dir + "/cap_embedder.f32", want)) {
            std::vector<float> cv(cd, cd + (size_t)Lc * 3840);
            fails += compare("device cap_embedder (RMS norm kernel + GEMM)", cv, want, 2e-2);
        } else fails++;
        /* the second noise refiner block on the output of the first (in place in xd) */
        if (zimage::dev_refiner_block(st, "noise_refiner.1", xd, N, td, fr) && read_raw(dir + "/noise_refiner_1.f32", want)) {
            std::vector<float> rv(xd, xd + (size_t)N * 3840);
            fails += compare("device noise_refiner.1", rv, want, 2e-2);
        } else fails++;
        /* the caption: its pad rows take the pad token, then the two context refiners (no modulation; RoPE ids (1 + t, 0, 0)) */
        if (cd && Lc == 32) {
            zimage::dev_caption_pad_rows(st, cd, 15, Lc);
            int ids[32 * 3];
            for (int t2 = 0; t2 < 32; ++t2) { ids[t2 * 3] = 1 + t2; ids[t2 * 3 + 1] = 0; ids[t2 * 3 + 2] = 0; }
            float* frc = static_cast<float*>(sddev::alloc(sizeof(float) * 32 * 64 * 2, false));
            zimage::rope_table_ids(ids, 32, frc);
            if (zimage::dev_refiner_block(st, "context_refiner.0", cd, Lc, nullptr, frc, false) && read_raw(dir + "/context_refiner_0.f32", want)) {
                std::vector<float> cv0(cd, cd + (size_t)Lc * 3840);
                fails += compare("device context_refiner.0", cv0, want, 2e-2);
            } else fails++;
            if (zimage::dev_refiner_block(st, "context_refiner.1", cd, Lc, nullptr, frc, false) && read_raw(dir + "/context_refiner_1.f32", want)) {
                std::vector<float> cv1(cd, cd + (size_t)Lc * 3840);
                fails += compare("device context_refiner.1", cv1, want, 2e-2);
            } else fails++;
        }
        /* the whole device stack over one joint buffer J ([288][3840]: the image rows, then the caption rows, written in place):
         * the embeddings, the refiners, the 30 main layers, the output; the layers and the output against the oracle */
        {
            const int NI = 256, LC = 32, S = NI + LC, DIM = 3840;
            float* J = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)S * DIM, false));
            float* jx = J;
            int nx = 0;
            zimage::dev_patch_embed(st, x.data(), 16, 32, 32, &jx, nx);
            float* jt = static_cast<float*>(sddev::alloc(sizeof(float) * 256, false));
            zimage::dev_timestep_embed(st, t[0], jt);
            float* fri = static_cast<float*>(sddev::alloc(sizeof(float) * NI * 64 * 2, false));
            zimage::rope_table_image(16, 16, 33, fri);
            bool ok = zimage::dev_refiner_block(st, "noise_refiner.0", J, NI, jt, fri) &&
                      zimage::dev_refiner_block(st, "noise_refiner.1", J, NI, jt, fri);
            float* jc = J + (size_t)NI * DIM;
            int lc = 0;
            ok = ok && zimage::dev_caption_embed(st, cap.data(), (int)(cap.size() / 2560), &jc, lc);
            zimage::dev_caption_pad_rows(st, jc, 15, LC);
            int ids[32 * 3];
            for (int t2 = 0; t2 < LC; ++t2) { ids[t2 * 3] = 1 + t2; ids[t2 * 3 + 1] = 0; ids[t2 * 3 + 2] = 0; }
            float* frc = static_cast<float*>(sddev::alloc(sizeof(float) * LC * 64 * 2, false));
            zimage::rope_table_ids(ids, LC, frc);
            ok = ok && zimage::dev_refiner_block(st, "context_refiner.0", jc, LC, nullptr, frc, false) &&
                 zimage::dev_refiner_block(st, "context_refiner.1", jc, LC, nullptr, frc, false);
            int idj[288 * 3];
            for (int r = 0; r < NI; ++r) { idj[r * 3] = 33; idj[r * 3 + 1] = r / 16; idj[r * 3 + 2] = r % 16; }
            for (int r = 0; r < LC; ++r) { idj[(NI + r) * 3] = 1 + r; idj[(NI + r) * 3 + 1] = 0; idj[(NI + r) * 3 + 2] = 0; }
            float* frj = static_cast<float*>(sddev::alloc(sizeof(float) * S * 64 * 2, false));
            zimage::rope_table_ids(idj, S, frj);
            for (int layer = 0; ok && layer < 30; ++layer) {
                ok = zimage::dev_refiner_block(st, "layers." + std::to_string(layer), J, S, jt, frj, true);
                if (layer == 0 && ok && read_raw(dir + "/layers_0.f32", want)) {
                    std::vector<float> lv(J, J + (size_t)S * DIM);
                    fails += compare("device layers.0 (joint sequence)", lv, want, 2e-2);
                }
                if (layer == 29 && ok && read_raw(dir + "/layers_29.f32", want)) {
                    std::vector<float> lv(J, J + (size_t)S * DIM);
                    fails += compare("device layers.29 (joint sequence)", lv, want, 2e-2);
                }
            }
            std::vector<float> lat(16 * 32 * 32);
            if (ok && zimage::dev_output(st, J, jt, lat.data()) && read_raw(dir + "/out.f32", want))
                /* 5%: the error of the whole stack (34 blocks, each rounding its activations to bf16 as the reference does): the
                 * final layer alone matches to 0.3% (the check below), the layers to 0.6-0.7% each */
                fails += compare("device output (final layer, unpatchify; whole stack)", lat, want, 5e-2, 0.9998);
            /* the device output alone, on the reference's own layers.29 (isolates the final layer from the layers' error) */
            if (read_raw(dir + "/layers_29.f32", want)) {
                float* J29 = static_cast<float*>(sddev::alloc(sizeof(float) * (size_t)S * DIM, false));
                for (size_t i = 0; i < want.size(); ++i) J29[i] = want[i];
                std::vector<float> lat2(16 * 32 * 32);
                std::vector<float> ref_out;
                if (zimage::dev_output(st, J29, jt, lat2.data()) && read_raw(dir + "/out.f32", ref_out))
                    fails += compare("device output on the reference layers.29 (final layer alone)", lat2, ref_out, 2e-2);
            }
            else { printf("[FAIL] device stack did not complete\n"); fails++; }
        }
        /* the sampler: 8 steps from the reference's initial noise (the pipeline's noise for seed 0), shift 3, against its final
         * latent (the reference run used the same schedule; each step is a full device forward pass) */
        {
            std::vector<float> noise, latf(16 * 32 * 32);
            if (read_raw(dir + "/noise.f32", noise) && zimage::dev_sample(st, noise.data(), cap.data(), (int)(cap.size() / 2560), 8, 3.0f, latf.data()) &&
                read_raw(dir + "/latent_final.f32", want))
                /* 10%: eight forward passes, each with the accumulated bf16 rounding (the whole stack alone: 3.3%); the cosine
                 * (0.99997) is what shows the trajectory and the schedule agree */
                fails += compare("device sampler (8 steps from the reference noise)", latf, want, 1e-1, 0.9999);
            else { printf("[FAIL] device sampler did not run\n"); fails++; }
        }
        /* the Flux AE decoder on the reference's final latent, against the reference decode (float pixels) */
        {
            std::vector<float> lat0, dec_ref, dec(3 * 256 * 256);
            if (read_raw(dir + "/vae_latent.f32", lat0) && read_raw(dir + "/vae_decode.f32", dec_ref) &&
                zimage::dev_vae_decode(vae, lat0.data(), dec.data()))
                fails += compare("device VAE decode (Flux AE, 16 channels)", dec, dec_ref, 2e-2, 0.9999);
            else { printf("[FAIL] device VAE did not run\n"); fails++; }
        }
    } else {
        printf("[SKIP] device path: %s\n", sddev::enabled() ? "the mapping cannot be registered" : "no device");
    }
    printf("%s\n", fails ? "FAILED" : "all stages PASS");
    return fails ? 1 : 0;
}
