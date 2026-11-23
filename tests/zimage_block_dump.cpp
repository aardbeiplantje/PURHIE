/* One DiT call of the engine at 256 x 256 on the inputs of /data/zimage-ref/blocks (written by ref_blocks.py: x.f32, cap.f32, t.f32),
 * with the per-block dump of dev_dit_forward (ZI_DIT_DUMP=dir must be set). Usage: zimage_block_dump [raw_dir] [dit.safetensors] */
#include "zimage_dit.h"
#include "sd_device.h"

#include <cstdio>
#include <string>
#include <vector>

static bool read_raw(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes / sizeof(float));
    const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/zimage-ref/blocks";
    const std::string dit_path = argc > 2 ? argv[2] : "/comfyui/diffusion_models/z_image_turbo_bf16.safetensors";
    std::vector<float> x, cap, t;
    if (!read_raw(dir + "/x.f32", x) || !read_raw(dir + "/cap.f32", cap) || !read_raw(dir + "/t.f32", t)) {
        std::fprintf(stderr, "cannot read the inputs in %s\n", dir.c_str());
        return 1;
    }
    zimage::SafeTensors st(dit_path);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the DiT file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    const int L = (int)(cap.size() / 2560);
    std::vector<float> vel((size_t)16 * 32 * 32);
    if (!zimage::dev_dit_forward(st, x.data(), cap.data(), L, t[0], vel.data(), 16, 16)) {
        std::fprintf(stderr, "the DiT call failed\n");
        return 1;
    }
    if (const char* dir = std::getenv("ZI_DIT_DUMP")) {  /* the output velocity, as the reference's out.f32 */
        std::string path = std::string(dir) + "/out.f32";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(vel.data(), sizeof(float), vel.size(), f);
            std::fclose(f);
        }
    }
    std::printf("engine DiT call done: L %d, dump in %s\n", L, std::getenv("ZI_DIT_DUMP") ? std::getenv("ZI_DIT_DUMP") : "(ZI_DIT_DUMP unset)");
    return 0;
}
