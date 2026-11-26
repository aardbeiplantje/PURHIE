// An image to a textured 3D model on the engine (trellis_image_to_3d.h), and its C interface with one job at a time on a background
// thread (trellis2_api.h).
#include "trellis_image_to_3d.h"

#include "birefnet.h"
#include "dinov3.h"
#include "image.h"
#include "sd_device.h"
#include "trellis2_api.h"
#include "trellis_bake.h"
#include "trellis_charts.h"
#include "trellis_image.h"
#include "trellis_shape.h"
#include "trellis_simplify.h"
#include "trellis_solid.h"
#include "trellis_tex.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <random>
#include <thread>

namespace trellis {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool read_file(const std::string& path, std::string& s) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    s.resize(n > 0 ? (size_t)n : 0);
    const size_t got = s.empty() ? 0 : std::fread(&s[0], 1, s.size(), f);
    std::fclose(f);
    return got == s.size();
}

// the 32 numbers of "name": [ ... ] within s[from, to)
bool json_array32(const std::string& s, size_t from, size_t to, const char* name, float out[32]) {
    const std::string key = std::string("\"") + name + "\"";
    const size_t k = s.find(key, from);
    if (k == std::string::npos || k >= to) return false;
    size_t i = s.find('[', k);
    if (i == std::string::npos || i >= to) return false;
    ++i;
    for (int c = 0; c < 32; ++c) {
        while (i < to && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
        char* end = nullptr;
        out[c] = std::strtof(s.c_str() + i, &end);
        if (end == s.c_str() + i) return false;
        i = (size_t)(end - s.c_str());
    }
    return true;
}

}  // namespace

bool read_trellis2_normalisation(const std::string& pipeline_json, float shape_mean[32], float shape_std[32], float tex_mean[32],
                                 float tex_std[32], std::string& error) {
    std::string s;
    if (!read_file(pipeline_json, s)) {
        error = "cannot read " + pipeline_json;
        return false;
    }
    for (int which = 0; which < 2; ++which) {
        const char* key = which == 0 ? "\"shape_slat_normalization\"" : "\"tex_slat_normalization\"";
        const size_t k = s.find(key);
        const size_t end = k == std::string::npos ? k : s.find('}', k);
        if (k == std::string::npos || end == std::string::npos ||
            !json_array32(s, k, end, "mean", which == 0 ? shape_mean : tex_mean) ||
            !json_array32(s, k, end, "std", which == 0 ? shape_std : tex_std)) {
            error = pipeline_json + ": no " + key + " with 32 means and stds";
            return false;
        }
    }
    return true;
}

bool crop_to_mask(const uint8_t* rgb, const float* mask, int w, int h, float margin, int size, std::vector<uint8_t>& out) {
    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (mask[(size_t)y * w + x] > 0.5f) {
                x0 = std::min(x0, x);
                x1 = std::max(x1, x);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y);
            }
    if (x1 < 0) return false;
    const double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
    const double side = std::max(x1 - x0 + 1, y1 - y0 + 1) * (double)margin;
    const int bx0 = (int)std::lround(cx - side / 2), by0 = (int)std::lround(cy - side / 2);
    const int cw = (int)std::lround(cx + side / 2) - bx0, ch = (int)std::lround(cy + side / 2) - by0;
    // the object over black, the square around it (black outside the image)
    std::vector<uint8_t> crop((size_t)cw * ch * 3, 0);
    for (int y = 0; y < ch; ++y) {
        const int sy = by0 + y;
        if (sy < 0 || sy >= h) continue;
        for (int x = 0; x < cw; ++x) {
            const int sx = bx0 + x;
            if (sx < 0 || sx >= w) continue;
            const float a = mask[(size_t)sy * w + sx];
            for (int c = 0; c < 3; ++c) {
                const float v = rgb[((size_t)sy * w + sx) * 3 + c] * a;
                crop[((size_t)y * cw + x) * 3 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v));
            }
        }
    }
    out = resize_lanczos_rgb8(crop.data(), cw, ch, size, size);
    return true;
}

struct ImageTo3D::Impl {
    ImageTo3DFiles files;
    std::unique_ptr<DinoV3> enc;
    std::unique_ptr<BiRefNet> bir;  // null: no background removal (only images with an alpha channel are cut out)
    std::unique_ptr<ShapePipeline> shape;
    std::unique_ptr<TexturePipeline> tex;
    float smean[32], sstd[32], tmean[32], tstd[32];
    std::vector<uint8_t> last_input;
    std::vector<std::pair<std::string, double>> timings;
    std::string error;
    bool ok = false;
};

ImageTo3D::ImageTo3D(const ImageTo3DFiles& files) : p_(new Impl) {
    Impl& m = *p_;
    m.files = files;
    if (!read_trellis2_normalisation(files.trellis2_dir + "/pipeline.json", m.smean, m.sstd, m.tmean, m.tstd, m.error)) return;
    const std::string ck = files.trellis2_dir + "/ckpts/";
    m.enc.reset(new DinoV3(files.dinov3));
    if (!m.enc->ok()) {
        m.error = "image encoder: " + m.enc->error();
        return;
    }
    if (!files.birefnet.empty()) {
        m.bir.reset(new BiRefNet(files.birefnet));
        if (!m.bir->ok()) {
            m.error = "background removal: " + m.bir->error();
            return;
        }
    }
    m.shape.reset(new ShapePipeline(ck + "ss_flow_img_dit_1_3B_64_bf16.safetensors", files.ss_dec,
                                    ck + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", ck + "shape_dec_next_dc_f16c32_fp16.safetensors",
                                    ck + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors"));
    if (!m.shape->ok()) {
        m.error = "shape: " + m.shape->error();
        return;
    }
    m.tex.reset(new TexturePipeline(ck + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", ck + "tex_dec_next_dc_f16c32_fp16.safetensors",
                                    ck + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"));
    if (!m.tex->ok()) {
        m.error = "texture: " + m.tex->error();
        return;
    }
    m.ok = true;
}

ImageTo3D::~ImageTo3D() = default;

bool ImageTo3D::ok() const { return p_->ok; }
const std::string& ImageTo3D::error() const { return p_->error; }
bool ImageTo3D::has_background_removal() const { return p_->bir != nullptr; }
const std::vector<uint8_t>& ImageTo3D::last_input() const { return p_->last_input; }
const std::vector<std::pair<std::string, double>>& ImageTo3D::last_timings() const { return p_->timings; }

bool ImageTo3D::generate(const uint8_t* rgba, int w, int h, bool has_alpha, const ImageTo3DOptions& opt, std::vector<uint8_t>& glb,
                         const ImageTo3DProgress& progress) {
    Impl& m = *p_;
    m.error.clear();
    m.timings.clear();
    glb.clear();
    if (!m.ok) {
        m.error = "the pipeline is not open";
        return false;
    }
    if (w <= 0 || h <= 0) {
        m.error = "no image";
        return false;
    }
    if (opt.resolution != 512 && opt.resolution != 1024) {
        m.error = "resolution: 512 or 1024";
        return false;
    }
    const bool textured = opt.texture && opt.format == ImageTo3DOptions::Format::Glb;
    if (opt.faces < 1000 || (textured && (opt.texture_size < 256 || opt.texture_size > 8192))) {
        m.error = "faces: at least 1000; texture size: 256 to 8192";
        return false;
    }
    if (opt.format == ImageTo3DOptions::Format::Stl && (!(opt.stl_size_mm >= 0) || opt.stl_size_mm > 100000)) {
        m.error = "the STL's size: 0 to 100000 millimetres";
        return false;
    }
    if (!(opt.hollow_mm >= 0) || opt.hollow_mm > 1000 || !(opt.drain_mm >= 0) || opt.drain_mm > 1000) {
        m.error = "hollow_mm and drain_mm: 0 to 1000 millimetres";
        return false;
    }
    if (opt.hollow_mm > 0 && opt.hollow_mm < 2) {
        m.error = "hollow_mm: 0 (solid), or walls of at least 2 millimetres";
        return false;
    }
    if (opt.hollow_mm > 0 && (opt.format != ImageTo3DOptions::Format::Stl || !(opt.stl_size_mm > 0) || !opt.solid)) {
        m.error = "a hollow (hollow_mm) is for an STL with its size in millimetres";
        return false;
    }
    auto report = [&](const char* stage, int step, int steps) {
        if (progress) progress(stage, step, steps);
    };
    auto t0 = std::chrono::steady_clock::now();

    // 1. the object: the mask (the alpha, BiRefNet, or the whole image), then the workflow's crop at 1024 x 1024
    report("background", 0, 0);
    const size_t npx = (size_t)w * h;
    std::vector<uint8_t> rgb(npx * 3);
    std::vector<float> mask(npx, 1.0f);
    bool opaque = true;
    for (size_t i = 0; i < npx; ++i) {
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = rgba[i * 4 + c];
        if (rgba[i * 4 + 3] != 255) opaque = false;
    }
    float margin = 1.1f;
    if (opt.remove_background == 0) {
        margin = 1.0f;  // the whole image, squared with black
    } else if (has_alpha && !opaque && opt.remove_background != 1) {
        for (size_t i = 0; i < npx; ++i) mask[i] = rgba[i * 4 + 3] / 255.0f;
    } else if (m.bir) {
        report("background removal", 0, 0);
        if (!m.bir->mask(rgb.data(), w, h, mask)) {
            m.error = "background removal: " + m.bir->error();
            return false;
        }
    } else {
        m.error = "no background removal (BiRefNet) in this pipeline: give an image with an alpha channel, or remove_background 0";
        return false;
    }
    if (!crop_to_mask(rgb.data(), mask.data(), w, h, margin, 1024, m.last_input)) {
        m.error = "the mask is empty: no object in the image";
        return false;
    }
    m.timings.push_back({"background and crop", seconds_since(t0)});

    // 2. the conditioning (DINOv3 at 512, and at 1024 for the cascade)
    t0 = std::chrono::steady_clock::now();
    report("conditioning", 0, 0);
    const bool cascade = opt.resolution == 1024;
    std::vector<float> cond_lr, cond_hr;
    if (!image_condition(*m.enc, m.last_input.data(), 1024, 1024, 512, cond_lr, m.error) ||
        (cascade && !image_condition(*m.enc, m.last_input.data(), 1024, 1024, 1024, cond_hr, m.error))) {
        m.error = "conditioning: " + m.error;
        return false;
    }
    m.timings.push_back({"conditioning", seconds_since(t0)});

    // 3. the shape and the texture (the samplers of pipeline.json; the cascade's 512 stage with the workflow's steps)
    ShapeConfig scfg;
    scfg.ss.steps = 12;
    scfg.ss.guidance = 7.5f;
    scfg.ss.guidance_rescale = 0.7f;
    scfg.ss.rescale_t = 5.0;
    scfg.slat.steps = 12;
    scfg.slat.guidance = 7.5f;
    scfg.slat.guidance_rescale = 0.5f;
    scfg.slat.rescale_t = 3.0;
    scfg.slat_lr_steps = cascade ? opt.slat_lr_steps : 0;
    for (int c = 0; c < 32; ++c) {
        scfg.slat_mean[c] = m.smean[c];
        scfg.slat_std[c] = m.sstd[c];
    }
    scfg.mesh_grid = opt.resolution;
    TexConfig tcfg;
    tcfg.sampler.steps = 12;
    tcfg.sampler.guidance = 1.0f;
    tcfg.sampler.guidance_rescale = 0.0f;
    tcfg.sampler.rescale_t = 3.0;
    tcfg.sampler.interval_lo = 0.6;
    tcfg.sampler.interval_hi = 0.9;
    for (int c = 0; c < 32; ++c) {
        tcfg.shape_mean[c] = m.smean[c];
        tcfg.shape_std[c] = m.sstd[c];
        tcfg.tex_mean[c] = m.tmean[c];
        tcfg.tex_std[c] = m.tstd[c];
    }
    // the progress of the samplers: the shape SLAT's first flow is the 512 one, a flow on another token count the 1024 one
    scfg.ss.on_step = [&](int step, const float*, size_t) { report("sparse structure", step + 1, scfg.ss.steps); };
    size_t slat_first = 0;
    scfg.slat.on_step = [&](int step, const float*, size_t count) {
        if (slat_first == 0) slat_first = count;
        const bool hr = count != slat_first;
        const int steps = hr || !cascade ? scfg.slat.steps : scfg.slat_lr_steps;
        report(hr ? "shape SLAT (1024)" : "shape SLAT (512)", step + 1, steps);
    };
    tcfg.sampler.on_step = [&](int step, const float*, size_t) { report("texture flow", step + 1, tcfg.sampler.steps); };
    std::mt19937 rng((uint32_t)(opt.seed ^ (opt.seed >> 32)));
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    NoiseFn noise = [&](float* dst, size_t count) {
        for (size_t i = 0; i < count; ++i) dst[i] = gauss(rng);
    };
    std::string err;
    if (!textured) {
        // the mesh alone: the shape (and its cascade), the simplification, the GLB or STL; no texture flow, unwrap or bake
        ShapeResult sr;
        const bool shape_ok = cascade ? m.shape->run_cascade(cond_lr.data(), (int)(cond_lr.size() / 1024), cond_hr.data(),
                                                             (int)(cond_hr.size() / 1024), noise, scfg, sr)
                                      : m.shape->run(cond_lr.data(), (int)(cond_lr.size() / 1024), noise, scfg, sr);
        if (!shape_ok) {
            m.error = (cascade ? "cascade: " : "shape: ") + m.shape->error();
            return false;
        }
        m.timings.push_back({"sparse structure", sr.seconds_ss});
        m.timings.push_back({"shape SLAT (512)", sr.seconds_slat});
        if (cascade) m.timings.push_back({"cascade (1024)", sr.seconds_hr});
        m.timings.push_back({"shape decoder", sr.seconds_dec});
        m.timings.push_back({"mesh", sr.seconds_mesh});
        std::vector<float> sv = std::move(sr.verts);
        std::vector<int32_t> sf = std::move(sr.faces);
        if (opt.solid) {
            // the closed solid a slicer needs (trellis_solid.h): the mesher's surface has holes, turned faces and loose pieces
            t0 = std::chrono::steady_clock::now();
            report("solid", 0, 0);
            std::vector<float> ov;
            std::vector<int32_t> of;
            SolidStats sost;
            SolidConfig scfg;
            if (opt.hollow_mm > 0) {
                // millimetres to the mesh's units: encode_stl makes the largest side stl_size_mm
                float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
                for (size_t i = 0; i < sv.size(); ++i) {
                    lo[i % 3] = std::min(lo[i % 3], sv[i]);
                    hi[i % 3] = std::max(hi[i % 3], sv[i]);
                }
                const double side = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
                scfg.wall = (float)(opt.hollow_mm * side / opt.stl_size_mm);
                scfg.drain = (float)(opt.drain_mm * side / opt.stl_size_mm);
            }
            if (!solidify(sv, sf, scfg, ov, of, sost, err)) {
                m.error = "solid: " + err;
                return false;
            }
            sv.swap(ov);
            sf.swap(of);
            m.timings.push_back({"solid", seconds_since(t0)});
        }
        t0 = std::chrono::steady_clock::now();
        report("simplify", 0, 0);
        SimplifyStats sst;
        if (!simplify_to_faces(sv, sf, opt.faces, sst, err, opt.solid)) {
            m.error = "simplify: " + err;
            return false;
        }
        m.timings.push_back({"simplify", seconds_since(t0)});
        t0 = std::chrono::steady_clock::now();
        const bool stl = opt.format == ImageTo3DOptions::Format::Stl;
        report(stl ? "stl" : "glb", 0, 0);
        if (!(stl ? encode_stl(sv, sf, opt.stl_size_mm, glb, err) : encode_glb_mesh(sv, sf, glb, err))) {
            m.error = std::string(stl ? "stl: " : "glb: ") + err;
            return false;
        }
        m.timings.push_back({stl ? "stl" : "glb", seconds_since(t0)});
        return true;
    }
    TexturedMesh tm;
    const bool run_ok = cascade ? run_textured_cascade(*m.shape, *m.tex, cond_lr.data(), (int)(cond_lr.size() / 1024), cond_hr.data(),
                                                       (int)(cond_hr.size() / 1024), noise, scfg, tcfg, tm, err)
                                : run_textured(*m.shape, *m.tex, cond_lr.data(), (int)(cond_lr.size() / 1024), noise, scfg, tcfg, tm, err);
    if (!run_ok) {
        m.error = err;
        return false;
    }
    m.timings.push_back({"sparse structure", tm.shape.seconds_ss});
    m.timings.push_back({"shape SLAT (512)", tm.shape.seconds_slat});
    if (cascade) m.timings.push_back({"cascade (1024)", tm.shape.seconds_hr});
    m.timings.push_back({"shape decoder", tm.shape.seconds_dec});
    m.timings.push_back({"mesh", tm.shape.seconds_mesh});
    m.timings.push_back({"texture flow", tm.tex.seconds_flow});
    m.timings.push_back({"texture decoder", tm.tex.seconds_dec});

    // 4. the export: the simplification, the charts, the unwrap, the bake, the GLB
    VoxelAttributes attr;
    attr.coords = tm.tex.coords;
    attr.feats = tm.tex.out;
    attr.channels = 6;
    attr.res = opt.resolution;
    std::vector<float> sv = std::move(tm.shape.verts);
    std::vector<int32_t> sf = std::move(tm.shape.faces);
    t0 = std::chrono::steady_clock::now();
    report("simplify", 0, 0);
    SimplifyStats sst;
    if (!simplify_to_faces(sv, sf, opt.faces, sst, err)) {
        m.error = "simplify: " + err;
        return false;
    }
    m.timings.push_back({"simplify", seconds_since(t0)});
    t0 = std::chrono::steady_clock::now();
    report("charts", 0, 0);
    std::vector<int32_t> chart_ids;
    ChartStats cst;
    if (!compute_charts(sv, sf, ChartConfig(), chart_ids, cst, err)) {
        m.error = "charts: " + err;
        return false;
    }
    m.timings.push_back({"charts", seconds_since(t0)});
    t0 = std::chrono::steady_clock::now();
    report("unwrap", 0, 0);
    UvMesh uvm;
    if (!unwrap_uv_by_charts(sv, sf, chart_ids, opt.texture_size, uvm, err)) {
        m.error = "unwrap: " + err;
        return false;
    }
    m.timings.push_back({"unwrap", seconds_since(t0)});
    t0 = std::chrono::steady_clock::now();
    report("bake", 0, 0);
    BakedTextures bt;
    if (!bake_textures(uvm, attr, opt.texture_size, bt, err)) {
        m.error = "bake: " + err;
        return false;
    }
    m.timings.push_back({"bake", seconds_since(t0)});
    t0 = std::chrono::steady_clock::now();
    report("glb", 0, 0);
    if (!encode_glb(uvm, vertex_normals(sv, sf), bt, glb, err)) {
        m.error = "glb: " + err;
        return false;
    }
    m.timings.push_back({"glb", seconds_since(t0)});
    return true;
}

}  // namespace trellis

// ---- the C interface: one job at a time on a background thread
struct t2_pipe {
    std::unique_ptr<trellis::ImageTo3D> model;
    std::mutex mu;
    std::thread th;
    int status = 2;  // 0 running, 1 done, -1 failed, 2 none
    std::string stage, error;
    int step = 0, steps = 0;
    std::chrono::steady_clock::time_point t0, t1;
    std::vector<uint8_t> glb;
};

namespace {
thread_local std::string g_t2_error;
}

extern "C" {

t2_pipe_t* t2_open(const char* trellis2_dir, const char* ss_dec, const char* dinov3, const char* birefnet) {
    try {
        if (!trellis2_dir || !ss_dec || !dinov3) {
            g_t2_error = "t2_open: the TRELLIS.2 directory, the sparse-structure decoder and DINOv3 are needed";
            return nullptr;
        }
        trellis::ImageTo3DFiles f;
        f.trellis2_dir = trellis2_dir;
        f.ss_dec = ss_dec;
        f.dinov3 = dinov3;
        f.birefnet = birefnet ? birefnet : "";
        std::unique_ptr<t2_pipe> p(new t2_pipe);
        p->model.reset(new trellis::ImageTo3D(f));
        if (!p->model->ok()) {
            g_t2_error = p->model->error();
            return nullptr;
        }
        return p.release();
    } catch (const std::exception& e) {
        g_t2_error = e.what();
        return nullptr;
    }
}

void t2_close(t2_pipe_t* p) {
    if (!p) return;
    if (p->th.joinable()) p->th.join();
    delete p;
}

const char* t2_last_error(void) { return g_t2_error.c_str(); }

int t2_has_background_removal(t2_pipe_t* p) { return p && p->model->has_background_removal() ? 1 : 0; }

int t2_start(t2_pipe_t* p, const unsigned char* image, size_t n, int resolution, unsigned long long seed, int faces, int texture_size,
             int remove_background, int format, double stl_size_mm, double hollow_mm, double drain_mm) {
    if (!p) {
        g_t2_error = "t2_start: no pipeline";
        return -1;
    }
    {
        std::lock_guard<std::mutex> lk(p->mu);
        if (p->status == 0) return 1;
    }
    if (p->th.joinable()) p->th.join();
    if (resolution != 512 && resolution != 1024) {
        g_t2_error = "resolution: 512 or 1024";
        return -1;
    }
    if (faces < 1000 || faces > 10000000) {
        g_t2_error = "faces: from 1000 to 10000000";
        return -1;
    }
    if (texture_size != 0 && (texture_size < 256 || texture_size > 8192)) {
        g_t2_error = "texture_size: from 256 to 8192, or 0 for none";
        return -1;
    }
    if (format != 0 && format != 1) {
        g_t2_error = "format: 0 (GLB) or 1 (STL)";
        return -1;
    }
    if (!(stl_size_mm >= 0) || stl_size_mm > 100000) {
        g_t2_error = "the STL's size: 0 to 100000 millimetres";
        return -1;
    }
    if (remove_background < -1 || remove_background > 1) {
        g_t2_error = "remove_background: -1 (auto), 0 or 1";
        return -1;
    }
    if (!(hollow_mm >= 0) || hollow_mm > 1000 || !(drain_mm >= 0) || drain_mm > 1000) {
        g_t2_error = "hollow_mm and drain_mm: 0 to 1000 millimetres";
        return -1;
    }
    if (hollow_mm > 0 && hollow_mm < 2) {
        g_t2_error = "hollow_mm: 0 (solid), or walls of at least 2 millimetres";
        return -1;
    }
    if (hollow_mm > 0 && (format != 1 || !(stl_size_mm > 0))) {
        g_t2_error = "a hollow (hollow_mm) is for an STL with its size in millimetres";
        return -1;
    }
    unsigned char* px = nullptr;
    int w = 0, h = 0, has_alpha = 0;
    if (image_decode_rgba(image, n, &px, &w, &h, &has_alpha) != 0) {
        g_t2_error = "the image cannot be decoded (PNG, JPEG, BMP, GIF or TGA)";
        return -1;
    }
    std::vector<uint8_t> rgba(px, px + (size_t)w * h * 4);
    image_free(px);
    bool opaque = true;
    for (size_t i = 3; i < rgba.size(); i += 4)
        if (rgba[i] != 255) {
            opaque = false;
            break;
        }
    const bool cut_by_alpha = has_alpha && !opaque && remove_background == -1;
    if (remove_background != 0 && !cut_by_alpha && !p->model->has_background_removal()) {
        g_t2_error = "this server removes no backgrounds (no BiRefNet): give an image with an alpha channel, or remove_background false";
        return -1;
    }
    trellis::ImageTo3DOptions o;
    o.resolution = resolution;
    o.seed = seed;
    o.faces = faces;
    o.texture = texture_size != 0;
    o.texture_size = texture_size != 0 ? texture_size : 4096;
    o.remove_background = remove_background;
    o.format = format == 1 ? trellis::ImageTo3DOptions::Format::Stl : trellis::ImageTo3DOptions::Format::Glb;
    o.stl_size_mm = stl_size_mm;
    o.hollow_mm = hollow_mm;
    o.drain_mm = drain_mm;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->status = 0;
        p->stage = "starting";
        p->step = p->steps = 0;
        p->error.clear();
        p->glb.clear();
        p->t0 = p->t1 = std::chrono::steady_clock::now();
    }
    p->th = std::thread([p, rgba = std::move(rgba), w, h, has_alpha, o]() {
        std::vector<uint8_t> glb;
        std::string err;
        bool ok = false;
        try {
            // the device's other users (the image and audio generators) wait while the job runs
            std::lock_guard<std::recursive_mutex> dev(sddev::run_mutex());
            ok = p->model->generate(rgba.data(), w, h, has_alpha != 0, o, glb, [p](const char* stage, int step, int steps) {
                std::lock_guard<std::mutex> lk(p->mu);
                p->stage = stage;
                p->step = step;
                p->steps = steps;
            });
            if (!ok) err = p->model->error();
        } catch (const std::exception& e) {
            ok = false;
            err = e.what();
        }
        std::lock_guard<std::mutex> lk(p->mu);
        p->t1 = std::chrono::steady_clock::now();
        if (ok) {
            p->glb = std::move(glb);
            p->status = 1;
            p->stage = "done";
        } else {
            p->error = err.empty() ? "the job failed" : err;
            p->status = -1;
            p->stage = "failed";
        }
        p->step = p->steps = 0;
    });
    return 0;
}

int t2_poll(t2_pipe_t* p, char* stage, size_t stage_len, int* step, int* steps, double* seconds) {
    if (!p) return 2;
    std::lock_guard<std::mutex> lk(p->mu);
    if (stage && stage_len) {
        std::strncpy(stage, p->stage.c_str(), stage_len - 1);
        stage[stage_len - 1] = 0;
    }
    if (step) *step = p->step;
    if (steps) *steps = p->steps;
    if (seconds)
        *seconds = p->status == 2 ? 0.0
                                  : std::chrono::duration<double>((p->status == 0 ? std::chrono::steady_clock::now() : p->t1) - p->t0).count();
    return p->status;
}

int t2_take(t2_pipe_t* p, unsigned char** glb, size_t* n) {
    if (!p) {
        g_t2_error = "t2_take: no pipeline";
        return -1;
    }
    std::lock_guard<std::mutex> lk(p->mu);
    if (p->status == -1) {
        g_t2_error = p->error;
        p->status = 2;
        return -1;
    }
    if (p->status != 1) {
        g_t2_error = p->status == 0 ? "the job is running" : "no job";
        return -1;
    }
    unsigned char* out = static_cast<unsigned char*>(std::malloc(p->glb.size() ? p->glb.size() : 1));
    if (!out) {
        g_t2_error = "out of memory";
        return -1;
    }
    if (!p->glb.empty()) std::memcpy(out, p->glb.data(), p->glb.size());
    *glb = out;
    *n = p->glb.size();
    p->glb.clear();
    p->glb.shrink_to_fit();
    p->status = 2;
    return 0;
}

void t2_free(void* ptr) { std::free(ptr); }

}  // extern "C"
