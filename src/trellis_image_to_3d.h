#pragma once
// An image to a textured 3D model (TRELLIS.2) on the engine, end to end, for the server and the tools. The object is cut out (the
// image's alpha channel, or BiRefNet when it has none) and cropped as the ComfyUI workflow does (ImageCropToMask: a square around the
// mask with a 1.1 margin, the object over black, 1024 x 1024); then the DINOv3 conditioning, the shape and texture pipelines (512, or
// the 1024 cascade), the simplification, the charts, the UV unwrap, the bake and the binary glTF. The models are opened once (read in
// place and registered for the device); generate runs one image at a time.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trellis {

struct ImageTo3DFiles {
    std::string trellis2_dir;  // the TRELLIS.2-4B directory: pipeline.json (the latent normalisations) and ckpts/*.safetensors
    std::string ss_dec;        // TRELLIS-image-large's ckpts/ss_dec_conv3d_16l8_fp16.safetensors
    std::string dinov3;        // dino_v3_L_naf_fp32.safetensors (ComfyUI's clip_vision copy)
    std::string birefnet;      // birefnet.safetensors (optional: without it only images with an alpha channel are cut out)
};

struct ImageTo3DOptions {
    int resolution = 1024;       // 512, or 1024 (the cascade)
    uint64_t seed = 0;           // the noise (a host Gaussian, not python's generator)
    int faces = 700000;          // the simplification's target (the workflow's DecimateMesh)
    int texture_size = 4096;     // the UV atlas and the bake, texels per side (the workflow's 4096)
    int remove_background = -1;  // -1: the alpha when the image has one, else BiRefNet; 0: the whole image; 1: BiRefNet always
    int slat_lr_steps = 20;      // the cascade's 512 shape-SLAT steps (the workflow's 20)
    bool texture = true;         // false: the mesh alone (no texture flow, unwrap or bake: faster)
    enum class Format { Glb, Stl } format = Format::Glb;  // Stl: a binary STL of the mesh (no texture), for 3D printing
    double stl_size_mm = 100;    // the STL's largest side in millimetres (0: the mesh's own units, a unit cube)
    bool solid = true;           // without texture (the mesh alone, the STL): rebuilt as a closed solid for 3D printing (trellis_solid.h)
    double hollow_mm = 0;        // the STL: hollow, with walls this thick in millimetres, at least 2 (0: solid; cheaper to print)
    double drain_mm = 3;         // with hollow_mm: each hollow's drain hole, this wide, from its lowest point down (0: none)
};

// the progress: the stage and its step (step, steps; 0, 0 for a stage without steps)
using ImageTo3DProgress = std::function<void(const char* stage, int step, int steps)>;

class ImageTo3D {
public:
    explicit ImageTo3D(const ImageTo3DFiles& files);
    ~ImageTo3D();
    ImageTo3D(const ImageTo3D&) = delete;
    ImageTo3D& operator=(const ImageTo3D&) = delete;

    bool ok() const;
    const std::string& error() const;
    bool has_background_removal() const;

    // rgba [h][w][4] (has_alpha false: the image had no alpha channel, its alpha is 255). glb: the model, a binary glTF (textured, or
    // the mesh alone) or a binary STL (opt.format)
    bool generate(const uint8_t* rgba, int w, int h, bool has_alpha, const ImageTo3DOptions& opt, std::vector<uint8_t>& glb,
                  const ImageTo3DProgress& progress = nullptr);

    // the last generate's prepared input (1024 x 1024 RGB: the object over black) and its timings (seconds by stage, in order)
    const std::vector<uint8_t>& last_input() const;
    const std::vector<std::pair<std::string, double>>& last_timings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// The latent normalisations of TRELLIS.2's pipeline.json: shape and texture, mean then std (32 each).
bool read_trellis2_normalisation(const std::string& pipeline_json, float shape_mean[32], float shape_std[32], float tex_mean[32],
                                 float tex_std[32], std::string& error);

// The workflow's crop (ImageCropToMask): rgb [h][w][3] and the mask [h][w] in [0, 1]; the object over black in a square around the mask's
// box (mask > 0.5) times the margin, resized (Lanczos) to size x size. False when the mask is empty.
bool crop_to_mask(const uint8_t* rgb, const float* mask, int w, int h, float margin, int size, std::vector<uint8_t>& out);

}  // namespace trellis
