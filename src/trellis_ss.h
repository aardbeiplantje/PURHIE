#pragma once
// The TRELLIS.2 sparse-structure flow (ss_flow_img_dit_1_3B_64, the bf16 checkpoint of microsoft/TRELLIS.2-4B) on the
// device: the stages of trellis2/models/sparse_structure_flow.py's forward. The weights are read in place (the checkpoint
// mapped and registered for the GPU); the small vectors (biases, norm gains) are converted to fp32 once at open.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace trellis {

class SsFlow {
public:
    explicit SsFlow(const std::string& safetensors_path);
    ~SsFlow();
    SsFlow(const SsFlow&) = delete;
    SsFlow& operator=(const SsFlow&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // The forward for one step. x: [8][16][16][16] (the model's input, channel-major); t: the timestep; cond: [Lc][1024]
    // (the image features); rope: [4096][64][2] (the model's rope_phases as (cos, sin) pairs); out: [8][16][16][16].
    // When stage is set, it is called after each named stage (fp32 host copy, after the device sync): "h0_input_layer"
    // [4096][1536], "t_embedder" [1536], "mod_adaLN" [9216], "block00", "block01", "block14", "block28", "block29"
    // [4096][1536], "out_pre_layer" [4096][8].
    // The stage check also gets block 0's taps (only when stage is set): "dbg_ln1", "dbg_qk", "dbg_attn", "dbg_self",
    // "dbg_cq_ln", "dbg_cq_lin", "dbg_cq", "dbg_ck", "dbg_cv", "dbg_cross_attn", "dbg_cross" (tools/trellis_ss_block0_oracle.py).
    bool forward(const float* x, float t, const float* cond, int Lc, const float* rope, float* out,
                 const std::function<void(const char*, const float*, size_t)>& stage = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
