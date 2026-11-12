// glm5_ref.h -- GLM-5.3-Flash (glm5-next) forward pass on the CPU, token by token: the reference the GPU kernels
// are checked against (tests/glm5next_verify.cpp; it is itself checked against upstream llama.cpp's glm5-next).
// Weights are read in place from the model's mapping (QuantWeight rows decoded one at a time, nothing copied);
// the state (KDA conv windows and delta-rule matrices, the MLA latents, the indexer's keys, gates and pooled keys)
// lives here. One sequence; tokens in order from position 0.

#ifndef GLM5_REF_H
#define GLM5_REF_H

#include <vector>

struct inference_model;

// What one layer computed for one token (for comparing with the reference's named activations)
struct Glm5LayerTrace {
    std::vector<float> attn_in;    // hc_attn_pre: the streams mixed into the block input [n_embd]
    std::vector<float> mixer_out;  // kda_out / attn_out: the token mixer's output [n_embd]
    std::vector<float> attn_post;  // hc_attn_post: the streams after the mixer [n_hc * n_embd]
    std::vector<float> ffn_in;     // hc_ffn_pre [n_embd]
    std::vector<float> ffn_out;    // ffn_out (dense) / the experts + shared expert [n_embd]
    std::vector<int> selected;     // MLA: the positions attended to (ascending)
    std::vector<float> scores;     // MLA with its own indexer: the scores of the complete pools (all visible)
    bool tie = false;              // ... and two of them equal across the top-k boundary: the choice among them is
                                   // the tie rule's (here: the lower pool index; llama.cpp's partial_sort: unordered)
};

struct Glm5Ref {
    explicit Glm5Ref(const inference_model* m);
    void reset();                                          // a new sequence
    // The streams of a token's embedding [n_hc * n_embd]
    void embed(int token, float* streams) const;
    // Layer l on the streams of the token at position pos (positions in order per layer), in place
    void layer(int l, int pos, float* streams, Glm5LayerTrace* tr = nullptr);
    // The LM head's logits [vocab] from the streams after the last layer
    void logits(const float* streams, float* out) const;
    // All layers of one token, then the logits (out may be null)
    void forward(int token, int pos, float* out);

    const inference_model* m;
    // The engine's CPU pass instead of the reference: threads, FP4 weights against Q8_0 activations (expert_cpu.h)
    bool fast = false;
    struct LayerState {
        std::vector<float> conv[3];                        // KDA: the last d_conv - 1 inputs per channel, q / k / v
        std::vector<float> S;                              // KDA: [n_head][head_dim][head_dim], S[i][j]
        std::vector<float> latent;                         // MLA: [pos][kv_lora]
        std::vector<float> ik, ig;                         // indexer: [pos][ix_dim] keys and pool gates
        std::vector<float> pooled;                         // indexer: [pool][ix_dim]
        std::vector<int> sel;                              // the last selection (a shared indexer's input)
    };
    std::vector<LayerState> st;
};

#endif
