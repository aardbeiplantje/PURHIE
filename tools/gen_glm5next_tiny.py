#!/usr/bin/env python3
# gen_glm5next_tiny.py -- a small random-weight glm5_next (GLM-5.3-Flash) GGUF for the oracle tests.
#
#   gen_glm5next_tiny.py <out.gguf> [seed]
#
# The real model (320B) does not fit here, so the architecture is verified on this file instead: the same
# tensor names and layer pattern as the release, with small widths. Weights are random blocks scaled to about
# 1/sqrt(K), so activations stay O(1) through the layers. Pure Python (no numpy): its own GGUF reader / writer.
#
# Architecture: 320B total / 18B active, 45 decoder layers, hidden 4096, vocab 154,880, 1M context.
# 34 linear-attention layers (KDA) + 11 sparse-attention layers (MLA) at layers 3, 7, 11 ... 43.
# 42 MoE layers (first 3 layers dense FFN), 288 routed experts + 1 shared, top-8, sigmoid routing.
# This tiny version: 4 layers (3 dense + 1 MLA/MoE), hidden 512, small expert count.

import math, os, random, struct, sys


# ---------------------------------------------------------------- GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCAL = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}


def read_header_kv(path):
    """The metadata of a GGUF (only the header is needed), as [(key, type, value, subtype)]."""
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8")
    assert f.read(4) == b"GGUF"
    rd("I"); rd("Q"); nkv = rd("Q")
    out = []
    for _ in range(nkv):
        k = rs(); t = rd("I")
        if t == ARR:
            st, n = rd("I"), rd("Q")
            v = [rs() if st == STR else rd(SCAL[st]) for _ in range(n)]
            out.append((k, t, v, st))
        else:
            out.append((k, t, rs() if t == STR else rd(SCAL[t]), None))
    return out


class Writer:
    def __init__(self):
        self.kv = []       # (key, type, value, subtype)
        self.tensors = []  # (name, dims, ggml type, bytes)

    def add(self, key, t, v, st=None):
        self.kv.append((key, t, v, st))

    def tensor(self, name, dims, gtype, data):
        self.tensors.append((name, dims, gtype, bytes(data)))

    def write(self, path, align=32):
        def s(x):
            b = x.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        def val(t, v):
            return s(v) if t == STR else struct.pack("<" + SCAL[t], v)
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for k, t, v, st in self.kv:
            out += s(k) + struct.pack("<I", t)
            if t == ARR:
                out += struct.pack("<IQ", st, len(v)) + b"".join(val(st, x) for x in v)
            else:
                out += val(t, v)
        off = 0
        for name, dims, gtype, data in self.tensors:
            out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            out += struct.pack("<IQ", gtype, off)
            off += (len(data) + align - 1) // align * align
        out += b"\0" * (-len(out) % align)
        with open(path, "wb") as f:
            f.write(out)
            for _, _, _, data in self.tensors:
                f.write(data)
                f.write(b"\0" * (-len(data) % align))


# ---------------------------------------------------------------- random quantized blocks
T_F32, T_F16, T_BF16 = 0, 1, 30
T_Q4_0, T_Q8_0, T_FP4, T_FP4F = 2, 8, 100, 101
rng = random.Random(1)


def f16(x):
    return struct.pack("<e", x)


def ue4m3_byte(d):
    """The ROCmFP4 scale byte whose half-scale value is closest to d (ue4m3_half in quant_types.cpp)."""
    best, bb = None, 0
    for e in range(0x7f):
        ex, m = e >> 3, e & 7
        v = m * 2.0 ** -10 if ex == 0 else (8 + m) * 2.0 ** (ex - 11)
        if best is None or abs(v - d) < abs(best - d):
            best, bb = v, e
    return bb


def quant(gtype, n_values, std):
    """n_values random weights of the given type with about this standard deviation."""
    if gtype == T_F32:
        return b"".join(struct.pack("<f", rng.gauss(0, std)) for _ in range(n_values))
    if gtype == T_F16:
        return b"".join(f16(rng.gauss(0, std)) for _ in range(n_values))
    if gtype == T_BF16:  # the top half of the float32
        return b"".join(struct.pack("<f", rng.gauss(0, std))[2:] for _ in range(n_values))
    nb = n_values // 32
    out = bytearray()
    if gtype in (T_FP4, T_FP4F):  # codebook rms ~5.36 at half scale
        scales = [ue4m3_byte(std / 5.36 * rng.uniform(0.7, 1.4)) for _ in range(8)]
        body = os.urandom(16 * nb)
        for i in range(nb):
            out += body[16 * i:16 * i + 16]
            out.append(scales[i % 8])
            if gtype == T_FP4:
                out.append(scales[(i + 3) % 8])
    elif gtype == T_Q8_0:  # d * q, q uniform -127..127 (std ~73.6)
        d = std / 73.6
        for i in range(nb):
            out += f16(d) + bytes(rng.randrange(-127, 128) & 0xFF for _ in range(32))
    elif gtype == T_Q4_0:  # d * (q - 8), q uniform 0..15 (std ~4.6)
        d = std / 4.6
        body = os.urandom(16 * nb)
        for i in range(nb):
            out += f16(d) + body[16 * i:16 * i + 16]
    else:
        raise ValueError(gtype)
    return out


def main():
    path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rng.seed(seed)

    # Tiny model widths (scaled down from 320B release)
    # Release: hidden 4096, 45 layers, 288 experts + 1 shared, top-8, 154,880 vocab
    # Tiny: hidden 512, 4 layers (3 dense + 1 MLA/MoE), 16 experts + 1 shared, top-2

    E = 512                       # embedding dimension
    NL = 4                        # total layers (3 dense + 1 MLA)
    ND = 3                        # dense FFN layers (first 3)
    HQ = 8                        # query heads
    HKV = 4                       # KV heads (for MLA: n_head_kv concept differs)
    HD = 128                      # head_dim
    NE = 16                       # expert count
    NU = 2                        # experts used per token (top-k)
    FE = 256                      # expert FFN width
    FS = 256                      # shared expert FFN width
    KV_LR = 64                    # kv_lora_rank (512 in release, scaled down)
    Q_LR = 128                    # q_lora_rank (1536 in release, scaled down)
    IH = 4                        # indexer head count
    IK = 64                       # indexer key length (128 in release)
    TOP_K = 64                    # indexer top-k (2048 in release)
    KPOOL = 4                     # key pooling factor (4 in release)
    MHC_MULT = 4                  # hyper-connection multiplier (4 in release)
    HC_EPS = 0.01                 # Sinkhorn convergence threshold

    s = lambda K: 1.0 / math.sqrt(K)

    w = Writer()
    a = "glm5_next."
    w.add("general.architecture", STR, "glm5_next")
    w.add("general.name", STR, "glm5-next-tiny-random")
    w.add("general.type", STR, "model")
    w.add("general.file_type", U32, 0)
    w.add("general.base_model_count", U32, 1)

    # Core architecture params
    for k, v, ty in [("context_length", 8192, U32), ("embedding_length", E, U32),
                   ("attention.head_count", HQ, U32), ("attention.head_count_kv", HKV, U32),
                   ("block_count", NL, U32), ("rope.freq_base", 10000.0, F32),
                   ("attention.layer_norm_rms_epsilon", 1e-5, F32),
                   ("expert_count", NE, U32), ("expert_used_count", NU, U32),
                   ("expert_feed_forward_length", FE, U32), ("expert_shared_feed_forward_length", FS, U32),
                   ("attention.key_length", HD, U32), ("attention.value_length", HD, U32)]:
        w.add(a + k, ty, v)

    # MLA parameters
    w.add(a + "attention.q_lora_rank", U32, Q_LR)
    w.add(a + "attention.kv_lora_rank", U32, KV_LR)
    w.add(a + "attention.key_length_mla", U32, HD)
    w.add(a + "attention.value_length_mla", U32, HD)
    w.add(a + "attention.qk_rope_head_dim", U32, 0)
    w.add(a + "attention.mla_use_nope", BOOL, False)

    # DSA indexer parameters
    w.add(a + "attention.indexer.head_count", U32, IH)
    w.add(a + "attention.indexer.key_length", U32, IK)
    w.add(a + "attention.indexer.top_k", U32, TOP_K)
    w.add(a + "attention.index_kpool", U32, KPOOL)
    w.add(a + "attention.index_share_for_mtp_iteration", BOOL, True)

    # MoE routing
    w.add(a + "mlp.gate.e_score_correction_bias", F32, 0.0)
    w.add(a + "expert_gating_function", STR, "sigmoid")

    # Hyper-connections
    w.add(a + "hyper_connection.count", U32, MHC_MULT)
    w.add(a + "hyper_connection.low_rank", U32, E // 2)
    w.add(a + "hyper_connection.sinkhorn_iters", U32, 20)
    w.add(a + "hyper_connection.epsilon", F32, HC_EPS)

    # Model-specific
    w.add(a + "swiglu_limit", F32, 10.0)
    w.add(a + "first_dense_layers", U32, ND)
    w.add(a + "nextn_predict_layers", U32, 0)

    # KV cache layout
    w.add(a + "attention.kv_bytes_per_token", U64, KV_LR * HD * 2)

    # Dense FFN width (for the first 3 layers)
    w.add(a + "feed_forward_length", U32, E * 4)

    # Copy tokenizer metadata from a reference model if available
    ref_kv = None
    if len(sys.argv) > 3:
        ref_kv = read_header_kv(sys.argv[3])
        for k, t, v, st in ref_kv:
            if k.startswith("tokenizer."):
                w.add(k, t, v, st)

    # Set vocab size from reference or default
    V = 1024  # tiny vocab for testing
    if ref_kv:
        for k, t, v, st in ref_kv:
            if k == "tokenizer.ggml.vocab_size":
                V = int(v)
                break
            if k == "tokenizer.ggml.tokens" and t == ARR:
                V = len(v)
                break
    w.add("general.vocab_size", U32, V)

    # Helper functions
    def rand_f32(n, std=1.0):
        return b"".join(struct.pack("<f", rng.gauss(0, std)) for _ in range(n))

    def rand_f16(n, std=1.0):
        return b"".join(f16(rng.gauss(0, std)) for _ in range(n))

    # Global tensors
    w.tensor("token_embd.weight", [E, V], T_Q4_0, quant(T_Q4_0, E * V, s(E)))
    w.tensor("output_norm.weight", [E], T_F32, rand_f32(E))
    w.tensor("output.weight", [E, V], T_Q4_0, quant(T_Q4_0, E * V, s(E)))

    # Output hyper-connections (similar to qwen4exp output_hc_*)
    HCD = MHC_MULT * E
    w.tensor("output_hc_norm.weight", [HCD], T_F32, rand_f32(HCD))
    w.tensor("output_hc_down.weight", [HCD, E // 2], T_FP4F, quant(T_FP4F, HCD * (E // 2), s(HCD)))
    w.tensor("output_hc_up.weight", [E // 2, HCD], T_FP4F, quant(T_FP4F, (E // 2) * HCD, s(HCD)))

    # Per-layer tensors
    for l in range(NL):
        b = f"blk.{l}."
        is_dense = l < ND
        is_mla = (l == NL - 1)  # last layer is MLA/MoE

        if is_dense:
            # Dense FFN layers (first 3 layers)
            # Attention: standard GQA with MLA-style projections
            # qkv_a_mqa: [n_embd, kv_lora_rank + qk_rope_head_dim]
            w.tensor(b + "attn_norm.weight", [E], T_F32, rand_f32(E))

            # MLA Q projection (low-rank)
            w.tensor(b + "attn_q_a.weight", [E, Q_LR], T_FP4F, quant(T_FP4F, E * Q_LR, s(E)))
            w.tensor(b + "attn_q_b.weight", [Q_LR, HQ * HD], T_FP4F, quant(T_FP4F, Q_LR * HQ * HD, s(Q_LR)))

            # MLA KV projection (MQA)
            w.tensor(b + "attn_kv_a.weight", [E, KV_LR], T_FP4F, quant(T_FP4F, E * KV_LR, s(E)))
            w.tensor(b + "attn_k_b.weight", [KV_LR, HKV * HD], T_FP4, quant(T_FP4, KV_LR * HKV * HD, s(KV_LR)))
            w.tensor(b + "attn_v_b.weight", [KV_LR, HKV * HD], T_FP4, quant(T_FP4, KV_LR * HKV * HD, s(KV_LR)))

            # Output projection
            w.tensor(b + "attn_output.weight", [HQ * HD, E], T_FP4F, quant(T_FP4F, HQ * HD * E, s(HQ * HD)))

            # Output norm
            w.tensor(b + "post_attention_norm.weight", [E], T_F32, rand_f32(E))

            # Dense FFN (SwiGLU with swiglu_limit clamp)
            FF = E * 4
            w.tensor(b + "ffn_norm.weight", [E], T_F32, rand_f32(E))
            w.tensor(b + "ffn_gate.weight", [E, FF], T_FP4F, quant(T_FP4F, E * FF, s(E)))
            w.tensor(b + "ffn_up.weight", [E, FF], T_FP4F, quant(T_FP4F, E * FF, s(E)))
            w.tensor(b + "ffn_down.weight", [FF, E], T_FP4, quant(T_FP4, FF * E, s(FF)))

        elif is_mla:
            # MLA + MoE layer (last layer)
            # Attention with DSA indexer
            w.tensor(b + "attn_norm.weight", [E], T_F32, rand_f32(E))

            # MLA projections
            w.tensor(b + "attn_q_a.weight", [E, Q_LR], T_FP4F, quant(T_FP4F, E * Q_LR, s(E)))
            w.tensor(b + "attn_q_b.weight", [Q_LR, HQ * HD], T_FP4F, quant(T_FP4F, Q_LR * HQ * HD, s(Q_LR)))
            w.tensor(b + "attn_kv_a.weight", [E, KV_LR], T_FP4F, quant(T_FP4F, E * KV_LR, s(E)))
            w.tensor(b + "attn_k_b.weight", [KV_LR, HKV * HD], T_FP4, quant(T_FP4, KV_LR * HKV * HD, s(KV_LR)))
            w.tensor(b + "attn_v_b.weight", [KV_LR, HKV * HD], T_FP4, quant(T_FP4, KV_LR * HKV * HD, s(KV_LR)))
            w.tensor(b + "attn_output.weight", [HQ * HD, E], T_FP4F, quant(T_FP4F, HQ * HD * E, s(HQ * HD)))

            # DSA Indexer tensors
            w.tensor(b + "indexer.k_norm.weight", [IK], T_F32, rand_f32(IK))
            w.tensor(b + "indexer.k_norm.bias", [IK], T_F32, rand_f32(IK))
            w.tensor(b + "indexer.proj.weight", [E, IH * IK], T_FP4F, quant(T_FP4F, E * IH * IK, s(E)))
            w.tensor(b + "indexer.attn_k.weight", [E, IK], T_FP4F, quant(T_FP4F, E * IK, s(E)))
            w.tensor(b + "indexer.attn_q_b.weight", [Q_LR, IH * IK], T_FP4F, quant(T_FP4F, Q_LR * IH * IK, s(Q_LR)))

            # Post attention norm
            w.tensor(b + "post_attention_norm.weight", [E], T_F32, rand_f32(E))

            # MoE FFN (sigmoid routing with correction bias)
            w.tensor(b + "ffn_norm.weight", [E], T_F32, rand_f32(E))
            w.tensor(b + "ffn_gate_inp.weight", [E, NE], T_F32, rand_f32(E * NE))
            w.tensor(b + "ffn_gate_exps.weight", [E, FE, NE], T_FP4F, quant(T_FP4F, E * FE * NE, s(E)))
            w.tensor(b + "ffn_up_exps.weight", [E, FE, NE], T_FP4F, quant(T_FP4F, E * FE * NE, s(E)))
            w.tensor(b + "ffn_down_exps.weight", [FE, E, NE], T_FP4F, quant(T_FP4F, FE * E * NE, s(FE)))

            # Shared expert
            w.tensor(b + "ffn_gate_shexp.weight", [E, FS], T_FP4F, quant(T_FP4F, E * FS, s(E)))
            w.tensor(b + "ffn_up_shexp.weight", [E, FS], T_FP4F, quant(T_FP4F, E * FS, s(E)))
            w.tensor(b + "ffn_down_shexp.weight", [FS, E], T_FP4F, quant(T_FP4F, FS * E, s(FS)))

            # Sigmoid routing bias
            w.tensor(b + "ffn_gate_inp.bias", [NE], T_F32, rand_f32(NE))

        # Print progress
        print(f"  Layer {l}: {'dense' if is_dense else 'MLA/MoE'}")

    w.write(path)
    print(f"{path}: {len(w.tensors)} tensors, {NL} layers (ND={ND} dense, {NL-ND} MLA/MoE), hidden {E}, vocab {V}")
    print(f"  Experts: {NE} total, {NU} used, FF {FE}/{FS} (shared)")
    print(f"  MLA: q_lora_rank={Q_LR}, kv_lora_rank={KV_LR}, head_dim={HD}")
    print(f"  Indexer: heads={IH}, key_len={IK}, top_k={TOP_K}, kpool={KPOOL}")


if __name__ == "__main__":
    main()
