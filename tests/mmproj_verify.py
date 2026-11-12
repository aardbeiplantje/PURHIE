#!/usr/bin/env python3
# Round-trip check of an mmproj GGUF against its checkpoint (phase V2): every model.visual tensor is read back from
# the GGUF with an independent reader and compared bit for bit with the checkpoint's BF16 widened to F32; the
# metadata keys are printed. usage: verify_mmproj.py CHECKPOINT_DIR MMPROJ.gguf
import json, os, struct, sys
import numpy as np

ckpt, path = sys.argv[1], sys.argv[2]

def read_gguf(p):
    with open(p, 'rb') as f:
        data = f.read()
    pos = 0
    def take(fmt):
        nonlocal pos
        v = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return v if len(v) > 1 else v[0]
    def string():
        nonlocal pos
        n = take('<Q')
        s = data[pos:pos + n].decode('utf-8')
        pos += n
        return s
    def value(t):
        scal = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
        if t == 8:
            return string()
        if t == 9:
            et, n = take('<I'), take('<Q')
            return [value(et) for _ in range(n)]
        return take(scal[t])
    assert data[:4] == b'GGUF', 'not a GGUF'
    pos = 4
    version = take('<I')
    n_t, n_kv = take('<Q'), take('<Q')
    kv = {}
    for _ in range(n_kv):
        k = string()
        t = take('<I')
        kv[k] = value(t)
    infos = []
    for _ in range(n_t):
        name = string()
        nd = take('<I')
        dims = [take('<Q') for _ in range(nd)]
        ttype = take('<I')
        off = take('<Q')
        infos.append((name, dims, ttype, off))
    align = kv.get('general.alignment', 32)
    base = (pos + align - 1) // align * align
    tensors = {}
    for name, dims, ttype, off in infos:
        assert ttype == 0, f'{name}: type {ttype} (expected F32 = 0)'
        n = int(np.prod(dims))
        arr = np.frombuffer(data, dtype='<f4', count=n, offset=base + off)
        tensors[name] = arr.reshape(list(reversed(dims)))  # GGUF ne[0] is the fastest-varying dimension
    return version, kv, tensors

def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
    return u.view(np.float32)

idx = json.load(open(os.path.join(ckpt, 'model.safetensors.index.json')))['weight_map']
src = {}
for fn in sorted(set(idx.values())):
    with open(os.path.join(ckpt, fn), 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == '__metadata__' or not k.startswith('model.visual.'):
                continue
            a, b = v['data_offsets']
            f.seek(base + a)
            raw = f.read(b - a)
            assert v['dtype'] == 'BF16', k
            src[k] = bf16_to_f32(raw).reshape(v['shape'])

def gname(n):
    # the names the converter writes (kept in step with vision_gname in Convert.pm)
    import re
    rules = [
        (r'^model\.visual\.patch_embed\.proj\.weight$', 'v.patch_embd.weight'),
        (r'^model\.visual\.patch_embed\.proj\.bias$', 'v.patch_embd.bias'),
        (r'^model\.visual\.pos_embed\.weight$', 'v.position_embd.weight'),
        (r'^model\.visual\.blocks\.(\d+)\.norm([12])\.(weight|bias)$', r'v.blk.\1.ln\2.\3'),
        (r'^model\.visual\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)$', r'v.blk.\1.attn_qkv.\2'),
        (r'^model\.visual\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$', r'v.blk.\1.attn_out.\2'),
        (r'^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc1\.(weight|bias)$', r'v.blk.\1.ffn_up.\2'),
        (r'^model\.visual\.blocks\.(\d+)\.mlp\.linear_fc2\.(weight|bias)$', r'v.blk.\1.ffn_down.\2'),
        (r'^model\.visual\.merger\.norm\.(weight|bias)$', r'v.post_ln.\1'),
        (r'^model\.visual\.merger\.linear_fc1\.(weight|bias)$', r'v.merger.fc1.\1'),
        (r'^model\.visual\.merger\.linear_fc2\.(weight|bias)$', r'v.merger.fc2.\1'),
    ]
    for pat, rep in rules:
        if re.search(pat, n):
            return re.sub(pat, rep, n)
    raise KeyError(n)

version, kv, got = read_gguf(path)
print('gguf version', version, 'tensors', len(got), 'checkpoint vision tensors', len(src))
for k in sorted(kv):
    if not k.startswith('tokenizer'):
        v = kv[k]
        print(f'  {k} = {v if not isinstance(v, list) or len(v) < 8 else v[:8]}')
bad = 0
for k, v in sorted(src.items()):
    g = gname(k)
    if g not in got:
        print('MISSING', g); bad += 1; continue
    want = v.reshape(got[g].shape) if v.size == got[g].size else None
    if k.endswith('patch_embed.proj.weight'):
        want = v.reshape(v.shape[0], -1)  # [out, in*T*H*W]: the pixel rows' order
    if want is None or got[g].shape != want.shape or not np.array_equal(got[g], want):
        print('DIFF', k, '->', g, got[g].shape, want.shape if want is not None else None); bad += 1
extra = sorted(set(got) - {gname(k) for k in src})
print('extra tensors:', extra)
print('ok: all', len(src), 'tensors equal bit for bit' if bad == 0 else f'{bad} problems')
sys.exit(1 if bad else 0)
