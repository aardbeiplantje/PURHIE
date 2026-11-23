# AudioLDM 2's tokenizers as plain text for the engine's test (the server reads the tokenizer.json files in Perl):
# /data/audioldm2-stages/clap_vocab.txt (one token per line, by id), clap_merges.txt ("a b" per line), t5_pieces.tsv ("piece<TAB>score").
import json
OUT = '/data/audioldm2-stages/'
t = json.load(open('/data/audioldm2/tokenizer/tokenizer.json'))
vocab = t['model']['vocab']
inv = [None] * (max(vocab.values()) + 1)
for k, v in vocab.items():
    inv[v] = k
assert all(x is not None for x in inv)
open(OUT + 'clap_vocab.txt', 'w', encoding='utf-8').write('\n'.join(inv) + '\n')
open(OUT + 'clap_merges.txt', 'w', encoding='utf-8').write('\n'.join(x if isinstance(x, str) else ' '.join(x) for x in t['model']['merges']) + '\n')
t5 = json.load(open('/data/audioldm2/tokenizer_2/tokenizer.json'))
with open(OUT + 't5_pieces.tsv', 'w', encoding='utf-8') as f:
    for p, s in t5['model']['vocab']:
        f.write(p + '\t' + repr(float(s)) + '\n')
print('clap vocab', len(inv), 't5 pieces', len(t5['model']['vocab']))
