#!/bin/sh
# fetch_meta.sh REPO DIR: a Hugging Face release's small files (config, tokenizer, chat template, the safetensors
# index, the sentence-transformers modules), no weights: the input of gen_tiny_ckpt.py
repo=$1; dir=$2
mkdir -p "$dir"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
         chat_template.jinja chat_template.json tokenizer.model added_tokens.json vocab.json merges.txt \
         model.safetensors.index.json preprocessor_config.json processor_config.json modules.json \
         config_sentence_transformers.json 1_Pooling/config.json 2_Dense/config.json 3_Dense/config.json; do
    mkdir -p "$dir/$(dirname "$f")"
    code=$(curl -sS -m 60 -L -o "$dir/$f.tmp" -w '%{http_code}' "https://huggingface.co/$repo/resolve/main/$f")
    if [ "$code" = 200 ]; then mv "$dir/$f.tmp" "$dir/$f"; echo "$repo: $f"; else rm -f "$dir/$f.tmp"; fi
done
find "$dir" -type d -empty -delete
