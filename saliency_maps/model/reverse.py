#!/usr/bin/env python
"""
Reverse of MedCLIP-SAM v2 convert.py:
HF-style BiomedCLIP checkpoint  →  OpenCLIP .pt
"""
import re, torch, argparse
from collections import defaultdict

# --- minimal inverse regex map (covers every key in BiomedCLIP base) ---
PATTERNS = [
    # vision
    (r"^vision_model\.embeddings\.patch_embedding\.(\w+)$",
     r"visual.trunk.patch_embed.proj.\1"),
    (r"^vision_model\.post_layernorm\.(\w+)$",
     r"visual.trunk.norm.\1"),
    (r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.(\w+)$",
     r"visual.trunk.blocks.\1.norm1.\2"),
    (r"^vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.(\w+)$",
     r"visual.trunk.blocks.\1.norm2.\2"),
    (r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(\w+)$",
     r"visual.trunk.blocks.\1.mlp.fc1.\2"),
    (r"^vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(\w+)$",
     r"visual.trunk.blocks.\1.mlp.fc2.\2"),
    (r"^vision_model\.embeddings\.class_embedding$",
     r"visual.trunk.cls_token"),
    (r"^vision_model\.embeddings\.position_embedding\.weight$",
     r"visual.trunk.pos_embed"),
    # text
    (r"^text_model\.embeddings\.word_embedding\.(\w+)$",
     r"text.transformer.embeddings.word_embeddings.\1"),
    (r"^text_model\.embeddings\.position_embedding\.(\w+)$",
     r"text.transformer.embeddings.position_embeddings.\1"),
    (r"^text_model\.embeddings\.layer_norm\.(\w+)$",
     r"text.transformer.embeddings.LayerNorm.\1"),
    (r"^text_model\.encoder\.layers\.(\d+)\.layer_norm1\.(\w+)$",
     r"text.transformer.encoder.layer.\1.attention.output.LayerNorm.\2"),
    (r"^text_model\.encoder\.layers\.(\d+)\.layer_norm2\.(\w+)$",
     r"text.transformer.encoder.layer.\1.output.LayerNorm.\2"),
    (r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(\w+)$",
     r"text.transformer.encoder.layer.\1.intermediate.dense.\2"),
    (r"^text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(\w+)$",
     r"text.transformer.encoder.layer.\1.output.dense.\2"),
]

def rename(hf_key: str) -> str:
    for pat, repl in PATTERNS:
        if re.match(pat, hf_key):
            return re.sub(pat, repl, hf_key)
    return hf_key  # unchanged (a handful of projections)

def merge_qkv(split_state):
    """Merge q_proj/k_proj/v_proj → qkv expected by OpenCLIP."""
    merged = {}
    buffers = defaultdict(dict)
    for k, v in split_state.items():
        m = re.match(r"(visual\.trunk\.blocks\.(\d+)\.attn)\.(q|k|v)_proj\.(weight|bias)", k)
        if m:
            prefix, blk, which, kind = m.groups()
            buffers[(prefix, kind)][which] = v
        else:
            merged[k] = v
    for (prefix, kind), d in buffers.items():
        merged[f"{prefix}.qkv.{kind}"] = torch.cat(
            (d["q"], d["k"], d["v"]), dim=0 if kind == "weight" else None)
    return merged

def main(in_file, out_file):
    sd = torch.load(in_file, map_location="cpu")
    sd = {rename(k): v for k, v in sd.items()}
    sd = merge_qkv(sd)
    torch.save({"state_dict": sd}, out_file)
    print(f"✔  Wrote OpenCLIP checkpoint → {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf",  required=True,
                   help="Path to MedCLIP-SAM v2 pytorch_model.bin")
    p.add_argument("--out", default="medclipsamv2_openclip.pt")
    args = p.parse_args()
    main(args.hf, args.out)
