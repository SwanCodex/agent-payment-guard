"""Copy the ORIGINAL tokenizer and config files from the Hugging Face repo into the artifact bundle.
 
    python scripts/save_tokenizer.py artifacts/layer2/deberta_v3_base_injection/0.1.0
 
Why copy instead of tokenizer.save_pretrained(): a tokenizer saved by one transformers version can fail to load in
another (for example "AttributeError: 'list' object has no attribute 'keys'" when a v5-saved tokenizer is loaded by
a v4 install). The files published on the hub load in both, so we copy those unchanged.
 
Creates <bundle>/tokenizer/ (about 3 MB). Needs Internet and `pip install huggingface_hub`.
"""
import shutil
import sys
from pathlib import Path
 
from huggingface_hub import snapshot_download
 
MODEL = "microsoft/deberta-v3-base"
FILES = ["config.json", "spm.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]
 
out = Path(sys.argv[1]) / "tokenizer"
if out.exists():
    shutil.rmtree(out)
src = Path(snapshot_download(MODEL, allow_patterns=FILES))
shutil.copytree(src, out)                                   # copies file contents, not cache symlinks
names = sorted(p.name for p in out.iterdir())
print("saved:", names)
missing = [f for f in ("config.json", "spm.model", "tokenizer_config.json") if f not in names]
if missing:
    raise SystemExit(f"missing from the download: {missing}")
 