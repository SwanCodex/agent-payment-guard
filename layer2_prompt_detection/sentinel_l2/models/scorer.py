"""Raw-margin scorer: text -> z = logit(attack) - logit(benign). Long texts use 256-token windows, max over windows.

torch/transformers are imported lazily so the rest of the package (and its unit tests) work without them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from sentinel_l2.models.bundle import Bundle


def window_starts(n_ids: int, window: int, stride: int) -> list[int]:
    """Start offsets of windows covering n_ids tokens. The last window always reaches the end."""
    starts = list(range(0, max(n_ids - window, 0) + 1, stride)) or [0]
    if starts[-1] + window < n_ids:
        starts.append(n_ids - window)
    return starts


class TorchScorer:
    def __init__(self, bundle: Bundle, device: Optional[str] = None, stride: int = 128, max_windows: int = 6,
                 batch_size: int = 32):
        self.bundle, self.stride, self.max_windows, self.batch_size = bundle, stride, max_windows, batch_size
        self.window = bundle.max_len - 2                     # room for [CLS] and [SEP]
        self._device_pref = device
        self._loaded = False

    def _load(self):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        d: Path = self.bundle.dir
        tok_dir = d / "tokenizer"
        self.tok = AutoTokenizer.from_pretrained(str(tok_dir if tok_dir.is_dir() else self.bundle.model_name))
        cfg_src = str(tok_dir) if (tok_dir / "config.json").exists() else self.bundle.model_name
        cfg = AutoConfig.from_pretrained(cfg_src, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg)      # no base-weight download
        weights = d / "model.pt" if (d / "model.pt").exists() else d / "model_fp16.pt"
        model.load_state_dict(torch.load(weights, map_location="cpu"))   # fp16 file -> fp32 weights
        self.device = torch.device(self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.float().to(self.device).eval()
        self.torch = torch
        self._loaded = True

    def score(self, texts: Sequence[str]) -> np.ndarray:
        if not self._loaded:
            self._load()
        torch, tok = self.torch, self.tok
        chunks, owner = [], []
        for i, txt in enumerate(texts):
            ids = tok(txt, add_special_tokens=False, truncation=True,
                      max_length=self.window + self.stride * (self.max_windows - 1))["input_ids"]
            for s in window_starts(len(ids), self.window, self.stride):
                chunks.append([tok.cls_token_id] + ids[s:s + self.window] + [tok.sep_token_id])
                owner.append(i)
        z = np.full(len(texts), -1e9)
        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            for j in range(0, len(chunks), self.batch_size):
                feats = [{"input_ids": c, "attention_mask": [1] * len(c)} for c in chunks[j:j + self.batch_size]]
                b = {k: v.to(self.device) for k, v in tok.pad(feats, return_tensors="pt").items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lg = self.model(**b).logits
                m = (lg[:, 1] - lg[:, 0]).float().cpu().numpy()
                for k, v in enumerate(m):
                    z[owner[j + k]] = max(z[owner[j + k]], v)
        return z
