import base64
import io
import re
from typing import List, Optional, Tuple
from .base import RuleFlag
from . import config

_TOK = re.compile(r"[a-z0-9]{3,}")

def _tokens(s: str) -> set:
    return set(_TOK.findall((s or "").lower()))

def check_r03(pdf_b64: Optional[str]) -> Tuple[List[RuleFlag], List[dict]]:
    if not pdf_b64:
        return [], []
    try:
        import pdfplumber
        import pytesseract
    except ImportError:
        return [], []                      # R03 disabled: dependencies missing
    try:
        raw = base64.b64decode(pdf_b64)
        layer_txt, ocr_txt = [], []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages[:3]:
                layer_txt.append(page.extract_text() or "")
                img = page.to_image(resolution=150).original
                ocr_txt.append(pytesseract.image_to_string(img))
        t, o = _tokens(" ".join(layer_txt)), _tokens(" ".join(ocr_txt))
        missing = t - o                    # in the text layer but not visibly rendered
        if (len(missing) >= config.R03_MIN_MISSING_TOKENS
                and t and len(missing) / len(t) >= config.R03_MISSING_FRACTION):
            hidden = " ".join(sorted(missing))
            return ([RuleFlag("R03", "L1:R03_PDF_TEXT_DIVERGENCE", "step_up",
                              f"{len(missing)}/{len(t)} text-layer tokens not visible in render")],
                    [{"source": "R03", "text": hidden[:2000]}])
    except Exception:
        return [], []
    return [], []