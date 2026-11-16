"""
Context extraction, validation, and forwarding for SENTINEL Layer 1 (L1).
Extracts and validates an allowlist of context fields from SentinelRequest.metadata
and SentinelRequest.transaction to construct an L1In object and an audit report.
"""
import re
from typing import Any, Dict, List, Optional, Tuple
from sentinel_contracts import L1In, SentinelRequest

# Limits and character sets
MAX_PAGE_CONTENT_CHARS = 1_000_000
MAX_PAGE_HTML_CHARS = 1_000_000
MAX_INVOICE_PDF_B64_CHARS = 5_000_000
MAX_PAYEE_ACCOUNT_CHARS = 64
MAX_PAYEE_DOMAIN_CHARS = 253
MAX_REDIRECT_COUNT = 50

_B64_PATTERN = re.compile(r"^[A-Za-z0-9+/=]+$")
_HOSTNAME_PATTERN = re.compile(r"^[a-zA-Z0-9.-]+$")

# Allowlist definition
ALLOWED_METADATA_KEYS = {
    "page_content",
    "page_html",
    "invoice_pdf_b64",
    "payee_account",
    "payee_domain",
    "redirect_count",
}


def build_l1_input(request: SentinelRequest) -> Tuple[L1In, Dict[str, Any]]:
    """
    Build L1In and context report from SentinelRequest.

    Validation rules:
    - metadata["page_content"]: str, <= 1,000,000 chars -> L1In.page_content
    - metadata["page_html"]: str, <= 1,000,000 chars -> L1In.payload["html"]
    - metadata["invoice_pdf_b64"]: str, base64 charset, <= 5,000,000 chars -> L1In.payload["pdf_b64"]
    - transaction/metadata["payee_account"]: str, stripped, 1..64 chars, no internal whitespace
      (first match wins: transaction, then metadata) -> L1In.payload["payee_account"]
    - transaction/metadata["payee_domain"]: str, 1..253 chars, hostname charset
      (first match wins: transaction, then metadata) -> L1In.payload["payee_domain"]
    - metadata["redirect_count"]: int (not bool), 0..50 -> L1In.payload["redirect_count"]

    Invalid or oversized values are dropped, never truncated or coerced.
    The report contains forwarded key names and rejected key names with reason codes, never content.
    """
    metadata = request.metadata if request.metadata is not None else {}
    transaction = request.transaction if request.transaction is not None else {}

    forwarded: List[str] = []
    rejected: Dict[str, str] = {}
    payload: Dict[str, Any] = {}
    page_content: Optional[str] = None

    # 1. Reject unknown or disallowed metadata keys
    for k in metadata:
        if k not in ALLOWED_METADATA_KEYS:
            rejected[k] = "DISALLOWED_FIELD" if k == "payload" else "UNKNOWN_FIELD"

    # 2. metadata["page_content"]
    if "page_content" in metadata:
        val = metadata["page_content"]
        if not isinstance(val, str):
            rejected["page_content"] = "TYPE_MISMATCH"
        elif len(val) > MAX_PAGE_CONTENT_CHARS:
            rejected["page_content"] = "OVERSIZED"
        else:
            page_content = val
            forwarded.append("page_content")

    # 3. metadata["page_html"] -> payload["html"]
    if "page_html" in metadata:
        val = metadata["page_html"]
        if not isinstance(val, str):
            rejected["page_html"] = "TYPE_MISMATCH"
        elif len(val) > MAX_PAGE_HTML_CHARS:
            rejected["page_html"] = "OVERSIZED"
        else:
            payload["html"] = val
            forwarded.append("page_html")

    # 4. metadata["invoice_pdf_b64"] -> payload["pdf_b64"]
    if "invoice_pdf_b64" in metadata:
        val = metadata["invoice_pdf_b64"]
        if not isinstance(val, str):
            rejected["invoice_pdf_b64"] = "TYPE_MISMATCH"
        elif len(val) > MAX_INVOICE_PDF_B64_CHARS:
            rejected["invoice_pdf_b64"] = "OVERSIZED"
        elif not _B64_PATTERN.fullmatch(val):
            rejected["invoice_pdf_b64"] = "INVALID_CHARSET"
        else:
            payload["pdf_b64"] = val
            forwarded.append("invoice_pdf_b64")

    # 5. payee_account (first match wins: transaction, then metadata) -> payload["payee_account"]
    account_val = None
    account_source = None
    if "payee_account" in transaction:
        account_val = transaction["payee_account"]
        account_source = "transaction"
    elif "payee_account" in metadata:
        account_val = metadata["payee_account"]
        account_source = "metadata"

    if account_source is not None:
        if not isinstance(account_val, str):
            rejected["payee_account"] = "TYPE_MISMATCH"
        else:
            stripped = account_val.strip()
            if not (1 <= len(stripped) <= MAX_PAYEE_ACCOUNT_CHARS):
                rejected["payee_account"] = "INVALID_LENGTH"
            elif any(c.isspace() for c in stripped):
                rejected["payee_account"] = "INVALID_FORMAT_WHITESPACE"
            else:
                payload["payee_account"] = stripped
                forwarded.append("payee_account")

    # 6. payee_domain (first match wins: transaction, then metadata) -> payload["payee_domain"]
    domain_val = None
    domain_source = None
    if "payee_domain" in transaction:
        domain_val = transaction["payee_domain"]
        domain_source = "transaction"
    elif "payee_domain" in metadata:
        domain_val = metadata["payee_domain"]
        domain_source = "metadata"

    if domain_source is not None:
        if not isinstance(domain_val, str):
            rejected["payee_domain"] = "TYPE_MISMATCH"
        elif not (1 <= len(domain_val) <= MAX_PAYEE_DOMAIN_CHARS):
            rejected["payee_domain"] = "INVALID_LENGTH"
        elif not _HOSTNAME_PATTERN.fullmatch(domain_val):
            rejected["payee_domain"] = "INVALID_CHARSET"
        else:
            payload["payee_domain"] = domain_val
            forwarded.append("payee_domain")

    # 7. metadata["redirect_count"] -> payload["redirect_count"]
    if "redirect_count" in metadata:
        val = metadata["redirect_count"]
        if isinstance(val, bool) or not isinstance(val, int):
            rejected["redirect_count"] = "TYPE_MISMATCH"
        elif not (0 <= val <= MAX_REDIRECT_COUNT):
            rejected["redirect_count"] = "OUT_OF_RANGE"
        else:
            payload["redirect_count"] = val
            forwarded.append("redirect_count")

    l1_input = L1In(
        request_id=request.request_id,
        agent_id=request.agent_id,
        user_id=request.user_id,
        transaction_id=request.transaction_id,
        url=request.url,
        page_content=page_content,
        payload=payload,
    )

    report = {
        "forwarded": forwarded,
        "rejected": rejected,
    }

    return l1_input, report
