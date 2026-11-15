"""URL lexical features. Pure function of the URL string: identical in training and serving."""
import ipaddress
import math
import re
from collections import Counter
from typing import Dict
from urllib.parse import urlsplit

from ..rules._domain import split_host
from ..rules.r10_shortener import SHORTENERS

_VOWELS = set("aeiou")
_HEX = set("0123456789abcdef")
_SPECIAL = set("@?&=_%~$!*+,;")
_TOKEN = re.compile(r"[A-Za-z0-9]+")

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "n_dots", "n_hyphens", "n_digits", "n_special",
    "host_hyphens", "host_digits", "host_labels", "max_label_len", "n_subdomains",
    "digit_ratio", "host_digit_ratio", "entropy_host", "entropy_path",
    "longest_token", "vowel_ratio_host", "hex_ratio_pathquery",
    "path_depth", "n_query_params",
    "has_at", "has_port", "is_ip_host", "has_punycode", "is_shortener",
    "is_https", "is_homepage", "double_slash_in_path",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in Counter(s).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def url_features(url: str) -> Dict[str, float]:
    u = (url or "").strip()
    if "://" not in u:
        u = "http://" + u
    try:
        s = urlsplit(u)
        host = (s.hostname or "").lower()
        try:
            has_port = s.port is not None
        except ValueError:
            has_port = True
    except ValueError:
        s, host, has_port = urlsplit("http://invalid/"), "", False

    path = s.path or "/"
    query = s.query
    ip = _is_ip(host)
    reg = "" if ip else split_host(host)[2]
    n_sub = max(0, len(host.split(".")) - len(reg.split("."))) if reg and reg != host else 0
    labels = host.split(".") if host else []
    segs = [x for x in path.split("/") if x]
    letters = [c for c in host if c.isalpha()]
    pq = (path + query).lower()
    pq_alnum = [c for c in pq if c.isalnum()]
    ulen = max(len(u), 1)
    tokens = _TOKEN.findall(u)

    f = {
        "url_len": len(u), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "n_dots": u.count("."), "n_hyphens": u.count("-"),
        "n_digits": sum(c.isdigit() for c in u), "n_special": sum(c in _SPECIAL for c in u),
        "host_hyphens": host.count("-"), "host_digits": sum(c.isdigit() for c in host),
        "host_labels": len(labels), "max_label_len": max((len(x) for x in labels), default=0),
        "n_subdomains": n_sub,
        "digit_ratio": sum(c.isdigit() for c in u) / ulen,
        "host_digit_ratio": sum(c.isdigit() for c in host) / max(len(host), 1),
        "entropy_host": _entropy(host), "entropy_path": _entropy(path),
        "longest_token": max((len(t) for t in tokens), default=0),
        "vowel_ratio_host": (sum(c in _VOWELS for c in letters) / len(letters)) if letters else 0.0,
        "hex_ratio_pathquery": (sum(c in _HEX for c in pq_alnum) / len(pq_alnum)) if pq_alnum else 0.0,
        "path_depth": len(segs), "n_query_params": len(query.split("&")) if query else 0,
        "has_at": "@" in u, "has_port": has_port, "is_ip_host": ip,
        "has_punycode": any(x.startswith("xn--") for x in labels),
        "is_shortener": reg in SHORTENERS,
        "is_https": s.scheme.lower() == "https",
        "is_homepage": (path == "/" and not query),
        "double_slash_in_path": "//" in path,
    }
    return {k: float(f[k]) for k in FEATURE_NAMES}