from typing import Optional, Tuple
from urllib.parse import urlparse
import tldextract

# suffix_list_urls=() -> use the bundled list, never go to the network
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

def host_of(url: Optional[str]) -> str:
    if not url:
        return ""
    u = url if "://" in url else "//" + url
    return (urlparse(u).hostname or "").lower()

def split_host(host: str) -> Tuple[str, str, str]:
    """returns (domain_label, suffix, registered_domain)"""
    e = _EXTRACT(host)
    if hasattr(e, "top_domain_under_public_suffix"):
        reg = e.top_domain_under_public_suffix
    else:
        reg = e.registered_domain
    return e.domain, e.suffix, reg