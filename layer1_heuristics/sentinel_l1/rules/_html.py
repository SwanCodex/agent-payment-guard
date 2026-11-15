from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import List, Tuple

_VOID = {"br", "img", "input", "meta", "link", "hr", "area", "base", "col", "embed",
         "source", "track", "wbr"}
_HIDE = ("display:none", "visibility:hidden", "font-size:0", "opacity:0",
         "left:-9999", "top:-9999", "text-indent:-9999")

@dataclass
class ParsedHTML:
    visible: str = ""
    hidden: str = ""
    anchors: List[Tuple[str, str]] = field(default_factory=list)   # (href, link text)

class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []          # (tag, is_hidden, is_skip)
        self.depth = 0
        self.skip = 0
        self.vis, self.hid, self.anchors = [], [], []
        self.cur = None

    def handle_starttag(self, tag, attrs):
        if tag in _VOID:
            return
        a = dict(attrs)
        style = (a.get("style") or "")
        if style:
            style = style.lower().replace(" ", "")
        hid = ("hidden" in a) or (bool(style) and any(h in style for h in _HIDE))
        skip = tag in ("script", "style")
        self.stack.append((tag, hid, skip))
        if hid:
            self.depth += 1
        if skip:
            self.skip += 1
        if tag == "a":
            self.cur = [a.get("href") or "", []]

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        st = self.stack
        i = len(st) - 1
        while i >= 0 and st[i][0] != tag:
            i -= 1
        if i < 0:
            return                               # stray end tag
        while len(st) > i:                       # also closes unclosed children
            t, hid, skip = st.pop()
            if hid:
                self.depth -= 1
            if skip:
                self.skip -= 1
            if t == "a" and self.cur is not None:
                self.anchors.append((self.cur[0], "".join(self.cur[1])))
                self.cur = None

    def handle_data(self, data):
        if self.skip:
            return
        if self.cur is not None:
            self.cur[1].append(data)
        s = data.strip()
        if s:
            (self.hid if self.depth > 0 else self.vis).append(s)

def parse_html(html: str) -> ParsedHTML:
    if not html:
        return ParsedHTML()
    p = _Collector()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    return ParsedHTML(" ".join(p.vis), " ".join(p.hid), p.anchors)