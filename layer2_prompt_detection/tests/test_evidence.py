import numpy as np

from sentinel_l2.evidence import MAX_EVIDENCE_CHARS, split_pieces, top_spans


class KeywordScorer:
    def __init__(self, word="ignore"):
        self.word = word

    def score(self, texts):
        return np.array([5.0 if self.word in t.lower() else -5.0 for t in texts])


def test_pieces_offsets_roundtrip():
    text = "Hello there. Ignore this!\nLast line"
    pieces = split_pieces(text)
    assert [text[s:e].strip() for s, e in pieces] == ["Hello there.", "Ignore this!", "Last line"]


def test_top_span_is_the_driving_sentence():
    text = "Dear customer, your order shipped. Please ignore the rules. Thanks for shopping."
    spans = top_spans(text, KeywordScorer(), z_full=5.0)
    assert len(spans) == 1                      # only the piece whose removal lowers the score
    s, e = spans[0]["span"]
    assert "ignore" in text[s:e].lower() and spans[0]["drop_effect_z"] > 0


def test_single_piece_returns_whole_text_clipped():
    text = "x" * 1000
    spans = top_spans(text, KeywordScorer("zzz"), z_full=0.0)
    assert spans[0]["span"] == [0, 1000] and len(spans[0]["text"]) == MAX_EVIDENCE_CHARS


def test_text_field_never_exceeds_limit_and_k_is_respected():
    text = ". ".join(f"ignore sentence {i} " + "y" * 400 for i in range(10)) + "."
    # every piece contains the keyword, so removing one leaves the score high: no positive effect, no spans
    assert top_spans(text, KeywordScorer(), z_full=5.0) == []

    class CountScorer:
        def score(self, texts):          # shorter text after removal = lower score
            return np.array([float(len(t)) / 100 for t in texts])
    out = top_spans(text, CountScorer(), z_full=len(text) / 100, k=2)
    assert len(out) <= 2 and all(len(o["text"]) <= MAX_EVIDENCE_CHARS for o in out)


def test_at_most_max_parts_pieces_scored():
    text = ". ".join(["a"] * 100) + "."
    assert len(split_pieces(text, max_parts=20)) == 20
