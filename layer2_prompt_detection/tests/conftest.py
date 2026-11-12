import numpy as np
import pytest

from sentinel_l2 import api
from sentinel_l2.models.bundle import Bundle


class FakeScorer:
    """z = +4 if the text contains an attack phrase, else -4. Counts calls so tests can check cost."""
    PHRASES = ("ignore all previous", "system note", "forward the customer")

    def __init__(self):
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        return np.array([4.0 if any(p in t.lower() for p in self.PHRASES) else -4.0 for t in texts])


@pytest.fixture
def bundle():
    # values of the shipped seed-1 artifact
    return Bundle(dir=None, T=0.624, delta=-4.6, tau_block=0.0500, tau_allow=0.0201)


@pytest.fixture
def fake(bundle):
    api.reset()
    scorer = FakeScorer()
    api.configure(bundle=bundle, scorer=scorer)
    yield scorer
    api.reset()
