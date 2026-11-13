import pytest

from l2_fakes import FakeScorer
from sentinel_l2 import scoring
from sentinel_l2.models.bundle import Bundle


@pytest.fixture
def bundle():
    # values of the shipped seed-1 artifact
    return Bundle(dir=None, T=0.624, delta=-4.6, tau_block=0.0500, tau_allow=0.0201)


@pytest.fixture
def fake(bundle):
    scoring.reset()
    scorer = FakeScorer()
    scoring.configure(bundle=bundle, scorer=scorer)
    yield scorer
    scoring.reset()
