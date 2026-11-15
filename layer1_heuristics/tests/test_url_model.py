import pytest
from sentinel_contracts import L1In, Verdict
from sentinel_l1.api import run_l1
from sentinel_l1.models.url_ensemble import UrlEnsemble, default_dir

pytestmark = pytest.mark.skipif(not default_dir().exists(), reason="url_ensemble artifacts not built")


@pytest.fixture(scope="module")
def model():
    return UrlEnsemble(default_dir())


def test_score_is_probability_and_deterministic(model):
    u = "http://secure-login.example-account.top/verify/update.php?id=12345"
    a, b = model.score(u), model.score(u)
    assert 0.0 <= a["p"] <= 1.0 and a == b


def test_empty_and_odd_urls_do_not_crash(model):
    for u in ["", "not a url", "http://", "https://" + "a" * 500 + ".com/" + "b" * 600]:
        assert 0.0 <= model.score(u)["p"] <= 1.0


def test_run_l1_reports_model_metadata():
    out = run_l1(L1In(request_id="t", url="https://www.wikipedia.org/"))
    m = out.metadata["url_model"]
    assert isinstance(m, dict) and 0.0 <= m["p"] <= 1.0
    assert out.verdict in (Verdict.ALLOW, Verdict.STEP_UP, Verdict.BLOCK)


def test_shortener_flag_still_reported():
    out = run_l1(L1In(request_id="t", url="https://bit.ly/abc123"))
    assert "L1:R10_URL_SHORTENER" in out.reason_codes