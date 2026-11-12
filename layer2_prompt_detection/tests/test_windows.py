from sentinel_l2.models.scorer import window_starts


def test_short_text_single_window():
    assert window_starts(10, 254, 128) == [0]
    assert window_starts(254, 254, 128) == [0]
    assert window_starts(0, 254, 128) == [0]


def test_last_window_reaches_the_end():
    assert window_starts(255, 254, 128) == [0, 1]
    assert window_starts(400, 254, 128) == [0, 128, 146]
    assert window_starts(894, 254, 128)[-1] + 254 == 894


def test_every_token_is_covered():
    for n in (1, 100, 254, 300, 700, 894):
        covered = set()
        for s in window_starts(n, 254, 128):
            covered.update(range(s, min(s + 254, n)))
        assert covered == set(range(n))
