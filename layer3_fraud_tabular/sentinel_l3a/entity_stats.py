"""Entity history store for Layer 3A (a plain dict now; the same methods can sit on top of Redis later).

For every key (a card, an address, a device) it remembers the earlier transactions.
All look-ups are STRICTLY before the current time t: a transaction never sees itself
or another transaction with the same timestamp. That is the same rule as the training features.
"""
import bisect

import numpy as np

HOUR = 3600.0
DAY = 86400.0
LONG_WINDOW = 400 * DAY      # "all history" window used in training


class EntityStore:
    def __init__(self):
        self.times = {}      # key -> list of transaction times (kept in increasing order)
        self.amounts = {}    # key -> list of amounts, same order
        self.distinct = {}   # (kind, key) -> {value: time it was first seen}

    # ---------- write ----------
    def add_transaction(self, key, t, amount):
        self.times.setdefault(key, []).append(t)
        self.amounts.setdefault(key, []).append(amount)

    def add_value(self, kind, key, value, t):
        self.distinct.setdefault((kind, key), {}).setdefault(value, t)   # keep the first time only

    # ---------- read (strictly before t) ----------
    def stats(self, key, t):
        times = self.times.get(key, [])
        k = bisect.bisect_left(times, t)                       # number of transactions with time < t

        def count_since(seconds):
            return k - bisect.bisect_left(times, t - seconds)  # window is [t - seconds, t)

        lo = bisect.bisect_left(times, t - LONG_WINDOW)
        amounts = np.asarray(self.amounts.get(key, [])[lo:k], dtype="float64")
        n = len(amounts)
        return {
            "cnt_1h": count_since(HOUR),
            "cnt_24h": count_since(DAY),
            "cnt_7d": count_since(7 * DAY),
            "cnt_all": n,
            "amt_mean": float(amounts.mean()) if n >= 1 else np.nan,
            "amt_std": float(amounts.std(ddof=1)) if n >= 2 else np.nan,
            "secs_since_last": float(t - times[k - 1]) if n >= 1 else np.nan,
        }

    def distinct_before(self, kind, key, t):
        seen = self.distinct.get((kind, key), {})
        return sum(1 for first_time in seen.values() if first_time < t)
