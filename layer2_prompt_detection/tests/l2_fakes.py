"""Test double for the L2 scorer (unique module name so it cannot collide with other conftest files)."""
import numpy as np


class FakeScorer:
    """z = +4 if the text contains an attack phrase, else -4. Counts calls so tests can check cost."""
    PHRASES = ("ignore all previous", "system note", "forward the customer")

    def __init__(self):
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        def z(t):
            low = t.lower()
            if "possibly suspicious" in low:
                return 0.7            # p is about 0.03 for the shipped thresholds: between tau_allow and tau_block
            return 4.0 if any(p in low for p in self.PHRASES) else -4.0
        return np.array([z(t) for t in texts])
