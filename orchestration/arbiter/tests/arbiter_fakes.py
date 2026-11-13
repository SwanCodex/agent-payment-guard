"""Test doubles for the Arbiter (unique module name so it cannot collide with another conftest)."""
import json
import time


class FakeClient:
    """Returns a canned reply (or raises / sleeps). Records the prompts it was given."""

    def __init__(self, reply='{"verdict": "STEP_UP", "reason_codes": ["INSUFFICIENT_EVIDENCE"], "confidence": 0.6}',
                 delay=0.0, exc=None):
        self.reply, self.delay, self.exc, self.calls, self.last = reply, delay, exc, 0, None

    def complete(self, system, user, *, timeout_s):
        self.calls += 1
        self.last = (system, user, timeout_s)
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


def reply(verdict="STEP_UP", codes=("INSTRUCTION_IN_DATA",), conf=0.8):
    return json.dumps({"verdict": verdict, "reason_codes": list(codes), "confidence": conf})
