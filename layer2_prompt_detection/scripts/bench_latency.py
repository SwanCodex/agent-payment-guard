"""Measure run_l2 latency (PyTorch baseline) on short, medium and long inputs.

    python layer2_prompt_detection/scripts/bench_latency.py            # real model from the artifact bundle
    python layer2_prompt_detection/scripts/bench_latency.py --fake     # smoke test of this script, no model

Handbook budget: <= 150 ms p95 per 512-token window for the int8 ONNX model, and <= 300 ms end to end for the
whole guard. This script gives the PyTorch number to compare against. Run it on the machine that will serve.
"""
import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))      # repo root, for the package-style import

from sentinel_contracts import L2In  # noqa: E402

SHORT = "Please pay invoice #4821 by Friday. Thank you for your business."
MEDIUM = " ".join(["Dear customer, your order has shipped and should arrive within five business days."] * 12)
LONG = " ".join(["Dear customer, your order has shipped and should arrive within five business days."] * 60)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="calls per input size")
    ap.add_argument("--fake", action="store_true", help="use a dummy scorer (tests the script only)")
    args = ap.parse_args()

    from layer2_prompt_detection.sentinel_l2 import scoring
    from layer2_prompt_detection.sentinel_l2.api import run_l2, warmup
    from layer2_prompt_detection.sentinel_l2.models.bundle import Bundle

    if args.fake:
        import numpy as np

        class Dummy:
            def score(self, texts):
                return np.full(len(texts), -4.0)
        scoring.configure(bundle=Bundle(dir=None, T=0.624, delta=-4.6, tau_block=0.05, tau_allow=0.0201), scorer=Dummy())

    print(f"warm-up (model load + first inference): {warmup():.2f} s")
    print(f"{'input':8s} {'chars':>6s} {'p50 ms':>8s} {'p95 ms':>8s} {'max ms':>8s}")
    for name, text in (("short", SHORT), ("medium", MEDIUM), ("long", LONG)):
        ms = []
        for i in range(args.n):
            t0 = time.perf_counter()
            run_l2(L2In(request_id=f"b{i}", prompt=text))
            ms.append((time.perf_counter() - t0) * 1000)
        print(f"{name:8s} {len(text):6d} {statistics.median(ms):8.1f} {pct(ms, 0.95):8.1f} {max(ms):8.1f}")


if __name__ == "__main__":
    main()
