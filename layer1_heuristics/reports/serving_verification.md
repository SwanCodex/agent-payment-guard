# Serving verification

## Serving path reproduces the evaluation numbers

Tolerance 0.003.

| split | metric | serving | expected | ok |
|---|---|---|---|---|
| val | ROC | 0.9374 | 0.9375 | PASS |
| val | PR | 0.7999 | 0.7999 | PASS |
| val | recall@0.5%FPR | 0.4450 | 0.4461 | PASS |
| test | ROC | 0.8958 | 0.8958 | PASS |
| test | PR | 0.6438 | 0.6438 | PASS |
| test | recall@0.5%FPR | 0.1818 | 0.1818 | PASS |

## Domain-capped metrics (at most 5 rows per registered domain and class)

| split | rows | ROC | PR | recall@0.5%FPR |
|---|---|---|---|---|
| val | 24763 | 0.9198 | 0.7115 | 0.3709 |
| test | 25908 | 0.8957 | 0.6432 | 0.1807 |

## Latency (single URL, one thread)

| path | p50 ms | p95 ms | p99 ms |
|---|---|---|---|
| UrlEnsemble.score | 0.46 | 0.79 | 1.15 |
| run_l1 (rules + model) | 1.40 | 2.24 | 2.68 |

Budget: URL-only model p95 <= 20 ms.

Overall parity check: PASS
