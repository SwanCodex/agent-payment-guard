# Layer 3A (tabular fraud): evaluation report

All numbers come from the held-out real IEEE-CIS splits. The test split was evaluated once, after the model,
calibration and policy were frozen on val1 / val2.

## Shipping model on the TEST split
| metric | value |
|---|---|
| rows | 59054.0000 |
| fraud rate | 0.0375 |
| PR-AUC | 0.2942 |
| ROC-AUC | 0.8356 |
| recall@0.1%FPR | 0.0809 |
| recall@0.5%FPR | 0.1676 |
| recall@1%FPR | 0.2309 |
| precision@top 0.5% | 0.7196 |
| precision@top 1% | 0.5888 |
| precision@top 2% | 0.4560 |
| ECE (15 equal-mass bins) | 0.0029 |
| mean predicted | 0.0350 |

## Slices on the test split
| slice | value | rows | fraud rate | PR-AUC | ROC-AUC | mean predicted |
|---|---|---|---|---|---|---|
| week | week 0 | 21045 | 0.0342 | 0.2620 | 0.8371 | 0.0360 |
| week | week 1 | 19955 | 0.0366 | 0.3038 | 0.8293 | 0.0350 |
| week | week 2 | 17790 | 0.0423 | 0.3212 | 0.8408 | 0.0338 |
| week | week 3 | 264 | 0.0417 |  |  | 0.0401 |
| ProductCD | C | 6481 | 0.1467 | 0.4513 | 0.7782 | 0.1072 |
| ProductCD | H | 1772 | 0.0632 | 0.3102 | 0.7939 | 0.0670 |
| ProductCD | R | 2938 | 0.0466 | 0.4183 | 0.8582 | 0.0503 |
| ProductCD | S | 2437 | 0.0464 | 0.0986 | 0.5959 | 0.0608 |
| ProductCD | W | 45426 | 0.0198 | 0.1482 | 0.7866 | 0.0211 |
| identity | identity absent | 45645 | 0.0201 | 0.1493 | 0.7876 | 0.0213 |
| identity | identity present | 13409 | 0.0965 | 0.3924 | 0.8003 | 0.0818 |
| card_history | cold card key | 1737 | 0.0426 | 0.3619 | 0.9076 | 0.0368 |
| card_history | warm card key | 57317 | 0.0373 | 0.2932 | 0.8332 | 0.0350 |

## Cost curve (a missed fraud costs r, a false alarm costs 1; calibrated probabilities)
| cost of missed fraud (r) | flag if p >= | flagged % | cost per txn | cost if flag nothing | cost if flag everything |
|---|---|---|---|---|---|
| 1 | 0.5000 | 0.5487 | 0.0354 | 0.0375 | 0.9625 |
| 2 | 0.3333 | 1.3191 | 0.0671 | 0.0749 | 0.9625 |
| 5 | 0.1667 | 3.6340 | 0.1486 | 0.1874 | 0.9625 |
| 10 | 0.0909 | 7.8200 | 0.2613 | 0.3747 | 0.9625 |
| 20 | 0.0476 | 16.4019 | 0.4206 | 0.7495 | 0.9625 |
| 50 | 0.0196 | 39.8313 | 0.6476 | 1.8737 | 0.9625 |

## ALLOW / REVIEW / BLOCK on the test split
Costs: review c = 0.05, wrong block b = 0.3 (in units of one missed fraud).
ALLOW below p = 0.05, BLOCK from p = 0.833.

| ALLOW % | REVIEW % | BLOCK % | fraud allowed % | fraud in review % | fraud blocked % | good blocked % | cost per transaction |
|---|---|---|---|---|---|---|---|
| 84.5057 | 15.4249 | 0.0694 | 38.6805 | 59.7379 | 1.5816 | 0.0106 | 0.0222 |

Cost per transaction if everything is allowed: 0.0375.

## Development model versus shipping model (PR-AUC)
The shipping model drops 5 features that the live system cannot provide (browser, os and their frequency encodings,
and distinct addresses per card), and is retrained without them.

| model | PR-AUC val1 | PR-AUC val2 |
|---|---|---|
| development model, 48 features (5 not available live) | 0.4110 | 0.2892 |
| SHIPPING model, 43 live features | 0.3919 | 0.2770 |

## Development-stage tables (val1 / val2, earlier 48-feature model)
### Model comparison (val1)
| model | PR-AUC | ROC-AUC | recall@1%FPR |
|---|---|---|---|
| Logistic regression (baseline) | 0.1757 | 0.7802 | 0.1349 |
| LightGBM | 0.4013 | 0.8689 | 0.3377 |
| XGBoost | 0.4038 | 0.8668 | 0.3328 |
| CatBoost | 0.3618 | 0.8407 | 0.2979 |
| LightGBM + CatBoost (average) | 0.3971 | 0.8651 | 0.3304 |
| LightGBM + XGBoost (average) | 0.4126 | 0.8722 | 0.3421 |
| All three (average) | 0.4092 | 0.8712 | 0.3489 |

### Five-seed comparison
| model | PR-AUC val1 | PR-AUC val2 | recall@1%FPR val1 | recall@1%FPR val2 |
|---|---|---|---|---|
| LightGBM tuned, 5 seeds | 0.4110 | 0.2892 | 0.3396 | 0.2577 |
| XGBoost, 5 seeds | 0.4112 | 0.2793 | 0.3406 | 0.2582 |
| LightGBM + XGBoost average | 0.4144 | 0.2874 | 0.3459 | 0.2604 |

### Ablation: history features
| set | PR-AUC | ROC-AUC | recall@1%FPR | n_features | trees |
|---|---|---|---|---|---|
| A: all Core features | 0.4013 | 0.8689 | 0.3377 | 48 | 938 |
| B: without long-history features | 0.3825 | 0.8589 | 0.3076 | 38 | 706 |
| C: without any history features | 0.3054 | 0.8495 | 0.2334 | 26 | 538 |

### Core versus Full feature set
| feature set | PR-AUC val1 | PR-AUC val2 | recall@1%FPR val1 | recall@1%FPR val2 | gray share val2 |
|---|---|---|---|---|---|
| Core (deployable) | 0.4110 | 0.2892 | 0.3396 | 0.2577 | 0.8575 |
| Full (offline ceiling) | 0.6663 | 0.5228 | 0.5900 | 0.4554 | 0.7243 |

### Missing-feature check (what the live system loses)
| version | PR-AUC val2 | recall@1%FPR val2 | PR-AUC change |
|---|---|---|---|
| as trained (all 48 features) | 0.2892 | 0.2577 | 0.0000 |
| browser and os not available | 0.2737 | 0.2388 | -0.0155 |
| card_distinct_addr not available | 0.2681 | 0.2253 | -0.0211 |
| has_identity taken from device type | 0.2892 | 0.2577 | -0.0000 |
| all three changes together | 0.2532 | 0.2112 | -0.0360 |

## Serving parity
The serving code (`sentinel_l3a/api.py`, `entity_stats.py`) replays the test period one transaction at a time.
The comparison table is in `reports/serving/serving_parity_report.csv`.
