# ULB credit card fraud: sanity check of the Layer 3A pipeline

Purpose: run the same split / training / calibration / metric steps on a dataset WITHOUT entities, to show that none of
that code is IEEE-CIS-specific in a misleading way. This dataset is never merged with IEEE-CIS, and the numbers below are
not comparable with the IEEE-CIS results (different data, no history features, anonymized PCA columns).

Data: ULB / Worldline credit card fraud (Sept 2013), 284,807 transactions, 492 fraud, columns Time, V1-V28, Amount, Class.

## Split (time order, same rule as IEEE-CIS)
| split | rows | fraud | fraud_rate | first_time | last_time |
|---|---|---|---|---|---|
| train | 199364 | 384 | 0.0019 | 0.0000 | 132928.0000 |
| val1 | 28481 | 33 | 0.0012 | 132929.0000 | 145247.0000 |
| val2 | 28481 | 53 | 0.0019 | 145248.0000 | 157640.0000 |
| test | 28481 | 22 | 0.0008 | 157641.0000 | 172792.0000 |

## Results
| split | model | PR-AUC | ROC-AUC | recall@0.1%FPR | recall@1%FPR | ECE | mean p | fraud rows |
|---|---|---|---|---|---|---|---|---|
| val2 | logistic regression | 0.8311 | 0.9822 | 0.7925 | 0.9057 | 0.0003 | 0.0020 | 53 |
| val2 | LightGBM 5 seeds (raw) | 0.8327 | 0.9848 | 0.8113 | 0.8868 | 0.0013 | 0.0031 | 53 |
| val2 | LightGBM 5 seeds (temperature calibrated) | 0.8327 | 0.9848 | 0.8113 | 0.8868 | 0.0003 | 0.0018 | 53 |
| test | logistic regression | 0.4531 | 0.9592 | 0.5909 | 0.7727 | 0.0007 | 0.0014 | 22 |
| test | LightGBM 5 seeds (raw) | 0.7014 | 0.9837 | 0.7273 | 0.7273 | 0.0017 | 0.0025 | 22 |
| test | LightGBM 5 seeds (temperature calibrated) | 0.7014 | 0.9837 | 0.7273 | 0.7273 | 0.0006 | 0.0011 | 22 |

Calibration: temperature scaling fitted on val1, T = 0.7575.

## Leakage check
Training on shuffled labels gives PR-AUC 0.00100 on val2, against a fraud rate of 0.00186.
This is close to the fraud rate, so the pipeline does not leak labels.

## Limits
- Only 492 fraud cases in total, so each of val1 / val2 / test holds few fraud rows (see the table): every metric here is noisy.
- The data covers two days, so there is no long-term drift to test.
- No history features: this checks the generic parts of the pipeline only.
- 4 of the 5 LightGBM seeds stopped after 1 tree, because val1 holds only 33 fraud rows; the averaged model is dominated by seed 55.
