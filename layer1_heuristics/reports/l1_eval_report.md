# Layer 1 URL ensemble: evaluation report

Built 2026-11-14, version 0.1.0.

Base models: weighted LR on char n-grams + weighted LightGBM (no host features), logit average;
calibration: Platt on z, then isotonic; thresholds and conformal qhat fit on val_natural.
Rule flags R06/R07/R10 are NOT inputs to the combiner (see notes); they act through the decision policy.
Selection of C, features and early stopping used the validation splits. Test-split history: see the notes at the end.

LightGBM best iteration 163; combiner coefficients {'z': 1.5883}.

## Discrimination

|                                                |    ROC |     PR |   recall@0.1%FPR |   recall@0.5%FPR |   recall@1%FPR |      ECE |
|:-----------------------------------------------|-------:|-------:|-----------------:|-----------------:|---------------:|---------:|
| ('ensemble z only', 'val_natural')             | 0.9365 | 0.8073 |           0.1821 |           0.4423 |         0.5652 | nan      |
| ('combiner + calibrated', 'val_natural')       | 0.9375 | 0.7999 |           0.1947 |           0.4461 |         0.5671 |   0.0021 |
| ('ensemble z only', 'val_matched')             | 0.9327 | 0.9364 |           0.1327 |           0.3309 |         0.4752 | nan      |
| ('combiner + calibrated', 'val_matched')       | 0.9337 | 0.9336 |           0.1435 |           0.3398 |         0.4893 |   0.1531 |
| ('ensemble z only', 'external_phiusiil')       | 0.6612 | 0.515  |           0.0811 |           0.1303 |         0.1553 | nan      |
| ('combiner + calibrated', 'external_phiusiil') | 0.661  | 0.5031 |           0.0794 |           0.128  |         0.155  |   0.124  |
| ('ensemble z only', 'test')                    | 0.8965 | 0.6579 |           0.0696 |           0.1846 |         0.2706 | nan      |
| ('combiner + calibrated', 'test')              | 0.8958 | 0.6438 |           0.069  |           0.1818 |         0.2715 |   0.0261 |

## Operating point (tau_allow 0.0082, tau_block 0.8333, qhat 0.2861)

|                   |     n |   FPR@block |   recall@block |   FNR@allow |   benign allowed |   benign gray |   abstain rate |   conformal coverage |
|:------------------|------:|------------:|---------------:|------------:|-----------------:|--------------:|---------------:|---------------------:|
| val_natural       | 25749 |      0.0045 |         0.4322 |      0.0049 |           0.2122 |        0.7833 |         0.0471 |               0.9083 |
| val_matched       |  6932 |      0.0072 |         0.4322 |      0.0049 |           0.2819 |        0.7109 |         0.106  |               0.7493 |
| external_phiusiil | 40000 |      0.0102 |         0.156  |      0.0148 |           0.032  |        0.9578 |         0.1564 |               0.6578 |
| test              | 25967 |      0.0041 |         0.164  |      0.0096 |           0.2107 |        0.7852 |         0.0643 |               0.866  |

`benign gray` is the share of benign URLs this layer sends to the gray zone (LLM Arbiter budget).

`abstain` = conformal prediction set empty or two-class (size != 1).

## Allow-side trade-off (how many benign URLs can be confidently allowed)

|   target FNR (val) |   tau_allow |   val benign allowed |   test FNR |   test benign allowed |
|-------------------:|------------:|---------------------:|-----------:|----------------------:|
|               0.01 |      0.0082 |                0.212 |      0.01  |                 0.211 |
|               0.02 |      0.0082 |                0.212 |      0.01  |                 0.211 |
|               0.05 |      0.023  |                0.584 |      0.068 |                 0.575 |
|               0.1  |      0.0515 |                0.787 |      0.153 |                 0.778 |

## Malicious recall at tau_block by first-seen quarter

val_natural:

| q      |    n |   recall_at_block |   mean_p |
|:-------|-----:|------------------:|---------:|
| 2024Q2 |   51 |             0.431 |    0.632 |
| 2024Q3 |  186 |             0.478 |    0.679 |
| 2024Q4 |  120 |             0.475 |    0.695 |
| 2025Q1 |  166 |             0.53  |    0.671 |
| 2025Q2 |  195 |             0.492 |    0.674 |
| 2025Q3 |  348 |             0.517 |    0.686 |
| 2025Q4 |  417 |             0.415 |    0.673 |
| 2026Q1 |  514 |             0.481 |    0.652 |
| 2026Q2 | 1234 |             0.314 |    0.533 |
| 2026Q3 |  143 |             0.741 |    0.854 |
| 2026Q4 |   92 |             0.576 |    0.656 |

test:

| q      |    n |   recall_at_block |   mean_p |
|:-------|-----:|------------------:|---------:|
| 2026Q2 |  402 |             0.174 |    0.425 |
| 2026Q3 | 1395 |             0.157 |    0.434 |
| 2026Q4 | 1740 |             0.167 |    0.456 |

## Test split by TLD (n >= 300)

| tld   |     n |   malicious share |   FPR@block |   recall@block |
|:------|------:|------------------:|------------:|---------------:|
| com   | 10715 |             0.106 |       0.002 |          0.182 |
| org   |  1081 |             0.028 |       0     |          0.167 |
| net   |  1049 |             0.077 |       0.003 |          0.198 |
| ru    |   920 |             0.003 |       0     |          0.667 |
| de    |   773 |             0.048 |       0.01  |          0.405 |
| nl    |   342 |             0.006 |       0     |          0.5   |
| co.uk |   313 |             0.01  |       0     |          0     |
| it    |   300 |             0.023 |       0.007 |          0.286 |

## History and notes

- Rule flag counts on val_natural (benign / malicious): {'R06': (4, 3), 'R07': (0, 0), 'R10': (0, 0)}.
- A first run kept URL-shortener hosts in the data and fed R06/R07/R10 to the combiner. Validation flag counts then showed
  R10 firing on 318 malicious and 0 benign URLs (sampling artifact: benign shorteners were never sampled), and the combiner
  gave it a large weight. That run's test numbers were ROC 0.892, PR 0.652, recall@0.5%FPR 0.179.
- This version removes shortener-host rows from all splits (R10 stays a gray rule), drops is_shortener from LightGBM and
  uses z alone in the combiner. The test split was therefore scored twice; both runs are disclosed here.

## Not done / limitations

- No HTML/page model; R03 disabled; no char-CNN.
- R06 uses a 20-brand seed list, not Tranco top-N.
- Benign data is Tranco homepages plus Common Crawl pages; malicious is PhishTank. External PhiUSIIL is homepage-heavy.
- Thresholds and calibration use the natural val mix (about 15% malicious), not a deployment prior.
