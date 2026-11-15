# Step 6 results (first models, untuned)

Training set: depth-matched train. Early stopping and the LR constant were chosen on val_matched.
Test split not used.

|                                          |     n |   mal_share |   PR-AUC |   ROC-AUC |   recall@FPR0.1% |   recall@FPR0.5% |   recall@FPR1% |
|:-----------------------------------------|------:|------------:|---------:|----------:|-----------------:|-----------------:|---------------:|
| ('LR char n-grams', 'val_matched')       |  7548 |      0.5    |   0.9049 |    0.9153 |           0.0156 |           0.1351 |         0.309  |
| ('LR char n-grams', 'val_natural')       | 26081 |      0.1457 |   0.4337 |    0.803  |           0.0018 |           0.0232 |         0.0866 |
| ('LR char n-grams', 'external_phiusiil') | 40000 |      0.2749 |   0.404  |    0.6408 |           0.0029 |           0.0233 |         0.0387 |
| ('LightGBM', 'val_matched')              |  7548 |      0.5    |   0.7779 |    0.7657 |           0.1361 |           0.1974 |         0.2353 |
| ('LightGBM', 'val_natural')              | 26081 |      0.1457 |   0.3696 |    0.5413 |           0.0584 |           0.1436 |         0.2266 |
| ('LightGBM', 'external_phiusiil')        | 40000 |      0.2749 |   0.2083 |    0.2875 |           0.0121 |           0.0147 |         0.0173 |

## LightGBM feature importance (share of total gain)

|                       |   gain share |
|:----------------------|-------------:|
| host_len              |       0.4607 |
| n_subdomains          |       0.1443 |
| url_len               |       0.1132 |
| tld_risk              |       0.0633 |
| path_depth            |       0.0604 |
| path_len              |       0.0379 |
| max_label_len         |       0.0225 |
| entropy_host          |       0.0173 |
| entropy_path          |       0.0167 |
| longest_token         |       0.0148 |
| vowel_ratio_host      |       0.0139 |
| host_labels           |       0.0103 |
| host_digits           |       0.0086 |
| n_hyphens             |       0.0048 |
| host_digit_ratio      |       0.0026 |
| brand_in_domain_label |       0.0019 |
| n_digits              |       0.0017 |
| n_dots                |       0.0015 |
| hex_ratio_pathquery   |       0.0014 |
| digit_ratio           |       0.0007 |
