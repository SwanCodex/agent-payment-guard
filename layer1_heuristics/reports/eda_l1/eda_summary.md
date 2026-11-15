# Layer 1 URL EDA

Built: 2026-11-14

## Class balance (train)

| source      |     0 |     1 |
|:------------|------:|------:|
| commoncrawl | 34439 |     0 |
| phishtank   |     0 | 16370 |
| tranco      | 70066 |     0 |

Malicious share overall: 13.5%

Timestamped malicious rows per year:

|   first_seen |   malicious rows |
|-------------:|-----------------:|
|         2012 |                1 |
|         2013 |                1 |
|         2016 |                2 |
|         2017 |                6 |
|         2018 |                3 |
|         2019 |                5 |
|         2020 |               13 |
|         2021 |               97 |
|         2022 |              294 |
|         2023 |             2190 |
|         2024 |             7728 |
|         2025 |              861 |
|         2026 |             5169 |

Last 12 months:

| first_seen   |   malicious rows |
|:-------------|-----------------:|
| 2025-11      |               21 |
| 2025-12      |               69 |
| 2026-01      |             1708 |
| 2026-02      |             2744 |
| 2026-03      |              260 |
| 2026-04      |               73 |
| 2026-05      |              105 |
| 2026-06      |               83 |
| 2026-07      |               58 |
| 2026-08      |               41 |
| 2026-09      |               19 |
| 2026-10      |               38 |
| 2026-11      |               40 |

## Feature distributions by class (train)

| feature      |   L0 mean |   L0 median |   L0 p90 |   L1 mean |   L1 median |   L1 p90 |
|:-------------|----------:|------------:|---------:|----------:|------------:|---------:|
| url_len      |    38.05  |      25     |   75     |    34.756 |      23     |   54     |
| host_len     |    14.66  |      14     |   21     |    11.299 |       7     |   21     |
| path_depth   |     0.705 |       0     |    2     |     1.27  |       1     |    2     |
| query_len    |     2.398 |       0     |    0     |     3.471 |       0     |    0     |
| digit_ratio  |     0.035 |       0     |    0.133 |     0.041 |       0.015 |    0.095 |
| entropy_host |     3.269 |       3.293 |    3.682 |     2.893 |       2.807 |    3.594 |
| n_subdomains |     0.13  |       0     |    1     |     0.53  |       0     |    1     |
| is_https     |     0.853 |       1     |    1     |     0.977 |       1     |    1     |
| is_homepage  |     0.677 |       1     |    1     |     0.021 |       0     |    0     |

## By source and label

|                    |     n |   homepage_share |   mean_depth |   median_url_len |   https_share |
|:-------------------|------:|-----------------:|-------------:|-----------------:|--------------:|
| ('commoncrawl', 0) | 34439 |            0.021 |        2.139 |               58 |         0.553 |
| ('phishtank', 1)   | 16370 |            0.021 |        1.27  |               23 |         0.977 |
| ('tranco', 0)      | 70066 |            1     |        0     |               22 |         1     |

## Path-depth distribution by source and label (share of rows)

|                    |   depth 0 |   depth 1 |   depth 2 |   depth 3 |   depth 4 |   depth 5+ |
|:-------------------|----------:|----------:|----------:|----------:|----------:|-----------:|
| ('commoncrawl', 0) |     0.044 |     0.279 |     0.384 |     0.167 |     0.079 |      0.047 |
| ('phishtank', 1)   |     0.024 |     0.788 |     0.118 |     0.049 |     0.015 |      0.007 |
| ('tranco', 0)      |     1     |     0     |     0     |     0     |     0     |      0     |

## Domains and shared infrastructure (train)

- Top-10 registered domains hold 85.3% of malicious rows:

| reg_domain   |   rows |
|:-------------|-------:|
| qrco.de      |   4413 |
| q-r.to       |   3454 |
| ead.me       |   3221 |
| adobe.com    |   1086 |
| google.com   |    862 |
| r2.dev       |    452 |
| dropbox.com  |    142 |
| zpr.io       |    116 |
| github.io    |    113 |
| flowcode.com |    106 |

- Registered domains present in BOTH classes: 69
- commoncrawl / phishtank: 41 shared registered domains
- commoncrawl / tranco: 443 shared registered domains
- phishtank / tranco: 34 shared registered domains

## Single-feature classifiers (tree fit on train, AUROC on val), natural data

A high value means the dataset itself may be teaching a shortcut.

| feature              |   auroc |
|:---------------------|--------:|
| path_len             |   0.799 |
| entropy_path         |   0.796 |
| path_depth           |   0.774 |
| hex_ratio_pathquery  |   0.743 |
| is_homepage          |   0.739 |
| n_subdomains         |   0.673 |
| n_dots               |   0.662 |
| host_labels          |   0.642 |
| host_len             |   0.629 |
| n_digits             |   0.572 |
| longest_token        |   0.571 |
| n_hyphens            |   0.568 |
| host_hyphens         |   0.555 |
| url_len              |   0.553 |
| max_label_len        |   0.545 |
| is_https             |   0.524 |
| digit_ratio          |   0.515 |
| entropy_host         |   0.515 |
| query_len            |   0.502 |
| has_punycode         |   0.501 |
| is_ip_host           |   0.5   |
| has_port             |   0.5   |
| double_slash_in_path |   0.5   |
| has_at               |   0.5   |
| is_shortener         |   0.5   |
| host_digits          |   0.498 |
| host_digit_ratio     |   0.494 |
| vowel_ratio_host     |   0.486 |
| n_query_params       |   0.455 |
| n_special            |   0.451 |

## After path-depth matching (benign resampled to the malicious depth histogram)

Train, rows per depth bin:

|   depth_bin |   malicious |   benign |   kept_each |
|------------:|------------:|---------:|------------:|
|           0 |         390 |    71585 |         390 |
|           1 |       12892 |     9607 |        9607 |
|           2 |        1933 |    13211 |        1933 |
|           3 |         805 |     5754 |         805 |
|           4 |         242 |     2715 |         242 |
|           5 |         108 |     1633 |         108 |

Train rows kept: 26170 of 120875. Malicious rows lost: 20.1%, benign rows lost: 87.5%.

Single-feature AUROC before and after matching (val):

| feature              |   natural |   depth-matched |
|:---------------------|----------:|----------------:|
| path_len             |     0.799 |           0.64  |
| entropy_path         |     0.796 |           0.607 |
| path_depth           |     0.774 |           0.51  |
| hex_ratio_pathquery  |     0.743 |           0.588 |
| is_homepage          |     0.739 |           0.51  |
| n_subdomains         |     0.673 |           0.59  |
| n_dots               |     0.662 |           0.493 |
| host_labels          |     0.642 |           0.564 |
| host_len             |     0.629 |           0.563 |
| n_digits             |     0.572 |           0.588 |
| longest_token        |     0.571 |           0.572 |
| n_hyphens            |     0.568 |           0.61  |
| host_hyphens         |     0.555 |           0.535 |
| url_len              |     0.553 |           0.628 |
| max_label_len        |     0.545 |           0.565 |
| is_https             |     0.524 |           0.632 |
| digit_ratio          |     0.515 |           0.495 |
| entropy_host         |     0.515 |           0.552 |
| query_len            |     0.502 |           0.507 |
| has_punycode         |     0.501 |           0.501 |
| is_ip_host           |     0.5   |           0.5   |
| has_port             |     0.5   |           0.5   |
| double_slash_in_path |     0.5   |           0.503 |
| has_at               |     0.5   |           0.509 |
| is_shortener         |     0.5   |           0.5   |
| host_digits          |     0.498 |           0.523 |
| host_digit_ratio     |     0.494 |           0.496 |
| vowel_ratio_host     |     0.486 |           0.495 |
| n_query_params       |     0.455 |           0.509 |
| n_special            |     0.451 |           0.523 |

## Notes

- Only the train split is profiled; val is used for the shortcut check; test is untouched.
- Matching equalizes benign and malicious counts per path-depth bin, so path depth carries no class information in the matched sets.
- Other features (URL length etc.) may still differ between classes for legitimate reasons or because of source artifacts; see the table above.

Figures: path_depth_hist.png, url_len_hist.png
