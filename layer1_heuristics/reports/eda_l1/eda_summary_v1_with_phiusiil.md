# Layer 1 URL EDA

Built: 2026-11-14

## Class balance (train)

| source      |     0 |     1 |
|:------------|------:|------:|
| commoncrawl |   832 |     0 |
| phishtank   |     0 | 18391 |
| phiusiil    | 92787 | 35536 |
| tranco      | 66934 |     0 |

Malicious share overall: 25.1%

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
|         2021 |               99 |
|         2022 |              294 |
|         2023 |             2258 |
|         2024 |             9018 |
|         2025 |             1369 |
|         2026 |             5322 |

Last 12 months:

| first_seen   |   malicious rows |
|:-------------|-----------------:|
| 2025-11      |               36 |
| 2025-12      |               90 |
| 2026-01      |             1757 |
| 2026-02      |             2775 |
| 2026-03      |              285 |
| 2026-04      |               78 |
| 2026-05      |              117 |
| 2026-06      |              100 |
| 2026-07      |               62 |
| 2026-08      |               47 |
| 2026-09      |               20 |
| 2026-10      |               41 |
| 2026-11      |               40 |

## Feature distributions by class (train)

| feature      |   L0 mean |   L0 median |   L0 p90 |   L1 mean |   L1 median |   L1 p90 |
|:-------------|----------:|------------:|---------:|----------:|------------:|---------:|
| url_len      |    26.166 |      26     |    33    |    37.133 |      26     |   61     |
| host_len     |    16.939 |      16     |    24    |    16.828 |      16     |   27     |
| path_depth   |     0.011 |       0     |     0    |     0.869 |       1     |    2     |
| query_len    |     0.024 |       0     |     0    |     3.386 |       0     |    0     |
| digit_ratio  |     0.007 |       0     |     0    |     0.041 |       0     |    0.13  |
| entropy_host |     3.32  |       3.346 |     3.72 |     3.224 |       3.281 |    3.827 |
| n_subdomains |     0.585 |       1     |     1    |     0.748 |       1     |    1     |
| is_https     |     0.998 |       1     |     1    |     0.525 |       1     |    1     |
| is_homepage  |     0.995 |       1     |     1    |     0.459 |       0     |    1     |

## By source and label

|                    |     n |   homepage_share |   mean_depth |   median_url_len |   https_share |
|:-------------------|------:|-----------------:|-------------:|-----------------:|--------------:|
| ('commoncrawl', 0) |   832 |            0.008 |        2.16  |               60 |         0.584 |
| ('phishtank', 1)   | 18391 |            0.017 |        1.237 |               22 |         0.98  |
| ('phiusiil', 0)    | 92787 |            1     |        0     |               28 |         1     |
| ('phiusiil', 1)    | 35536 |            0.688 |        0.679 |               29 |         0.29  |
| ('tranco', 0)      | 66934 |            1     |        0     |               22 |         1     |

## Path-depth distribution by source and label (share of rows)

|                    |   depth 0 |   depth 1 |   depth 2 |   depth 3 |   depth 4 |   depth 5+ |
|:-------------------|----------:|----------:|----------:|----------:|----------:|-----------:|
| ('commoncrawl', 0) |     0.025 |     0.27  |     0.444 |     0.142 |     0.072 |      0.047 |
| ('phishtank', 1)   |     0.019 |     0.815 |     0.104 |     0.043 |     0.013 |      0.005 |
| ('phiusiil', 0)    |     1     |     0     |     0     |     0     |     0     |      0     |
| ('phiusiil', 1)    |     0.694 |     0.14  |     0.069 |     0.051 |     0.019 |      0.028 |
| ('tranco', 0)      |     1     |     0     |     0     |     0     |     0     |      0     |

## Domains and shared infrastructure (train)

- Top-10 registered domains hold 30.2% of malicious rows:

| reg_domain   |   rows |
|:-------------|-------:|
| qrco.de      |   4425 |
| q-r.to       |   3455 |
| ead.me       |   3221 |
| bit.ly       |   1706 |
| adobe.com    |   1089 |
| google.com   |    919 |
| r2.dev       |    529 |
| tinyurl.com  |    422 |
| t.co         |    274 |
| github.io    |    250 |

- Registered domains present in BOTH classes: 232
- commoncrawl / phishtank: 3 shared registered domains
- commoncrawl / phiusiil: 19 shared registered domains
- commoncrawl / tranco: 6 shared registered domains
- phishtank / phiusiil: 179 shared registered domains
- phishtank / tranco: 32 shared registered domains
- phiusiil / tranco: 117 shared registered domains

## Single-feature classifiers (tree fit on train, AUROC on val), natural data

A high value means the dataset itself may be teaching a shortcut.

| feature              |   auroc |
|:---------------------|--------:|
| is_homepage          |   0.753 |
| is_https             |   0.753 |
| path_len             |   0.749 |
| entropy_path         |   0.748 |
| path_depth           |   0.748 |
| hex_ratio_pathquery  |   0.738 |
| url_len              |   0.691 |
| n_digits             |   0.673 |
| digit_ratio          |   0.668 |
| n_dots               |   0.633 |
| n_hyphens            |   0.626 |
| entropy_host         |   0.611 |
| longest_token        |   0.589 |
| n_subdomains         |   0.589 |
| host_len             |   0.589 |
| host_hyphens         |   0.585 |
| host_digits          |   0.576 |
| host_digit_ratio     |   0.574 |
| host_labels          |   0.565 |
| n_special            |   0.558 |
| query_len            |   0.543 |
| n_query_params       |   0.543 |
| max_label_len        |   0.538 |
| vowel_ratio_host     |   0.516 |
| has_at               |   0.506 |
| is_shortener         |   0.501 |
| has_punycode         |   0.5   |
| has_port             |   0.5   |
| is_ip_host           |   0.5   |
| double_slash_in_path |   0.5   |

## After path-depth matching (benign resampled to the malicious depth histogram)

Train, rows per depth bin:

|   depth_bin |   malicious |   benign |   kept_each |
|------------:|------------:|---------:|------------:|
|           0 |       25030 |   159742 |       25030 |
|           1 |       19960 |      225 |         225 |
|           2 |        4355 |      369 |         369 |
|           3 |        2593 |      118 |         118 |
|           4 |         902 |       60 |          60 |
|           5 |        1087 |       39 |          39 |

Train rows kept: 51682 of 214480. Malicious rows lost: 52.1%, benign rows lost: 83.9%.

Single-feature AUROC before and after matching (val):

| feature              |   natural |   depth-matched |
|:---------------------|----------:|----------------:|
| is_homepage          |     0.753 |           0.511 |
| is_https             |     0.753 |           0.9   |
| path_len             |     0.749 |           0.511 |
| entropy_path         |     0.748 |           0.506 |
| path_depth           |     0.748 |           0.5   |
| hex_ratio_pathquery  |     0.738 |           0.502 |
| url_len              |     0.691 |           0.579 |
| n_digits             |     0.673 |           0.559 |
| digit_ratio          |     0.668 |           0.558 |
| n_dots               |     0.633 |           0.629 |
| n_hyphens            |     0.626 |           0.552 |
| entropy_host         |     0.611 |           0.59  |
| longest_token        |     0.589 |           0.591 |
| n_subdomains         |     0.589 |           0.655 |
| host_len             |     0.589 |           0.613 |
| host_hyphens         |     0.585 |           0.553 |
| host_digits          |     0.576 |           0.555 |
| host_digit_ratio     |     0.574 |           0.553 |
| host_labels          |     0.565 |           0.627 |
| n_special            |     0.558 |           0.511 |
| query_len            |     0.543 |           0.512 |
| n_query_params       |     0.543 |           0.512 |
| max_label_len        |     0.538 |           0.536 |
| vowel_ratio_host     |     0.516 |           0.587 |
| has_at               |     0.506 |           0.5   |
| is_shortener         |     0.501 |           0.5   |
| has_punycode         |     0.5   |           0.5   |
| has_port             |     0.5   |           0.5   |
| is_ip_host           |     0.5   |           0.501 |
| double_slash_in_path |     0.5   |           0.5   |

## Notes

- Only the train split is profiled; val is used for the shortcut check; test is untouched.
- Matching equalizes benign and malicious counts per path-depth bin, so path depth carries no class information in the matched sets.
- Other features (URL length etc.) may still differ between classes for legitimate reasons or because of source artifacts; see the table above.

Figures: path_depth_hist.png, url_len_hist.png
