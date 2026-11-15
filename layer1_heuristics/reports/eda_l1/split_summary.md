# Layer 1 URL dataset: split summary

Built: 2026-11-14  seed=42

## Rows by source and label

| source      |     0 |     1 |
|:------------|------:|------:|
| commoncrawl | 49219 |     0 |
| phishtank   |     0 | 23373 |
| tranco      | 99999 |     0 |

## Rows by split and label

| split   |      0 |     1 |
|:--------|-------:|------:|
| test    |  22430 |  3537 |
| train   | 104505 | 16370 |
| val     |  22283 |  3466 |

## First-seen range of timestamped rows per split

| split   | min                       | max                       |   count |
|:--------|:--------------------------|:--------------------------|--------:|
| test    | 2026-06-10 13:06:44+00:00 | 2026-11-13 19:05:10+00:00 |    3537 |
| train   | 2012-06-14 15:19:26+00:00 | 2026-11-13 13:24:12+00:00 |   16370 |
| val     | 2024-06-01 22:29:22+00:00 | 2026-11-13 10:27:33+00:00 |    3466 |

## Checks

- registered domains in more than one split: **0** (asserted)
- total rows: 172591, registered domains: 150453

## Known limitations

- Only PhishTank and URLhaus carry per-URL first-seen dates; those domains are split in time order.
- Benign rows (Tranco, Common Crawl, PhiUSIIL) have no first-seen date and are split randomly by domain, so the time split covers malicious rows only.
- OpenPhish is a snapshot (no first-seen dates).
