# R02 N is the measured 99.5th percentile of hidden-text words on 1,200 real
# Common Crawl shop-like pages (CC-MAIN-2026-43). Re-measure on a larger sample.
R02_MIN_HIDDEN_WORDS = 1276    # flag (step_up) at/above this
R02_FORWARD_MIN_WORDS = 15     # forward to L2 (no flag) at/above this

R03_MISSING_FRACTION = 0.5     # PROVISIONAL (unmeasured, needs real PDFs)
R03_MIN_MISSING_TOKENS = 10
R03_ENABLED = False    # unmeasured: needs real invoice PDFs (e.g. DocILE) before enabling
R06_MAX_EDIT = 2
R06_MAX_EDIT_SHORT = 1
R06_SHORT_LABEL_LEN = 8
R06_MIN_BRAND_LEN_EDIT = 6
R09_MIN_DECODED_CHARS = 12
R10_MAX_REDIRECTS = 3
