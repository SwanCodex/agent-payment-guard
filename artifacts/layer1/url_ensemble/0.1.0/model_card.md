# Model card: Layer 1 URL ensemble (url_ensemble 0.1.0)

## Intended use
A risk signal on payee and page URLs inside the SENTINEL agent payment guard. It scores how likely a URL is to be phishing, using only the URL string. It is one input to the guard's decision policy, next to the rule engine (R01-R10) and the other layers. It is not a standalone blocker and not a judgement about any person. Do not use it on URLs that are shortener links, because their risk lies in the destination.

## Data
Sources (public feeds, no personal data):
- Malicious: PhishTank verified online-valid CSV, downloaded 2026-11-14 (45,711 rows loaded, 23,373 after cleaning). First-seen date = PhishTank submission time.
- Benign: Tranco top-1M, 100,000 random domains as homepages (99,999 after cleaning); Common Crawl CC-MAIN-2026-43, HTTP 200 HTML pages, URLs only, at most 3 per host (52,265 loaded, 49,219 after cleaning).
- Not used for training: PhiUSIIL (kept as an external test, see Metrics); OpenPhish and URLhaus (not obtained).

Cleaning: URL normalisation (lowercase host, fragment removed, query kept); exact duplicates removed; URLs with conflicting labels dropped; template duplicates removed (same registered domain, same path shape, same label); URL-shortener hosts removed (12 domains, 2,249 rows).

Splits: by registered domain, 70/15/15, asserted disjoint. Malicious domains are ordered by their earliest PhishTank submission time (train starts 2012, test covers 2026-06-10 to 2026-11-13). Benign domains have no first-seen date and are split randomly, so the time ordering covers malicious rows only. Rows: train 120,875 (104,505 benign / 16,370 malicious), val 25,749 (22,283 / 3,466), test 25,967 (22,430 / 3,537).

## Model
- LR: logistic regression (C=100) on hashed character 2-5-grams of the URL (2^20 buckets, sublinear TF-IDF), scheme and leading www. removed.
- LightGBM: 163 trees on 25 features (19 lexical, 5 brand/risky-word, 1 TLD risk). Host-shape features, is_shortener and is_https are excluded on purpose (see feature_spec.yaml and the EDA).
- Both trained on all training rows with weights that give each class equal total weight within each path-depth bin (cap 20), so path depth carries no class prior without discarding benign homepages.
- Ensemble: average of the two models in logit space, then Platt scaling and isotonic calibration fit on the natural validation split. Rule flags R06/R07/R10 are not model inputs: they fire too rarely to learn weights and act through the decision policy.
- Thresholds (validation): tau_block = 0.8333 (benign FPR <= 0.5%), tau_allow = 0.0082 (malicious FNR <= 1%), conformal qhat = 0.2861 (alpha 0.1).
- Decision Policy Integration: Probabilistic URL model risk (p >= tau_block) triggers STEP_UP with reason code 'L1:URL_MODEL_HIGH_RISK'. The URL model cannot independently issue hard BLOCK verdicts; deterministic BLOCK is strictly reserved for deterministic rules with severity='block'.
- Model-Unavailable Behavior: If the URL model is missing, fails to load, or throws during inference on a provided URL, L1 sets reason code 'SYSTEM:MODEL_UNAVAILABLE' and issues STEP_UP to prevent silent approval.

## Metrics
All numbers are from held-out real splits.
- Test (domain-disjoint, newest malicious domains): ROC-AUC 0.8958, PR-AUC 0.6438 (malicious share 13.6%), recall at FPR 0.1% / 0.5% / 1% = 6.9% / 18.2% / 27.2%, ECE 0.026.
- Validation (natural mix): ROC-AUC 0.9375, PR-AUC 0.7999, recall at 0.5% FPR 44.6%. With at most 5 rows per domain and class: ROC-AUC 0.9198, recall 37.1%; the same cap leaves test unchanged (0.8957, 18.1%).
- External PhiUSIIL (40,000-row sample, never trained on, mostly homepages): ROC-AUC 0.661, PR-AUC 0.503.
- Operating point on test: FPR at tau_block 0.41%, recall at tau_block 16.4%, FNR at tau_allow 0.96%, benign confidently allowed 21.1%, benign in the gray zone 78.5%, conformal abstain rate 6.4%, conformal coverage 86.6% (validation 90.8%).
- Serving check: the serving code reproduces these metrics within 0.003; single-URL latency p95 0.83 ms (model), 2.2 ms (run_l1 with rules).

## Limitations
- Modest signal. At 0.4% benign FPR it blocks about 16% of the newest phishing URLs. It must not be the only line of defence.
- Validation is much better than test (recall at 0.5% FPR 45% vs 18%). Heavy-hitter domains explain part of the gap; the rest is unexplained (possible causes: drift between older and newer domains, selection on validation).
- One malicious source (PhishTank) and a benign class made of popular-site homepages and generic crawled pages. The model may learn source differences. After depth balancing, path features (path_len, entropy_path) still separate the classes.
- Poor transfer to a differently sourced, homepage-heavy phishing set (external ROC-AUC 0.66).
- About 78% of benign URLs fall in the gray zone at the handbook's allow rule (FNR <= 1%). Looser allow targets: FNR 5% allows 58% of benign (test FNR 6.8%), FNR 10% allows 78% (test FNR 15.3%). The allow side is advisory; the decision-policy owner should not require this layer to be confident-allow.
- Calibration and thresholds use the natural validation mix (about 15% malicious), not a deployment prior.
- Shortener hosts were removed from the data. Their URLs cannot be judged from text; rule R10 flags them (gray).
- Known Anomaly on High-Profile Brand Domains (The Google URL Anomaly — Hypotheses Supported by Evidence):
  - In the training set (`train.csv`), URLs whose host contains "google" count 891 malicious vs 77 benign (92.0% malicious).
  - The registered domain `google.com` in `train.csv` counts 862 malicious rows vs 5 benign rows (99.4% malicious).
  - In `LRScorer`, the host part `'google.com'` contributes +9.5361 to the log-odds decision for `https://www.google.com/`, producing p_lr = 0.8232, LightGBM p_gbm = 0.4291, combiner s = 1.5883 * z + 3.9714, and isotonic calibrated p = 0.9844 >= tau_block (0.8333).
  - Similar high scores occur for other major brand sign-in endpoints: accounts.google.com/signin (p = 1.0000), github.com/login (p = 1.0000), netflix.com/login (p = 1.0000), dropbox.com/login (p = 1.0000), and paypal.com/signin (p = 0.9397).
  - The anomaly is MITIGATED by policy (STEP_UP instead of BLOCK, with score mapped to 0.60 and reason code 'L1:URL_MODEL_HIGH_RISK') and NOT fixed in model 0.1.0; payments to major brands will still STEP_UP.
  - Recommended fix for v0.2.0: host-aware tokenization (registered domain vs subdomain vs path) and a brand-regression evaluation set in training.

## Gray-zone policy
- Deliberate policy decision: URLs where `tau_allow < p < tau_block` (0.0082 < p < 0.8333) yield L1 verdict `ALLOW`.
- Rationale: In the validation split, approximately 78.5% of benign URLs fall within this gray zone under the conservative allow threshold (FNR <= 1%). Forcing a STEP_UP or human arbiter review on all gray-zone URLs would flood operational pipelines, defeat product usability, and exceed arbiter budgets.
- Metadata fields emitted in `metadata["url_model"]`:
  - `p`: Raw calibrated probability of phishing risk on the validation mix.
  - `z`: Combined ensemble logit before calibration.
  - `gray`: Boolean flag indicating `tau_allow < p < tau_block`.
  - `abstain`: Boolean flag indicating conformal prediction uncertainty (`tau_allow < p < tau_block`).
  - `tau_allow`: Lower threshold (0.008197).
  - `tau_block`: Upper threshold (0.833333).
- Consumption today in orchestration: Inspection of `orchestration/decision/engine.py` reveals that **no arbiter currently exists**, and the decision engine **does not read** `url_model.gray` or `url_model.abstain`.
- Residual risk: A suspicious URL with probability below `tau_block` (e.g. p = 0.80) is returned as `ALLOW` by L1. Protection against such URLs relies entirely on deterministic rule flags (R01-R10) and upstream/downstream layers (L2 prompt injection defense, L3 tabular fraud detection, and L3B GraphSAGE).
- Future requirement: Deploying an active arbiter tier that samples or selectively escalates high-risk gray-zone transactions when secondary risk factors (e.g., transaction amount, payee novelty) are present.

## Provenance and reproducibility
- Current bundle status: `reproducible: false`.
- Provenance disclosure:
  - Model weights were trained from an uncommitted working tree at commit `eda414d8a79b90db3f73abb935a33a8585fa5e5e` (`training_git_dirty: true`).
  - The artifact bundle was packaged from a dirty working tree (`packaging_git_dirty: true`).
  - Processed training datasets (`data/processed/url_dataset/`) are git-ignored and not versioned in git.
- Remaining clean-build requirements:
  1. Commit and tag a clean baseline commit for training.
  2. Pin and hash all raw data snapshots. Note: In `reports/data_manifest.json`, the Tranco dataset records the download date (`2026-11-13`) and SHA-256 hash, but **does not record the specific Tranco list ID**; future clean builds must record the exact list ID.
  3. Re-run `scripts/build_url_dataset.py` and `scripts/train_final_l1.py` in a reproducible container.
  4. Compare metrics against the baseline within tolerance (ROC-AUC delta <= 0.003).
  5. Publish and tag the clean reproducible bundle as a new version (e.g., `0.2.0`).

## Known gaps / not done
- No HTML or page-content model, so rule R08 is not used by the model. Rule R03 (PDF text vs OCR) is implemented but disabled and unmeasured (no real invoice PDFs). No character CNN.
- Rule R06 uses a 20-brand seed list, not Tranco top-N.
- Test-split history: the test split was scored twice. A first run kept shortener hosts in the data and fed rule flags to the combiner; validation flag counts then showed rule R10 firing on 318 malicious and 0 benign URLs (a sampling artifact). That run scored ROC-AUC 0.892, PR-AUC 0.652, recall at 0.5% FPR 0.179. The final version removes the artifact; its numbers are above.

## Reproducing
scripts/build_url_dataset.py, scripts/collect_cc_urls.py, scripts/eda_url.py, scripts/train_final_l1.py, then scripts/package_bundle.py, scripts/validate_artifacts.py and scripts/verify_serving.py. Seed 42. See manifest.json for versions and file hashes.