# SENTINEL Layer 1 (L1) — Deterministic Rules & URL Ensemble Model

SENTINEL Layer 1 (L1) is the first line of defense in SENTINEL's autonomous payment-security pipeline. It performs low-latency heuristic filtering and URL phishing risk scoring on incoming payment requests, invoices, and web pages before downstream layers (L2 prompt injection detection, L3A tabular fraud detection, and L3B fraud graph scoring) are engaged.

---

## 1. Architecture Overview

L1 combines two distinct, complementary paradigms:
1. **Deterministic Rules Engine (`sentinel_l1/rules/`)**: A fast, zero-dependency pattern and structural validation engine executing rules `R01` through `R10`. It inspects raw page content, parsed HTML, text, and transaction payload fields for structural anomalies, homoglyphs, and known attack vectors.
2. **Probabilistic URL Ensemble Model (`sentinel_l1/models/url_ensemble.py`)**: A dual-model machine learning ensemble that scores the phishing likelihood of input URLs based strictly on character n-grams and lexical features.

```
Incoming Request (L1In: url, page_content, payload)
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
Deterministic Rules (R01-R10)   URL Model Scorer (UrlEnsemble)   HTML Extraction (_html.py)
   - Unicode / Control (R01)       - Character N-Gram LR Scorer     - Visible Text
   - Hidden Web Text (R02)         - Lexical LightGBM Scorer        - Hidden Text
   - PDF Divergence (R03)          - Logit Combiner (1.5883z+3.9714)- Anchor Links
   - IBAN Mod-97 (R04)             - Isotonic Calibration
   - Payment Changes (R05)
   - Lookalike / Skeleton (R06)
   - Punycode IDN (R07)
   - Link Mismatches (R08)
   - Encoded Blobs (R09)
   - URL Shorteners (R10)
       │                                 │
       ▼                                 ▼
 Rule Flags (severity, code)     Calibrated p, z, gray, abstain
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           Decision Logic (run_l1)
   - Hard BLOCK: Explicit deterministic block rules ONLY
   - STEP_UP: Rule step_up OR URL model high risk (p >= tau_block) OR unavailable model
   - ALLOW: Clean input / gray zone without flags
   - Score: Capped at 0.60 for model risk (0.49 for p < tau_block); raw p in metadata
                        │
                        ▼
            LayerOut (Verdict, Score, Codes, Metadata)
```

---

## 2. Deterministic Rules Engine (R01–R10)

The deterministic rules engine executes sequential, stateless inspections. Flags specify a severity level (`gray`, `step_up`, or `block`), a standard reason code, and evidence.

| Rule | Name | Detection Scope | Severity | Reason Code | Description / Thresholds |
|---|---|---|---|---|---|
| **R01** | Unicode & Hidden Control Characters | Raw text, HTML, URL | `step_up` | `L1:R01_HIDDEN_CONTROL_CHAR`, `L1:R01_MIXED_SCRIPT` | Flags zero-width spaces (`\u200b`–`\u200f`), bidi overrides (`\u202a`–`\u202e`), word joiners (`\u2060`), and mixed-script Cyrillic/Greek/Latin within words. Allows leading BOM (`\ufeff`) and emoji ZWJ. |
| **R02** | Hidden Webpage Text | Parsed HTML | `step_up` (flag), forwarded | `L1:R02_HIDDEN_TEXT` | Identifies hidden text via CSS (`display:none`, `visibility:hidden`, `font-size:0`, `opacity:0`, off-screen positioning). Forwards text to L2 if $\ge 15$ words; flags `step_up` if $\ge 1276$ words (99.5th percentile of Common Crawl sample). |
| **R03** | PDF Text Divergence | Base64 PDF | `step_up` (flag), forwarded | `L1:R03_PDF_TEXT_DIVERGENCE` | Compares programmatic text layer (`pdfplumber`) against visible OCR rendering (`pytesseract`). Flags if $\ge 10$ tokens and $\ge 50\%$ missing from render. **Status: Disabled by default (`R03_ENABLED = False`)** pending dataset calibration. Returns `[], []` immediately when disabled. |
| **R04** | IBAN Validation | Raw text, `payee_account` | `step_up` | `L1:R04_IBAN_INVALID` | Validates ISO 13616 country formats and applies the MOD-97 checksum algorithm. Scans both the explicit payee account and embedded text strings. Ignores non-IBAN formats (e.g. US account numbers). |
| **R05** | Payment Detail Change | Extracted text | `gray`, forwarded | `L1:R05_PAYMENT_DETAIL_CHANGE` | Flags phrases indicative of supplier bank account changes (e.g., "new bank details", "account has changed"). Forwards context window ($\pm 100$ chars) to L2. |
| **R06** | Lookalike & Homoglyph Domains | URL, `payee_domain` | `step_up` / `gray` | `L1:R06_LOOKALIKE_DOMAIN` | Protects top brands against typo-squatting and homoglyphs. Uses skeleton normalization (`rn` $\to$ `m`, `0` $\to$ `o`, `1` $\to$ `l`), optimal string alignment edit distance ($\le 1$ for short brands $<8$ chars, $\le 2$ for brands $\ge 8$ chars), and cross-TLD matching. Genuine country-code sites (e.g. `google.in`) are permitted. |
| **R07** | Mixed-Script Punycode | URL host | `step_up` | `L1:R07_PUNYCODE_HOST` | Decodes IDN labels (`xn--`) and checks whether multiple scripts (Latin, Cyrillic, Greek) are mixed in a single domain label. Pure single-script IDNs are permitted. |
| **R08** | Link Text Mismatch | Parsed HTML anchors | `gray` | `L1:R08_LINK_TEXT_MISMATCH` | Flags hyperlinks where anchor text visibly advertises a protected brand domain (e.g. `paypal.com`) but the destination `href` navigates to an unrelated third-party domain. Same-owner shorteners (e.g. `amzn.to`, `goo.gl`) are whitelisted. |
| **R09** | Encoded Blobs | Extracted text | `gray`, forwarded | `L1:R09_ENCODED_BLOB` | Detects Base64 ($\ge 40$ chars), Hex ($\ge 20$ bytes), and URL-encoded strings ($\ge 8$ encodings) that decode to printable ASCII/UTF-8 with length $\ge 12$ chars. Forwards decoded payload to L2. |
| **R10** | Shorteners & Redirect Chains | URL, `redirect_count` | `gray` | `L1:R10_URL_SHORTENER`, `L1:R10_LONG_REDIRECT_CHAIN` | Detects known URL shorteners (`bit.ly`, `tinyurl.com`, `t.co`, etc.) and redirect chains exceeding 3 hops. |

> **Note on Rule Severity**: Currently, no shipped deterministic rule emits `severity == "block"`. Shipped rules emit `severity == "step_up"` or `severity == "gray"`. The deterministic-BLOCK engine branch is fully implemented and tested via mock overrides should critical blocking rules be introduced in future policies.

---

## 3. URL Ensemble Model Architecture

The URL model is an ensemble trained to predict the probability that a given URL is a phishing link using only the URL string:

1. **Character N-Gram Logistic Regression (`LRScorer`)**:
   - Analyzes sublinear TF-IDF character 2-to-5 grams extracted from the stripped URL (scheme and leading `www.` removed).
   - Utilizes `FeatureHasher` with $2^{20} = 1,048,576$ buckets.
   - Evaluated using NumPy: decision $d = \frac{v \cdot \text{coef}}{\|v\|_2} + b$, probability $p_{\text{lr}} = \sigma(d)$.
2. **Lexical LightGBM Classifier (`Booster`)**:
   - 163 trees trained over 25 features (19 lexical features like length, dots, digit ratio, token entropy; 5 brand and risk-keyword counts; 1 public suffix risk prior `tld_risk`).
   - Evaluated as $p_{\text{gbm}} = \text{Booster.predict}(\mathbf{x})$.
3. **Logit Average Combiner & Isotonic Calibration**:
   - Logit-space combination: $z = 0.5 \cdot (\text{logit}(p_{\text{lr}}) + \text{logit}(p_{\text{gbm}}))$.
   - Linear score scaling: $s = 1.58830047 \cdot z + 3.97142976$ (from `combiner.json`).
   - Calibration: Isotonic interpolation mapping $s \to p \in [0, 1]$.
   - Conformal Prediction: Threshold $q_{\text{hat}} = 0.2861$ ($\alpha = 0.10$). Abstains when the prediction interval does not contain exactly one class.

---

## 4. Model Version, Provenance & Calibration

- **Artifact Version**: `0.1.0` located at `artifacts/layer1/url_ensemble/0.1.0/`.
- **Training Data**:
  - Malicious: PhishTank verified online-valid dataset (16,370 train / 3,466 val / 3,537 test).
  - Benign: Tranco Top-1M homepages + Common Crawl CC-MAIN-2026-43 deep URLs (104,505 train / 22,283 val / 22,430 test).
  - Splitting: Domain-disjoint 70/15/15 split. Malicious test set is time-ordered (newest domains from 2026).
- **Provenance Preservation**:
  - Model weights were trained from an uncommitted tree at base commit `eda414d8a79b90db3f73abb935a33a8585fa5e5e`.
  - `package_bundle.py` strictly preserves `training_git_sha`, `training_date`, and `training_note`, recording packaging metadata (`packaging_git_sha`, `packaging_date`) separately without overwriting training provenance.
- **Operating Thresholds**:
  - Allow Threshold ($\tau_{\text{allow}}$): `0.008197` (Validation malicious FNR $\le 1.0\%$).
  - Block/High-Risk Threshold ($\tau_{\text{block}}$): `0.833333` (Validation benign FPR $\le 0.5\%$).
  - Conformal Threshold ($q_{\text{hat}}$): `0.286064` ($\alpha = 0.10$, 90% target coverage).
- **Evaluation Parity**:
  - Validation: ROC-AUC `0.9374`, PR-AUC `0.7999`, Recall@0.5%FPR `44.5%`.
  - Test: ROC-AUC `0.8958`, PR-AUC `0.6438`, Recall@0.5%FPR `18.2%`.
  - Latency: UrlEnsemble single-URL score p95 = `0.90 ms`, `run_l1` p95 = `2.56 ms` (budget $\le 20\text{ ms}$).

---

## 5. Decision-Policy Integration & Score Mapping

### Probabilistic vs Deterministic Findings:
- **No Standalone Model Hard BLOCK**: The URL ensemble is a probabilistic heuristic signal. High URL model risk ($p \ge \tau_{\text{block}}$) emits `Verdict.STEP_UP` accompanied by reason code `"L1:URL_MODEL_HIGH_RISK"`. The model cannot unilaterally hard-block payments.
- **L1 Score Mapping**:
  - When $p \ge \tau_{\text{block}}$, model score contribution is `0.60` (standard step-up score).
  - When $p < \tau_{\text{block}}$, model score contribution is $\min(p, 0.49)$ (staying within the allow range).
  - The unmapped, calibrated probability $p$ is always preserved in `metadata["url_model"]["p"]`.
  - L1 score is computed as `max(rule_flag_scores, model_contribution)`.
- **Deterministic BLOCK Authority**: Only explicit deterministic rule violations with `severity == "block"` issue a hard `Verdict.BLOCK` (`score = 0.95`).
- **Advisory Gray Zone**: Approximately 78.5% of benign URLs score between $\tau_{\text{allow}}$ and $\tau_{\text{block}}$ ($0.0082 < p < 0.8333$). This gray zone is advisory. Clean URLs in the gray zone without rule flags evaluate to `Verdict.ALLOW`.

### Model Metadata Schema:
`metadata["url_model"]` is **always** a dictionary with the following schema:
- `status`: One of `{"ok", "unavailable", "error", "not_applicable"}`.
- `reason`: String explanation of failure/error, or `null`.
- `version`: String version (e.g. `"0.1.0"`), or `null`.
- When `status == "ok"`: Also contains `p`, `z`, `gray`, `abstain`, `tau_allow`, `tau_block`.

### Model-Unavailable Behavior & Fail-Safe:
- If a request contains a URL and the model is missing, corrupt, or throws an exception during inference:
  1. Reason code `"SYSTEM:MODEL_UNAVAILABLE"` is appended to `reason_codes`.
  2. L1 score is set to `0.60` (fail-closed).
  3. L1 verdict is `Verdict.STEP_UP` by default (ensuring orchestration never silently approves unscored URLs).
  4. Metadata records `status: "unavailable"` or `status: "error"`.
- **Local Development Opt-Out**:
  - Setting the environment variable `SENTINEL_L1_ON_MODEL_FAILURE=allow` allows local development without downloaded model artifacts to return `Verdict.ALLOW` when no deterministic rule flags fire.
- **Thread-Safe Cooldown**:
  - Model loading failures are protected by a thread lock (`_MODEL_LOCK`) and retried after a 30-second cooldown period, preventing repeated failing disk I/O under high concurrency while allowing runtime recovery if artifacts are deployed.

---

## 6. Artifact Integrity Verification at Load

When `UrlEnsemble` is instantiated:
1. It reads `manifest.json` and parses the model version.
2. It verifies the SHA-256 hash of all 10 bundle files against `manifest["files"]`.
3. It loads `feature_spec.yaml` and confirms that `booster.feature_name()` exactly matches the YAML feature order.
4. If any file hash fails or feature names differ, it raises `IntegrityError("integrity_mismatch: <file>")`.
5. `run_l1` catches this mismatch, records `status: "unavailable"`, sets `reason: "integrity_mismatch: <file>"`, and steps up with `"SYSTEM:MODEL_UNAVAILABLE"`.

---

## 7. Known Limitations & The Google URL Anomaly

1. **The Google URL Anomaly (PhishTank Brand Representation Artifact)**:
   - *Evidence & Measurements*:
     - In `train.csv`, URLs whose host contains "google" count 891 malicious vs 77 benign (92.0% malicious).
     - The registered domain `google.com` in `train.csv` counts 862 malicious rows vs 5 benign rows (99.4% malicious).
     - In `LRScorer`, the host part `'google.com'` contributes `+9.5361` to the log-odds decision for `https://www.google.com/`, producing $p_{\text{lr}} = 0.8232$, LightGBM $p_{\text{gbm}} = 0.4291$, combiner $s = 1.5883 \cdot z + 3.9714$, and calibrated $p = 0.9844 \ge \tau_{\text{block}}$.
     - Similar high scores occur for other brand sign-in endpoints: `accounts.google.com/signin` ($p = 1.0000$), `github.com/login` ($p = 1.0000$), `netflix.com/login` ($p = 1.0000$), `dropbox.com/login` ($p = 1.0000$), and `paypal.com/signin` ($p = 0.9397$).
   - *Mitigation*: The anomaly is **mitigated by policy (STEP_UP instead of BLOCK)** and is **not fixed in model 0.1.0**. Legitimate payments and authentications involving these domains will still trigger `STEP_UP`.
   - *Future Fix (v0.2.0)*: Implement host-aware tokenization separating registered domain, subdomain, and path, and evaluate against a curated brand-regression test set.
2. **External Domain Generalization**:
   - On the out-of-distribution PhiUSIIL dataset, ROC-AUC drops to `0.661` (vs `0.896` on PhishTank test).
3. **Disabled/Incomplete Rules**:
   - Rule `R03` (PDF text divergence) is disabled (`config.R03_ENABLED = False`) pending real invoice PDF calibration corpora.

---

## 8. Orchestration Gap Analysis

In [`orchestration/api/guard_service.py`](file:///c:/Users/Shreya/Desktop/agent-payment-guard/orchestration/api/guard_service.py):
```python
if request.url is not None:
    l1_input = L1In(
        request_id=request.request_id,
        agent_id=request.agent_id,
        user_id=request.user_id,
        transaction_id=request.transaction_id,
        url=request.url,
    )
    outputs.append(run_l1(l1_input))
```
- **The Gap**: `SentinelRequest` currently lacks `page_content` or `payee` fields. Consequently, `guard_service.process_request()` passes only `url` to L1. Rules `R02` (hidden text), `R03` (PDF divergence), `R04` (payee account IBAN), `R05` (payment change text), `R08` (link mismatch HTML), and `R09` (encoded blobs in text) cannot be triggered through `/v1/guard` requests.
- **Proposed Contract-Compatible Wiring** (for future approval):
  Forward `page_content` and `payload` from `SentinelRequest.metadata`:
  ```python
  l1_input = L1In(
      request_id=request.request_id,
      agent_id=request.agent_id,
      user_id=request.user_id,
      transaction_id=request.transaction_id,
      url=request.url,
      page_content=request.metadata.get("page_content"),
      payload=request.metadata.get("payload", {}),
  )
  ```

---

## 9. Testing & Verification

### Running the Test Suite
```powershell
$env:PYTHONPATH=".;contracts;layer1_heuristics;layer3_fraud_tabular;layer3_fraud_graph"
.\layer1_heuristics\.venv\Scripts\python.exe -m pytest -o rootdir=. --import-mode=importlib contracts/tests layer1_heuristics/tests layer3_fraud_graph/tests -v
```

### Artifact-Only Validation Procedure
```powershell
.\layer1_heuristics\.venv\Scripts\python.exe layer1_heuristics/scripts/validate_artifacts.py
```
Validates manifest file hashes, monotonic calibration, feature order matching LightGBM, and deterministic scoring.

### Serving Parity Verification Note
```powershell
.\layer1_heuristics\.venv\Scripts\python.exe layer1_heuristics/scripts/verify_serving.py
```
> **Reproducibility Note**: `verify_serving.py` scores the full 25,749 validation and 25,967 test rows. These processed CSV files (`data/processed/url_dataset/`) were present locally on the development machine (git-ignored) and are not reproducible from a clean clone without downloading the dataset. In environments without the processed CSVs, use `validate_artifacts.py` for artifact-only smoke verification.
