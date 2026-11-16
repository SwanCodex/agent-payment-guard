# SENTINEL Layer 1 (L1) — Deterministic Rules & URL Ensemble Model

SENTINEL Layer 1 (L1) is the first line of defense in SENTINEL's autonomous payment-security pipeline. It performs low-latency heuristic filtering and URL phishing risk scoring on incoming payment requests, invoices, and web pages before downstream layers (L2 prompt injection detection, L3A tabular fraud detection, and L3B fraud graph scoring) are engaged.

---

## 1. Architecture Overview

L1 combines two distinct, complementary paradigms:
1. **Deterministic Rules Engine (`sentinel_l1/rules/`)**: A fast, zero-dependency pattern and structural validation engine executing rules `R01` through `R10`. It inspects raw page content, parsed HTML, text, and transaction payload fields for structural anomalies, homoglyphs, and known attack vectors.
2. **Probabilistic URL Ensemble Model (`sentinel_l1/models/url_ensemble.py`)**: A dual-model machine learning ensemble that scores the phishing likelihood of input URLs based strictly on character n-grams and lexical features.

```
Incoming Request (SentinelRequest)
       │
       ▼
[Context Forwarding: orchestration/api/l1_context.py]
  - Validates allowlist: page_content, page_html, invoice_pdf_b64, payee_account, payee_domain, redirect_count
  - Rejects unknown/oversized/malformed fields without echoing content
  - Runs L1 if request.url is set OR at least one valid context field is present
       │
       ▼
L1In (url, page_content, payload)
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
   - Score: Policy contribution (capped at 0.60 for model risk; 0.49 for p < tau_block)
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
| **R04** | IBAN Validation | Raw text, `payee_account` | `step_up` | `L1:R04_IBAN_INVALID` | Validates ISO 13616 country formats and applies the MOD-97 checksum algorithm. Scans both explicit payee account and embedded text strings. Ignores non-IBAN formats (e.g. US account numbers). |
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

## 4. Context Forwarding in the Guard (`orchestration/api/l1_context.py`)

To allow L1 deterministic rules to inspect invoice data and payment attributes safely without bypassing contracts, `orchestration/api/l1_context.py` provides `build_l1_input(request)`.

### Allowlist Specification & Limits

| Source Field (Precedence: transaction then metadata) | Target in `L1In` | Strict Validation Rule | Target Engine Rule |
|---|---|---|---|
| `metadata["page_content"]` | `page_content` | `str`, at most 1,000,000 chars | R01, R04, R05, R09 |
| `metadata["page_html"]` | `payload["html"]` | `str`, at most 1,000,000 chars | R02, R08 |
| `metadata["invoice_pdf_b64"]` | `payload["pdf_b64"]` | `str`, base64 charset (`^[A-Za-z0-9+/=]+$`), at most 5,000,000 chars | R03 |
| `transaction/metadata["payee_account"]` | `payload["payee_account"]` | `str`, stripped, 1 to 64 chars, no internal whitespace | R04 |
| `transaction/metadata["payee_domain"]` | `payload["payee_domain"]` | `str`, 1 to 253 chars, hostname charset (`^[a-zA-Z0-9.-]+$`) | R06 |
| `metadata["redirect_count"]` | `payload["redirect_count"]` | `int` (strictly not `bool`), 0 to 50 | R10 |

### Precedence & Drop Rules:
- **First-Match Precedence**: For `payee_account` and `payee_domain`, `transaction` is evaluated first; if absent, `metadata` is checked.
- **Fail-Safe Dropping**: Invalid or oversized fields are dropped immediately. Values are never truncated, coerced, or auto-repaired.
- **Audit Reporting (No Echoing)**: The audit report contains forwarded field names (`["page_content", ...]`) and rejected field names with reason codes (e.g. `{"redirect_count": "TYPE_MISMATCH", "payload": "DISALLOWED_FIELD"}`). Payload content is **never** logged or echoed. Attached to `SentinelResponse.metadata["l1_context"]`.
- **Execution Gate**: L1 is invoked if `request.url` is set **OR** if at least one valid context field is present. Requests containing only invalid/unrecognized context without a URL do not trigger L1.

---

## 5. Gray-Zone Policy

### Deliberate Policy Decision
URLs scoring between $\tau_{\text{allow}}$ and $\tau_{\text{block}}$ ($0.008197 < p < 0.833333$) receive an L1 verdict of **`Verdict.ALLOW`**, with raw probability $p$ preserved in metadata.

### Why This Policy Exists
In held-out natural validation distributions, approximately **78.5% of legitimate benign URLs fall within this gray zone**. Under the conservative allow threshold ($\tau_{\text{allow}} = 0.0082$, ensuring malicious FNR $\le 1.0\%$), forcing a step-up or human arbiter escalation on all gray-zone URLs would create severe false-alarm fatigue, disrupt normal business operations, and rapidly exceed operational review budgets.

### Metadata Emitted
Every URL scoring step provides full transparency in `metadata["url_model"]`:
- `p`: Raw calibrated URL probability.
- `z`: Combiner logit.
- `gray`: `True` when $\tau_{\text{allow}} < p < \tau_{\text{block}}$.
- `abstain`: `True` when conformal prediction interval is inconclusive.
- `tau_allow`: `0.008197`
- `tau_block`: `0.833333`

### Honest Assessment of Consumption Today
Inspection of `orchestration/decision/engine.py` demonstrates that:
1. **No arbiter tier exists today.**
2. The orchestration decision engine **does not read** `url_model.gray` or `url_model.abstain`.
3. **Residual Risk**: A moderately suspicious URL (e.g. $p = 0.80$) will be permitted by L1 with `ALLOW`. Defense against such threats relies entirely on deterministic rule flags (R01–R10) and subsequent defense layers (L2 prompt injection analysis, L3 tabular fraud detection, and L3B GraphSAGE).
4. **Future Arbiter Requirement**: A future arbiter component must be implemented to ingest `url_model.gray` alongside composite risk signals (such as high transaction amounts or anomalous payee novelty) before authorizing execution.

---

## 6. Risk-Score Semantics

### Policy Contribution vs Fraud Probability
The score returned by Layer 1 (`LayerOut.score` $\in [0.0, 1.0]$) is a **policy contribution score**, NOT a calibrated fraud probability:
- Clean allow baseline: `0.0` (or $\min(p, 0.49)$ when URL model scores below $\tau_{\text{block}}$).
- Gray rule flag: `0.30`.
- Model high-risk / rule step_up / model failure: `0.60`.
- Deterministic hard block: `0.95`.

The calibrated probability from the URL model on the validation mix is stored strictly in `metadata["url_model"]["p"]` and must be interpreted as:
> *"Calibrated URL-model probability on the validation mix; not a fraud probability."*

### Explicit Metadata Declarations
Every L1 output provides:
```json
{
  "score_semantics": "policy_contribution_not_a_probability",
  "score_components": {
    "rule_flags": 0.0,
    "url_model_policy": 0.60
  }
}
```
In `orchestration/api/guard_service.py`, `SentinelResponse.metadata["risk_score_semantics"]` is declared as:
`"max of per-layer policy scores; not a calibrated probability"`.

---

## 7. Model Failure Handling & Fail-Closed Behavior

L1 strictly enforces a **fail-closed** architecture:
- If `request.url` is provided and the model is missing, corrupt, fails hash integrity checks, or encounters an unexpected exception:
  1. Reason code `"SYSTEM:MODEL_UNAVAILABLE"` is appended.
  2. L1 score is set to **`0.60`**.
  3. L1 verdict is **`Verdict.STEP_UP`**, always.
  4. Metadata records `status: "unavailable"` or `status: "error"`, with the exact error message in `reason`.
- **Zero Fail-Open Overrides**: There are no environment variables or configuration switches that can force an unavailable model to return `Verdict.ALLOW`.
- **Thread-Safe Cooldown**: Failed loads enter a thread-safe 30-second cooldown period, preventing redundant disk access and error cascades under concurrent workloads while allowing runtime recovery if missing model artifacts are restored.
- **Context-Only Requests**: When a request has no URL (`request.url is None`), the URL model is not applicable (`status: "not_applicable"`), and failure flags are not raised.

---

## 8. Artifact Integrity Verification at Load

When `UrlEnsemble` loads its bundle:
1. Reads `manifest.json` and parses file hashes.
2. Computes the SHA-256 hash of all bundle files (`model.txt`, `lr_ngram.npz`, `combiner.json`, `calibration.json`, `thresholds.json`, `feature_spec.yaml`, `model_card.md`, `eval_report.md`, `tld_risk.json`, `brands.json`).
3. Verifies that `booster.feature_name()` in LightGBM strictly equals the feature list in `feature_spec.yaml`.
4. If any hash fails or feature order diverges, it raises `IntegrityError("integrity_mismatch: <file>")`.
5. `run_l1` maps this to `status: "unavailable"`, reason `"integrity_mismatch: <file>"`, reason code `"SYSTEM:MODEL_UNAVAILABLE"`, and verdict `Verdict.STEP_UP`.

---

## 9. Provenance and Reproducibility

- **Current Bundle Status**: `reproducible: false`.
- **Provenance Disclosures**:
  - Model weights were trained from an uncommitted working tree at base commit `eda414d8a79b90db3f73abb935a33a8585fa5e5e` (`training_git_dirty: true`).
  - The bundle was packaged from a dirty working tree (`packaging_git_dirty: true`).
  - Training dataset files (`data/processed/url_dataset/`) are git-ignored.
- **Clean-Build Requirements for v0.2.0**:
  1. Commit and tag a clean baseline git commit for training.
  2. Pin and hash all raw dataset snapshots. Note: In `reports/data_manifest.json`, the Tranco entry lists date `2026-11-13` and SHA-256, but **does not record the specific Tranco list ID**; clean builds must record the exact list ID.
  3. Rerun `scripts/train_final_l1.py` inside a pinned container.
  4. Compare parity metrics (within 0.003 tolerance).
  5. Publish and tag the clean bundle as a new version.

---

## 10. The Google URL Anomaly & Mitigation

1. **Root Cause**:
   - In `train.csv`, URLs containing `google` in host count 891 malicious vs 77 benign (92.0% malicious), and registered domain `google.com` counts 862 malicious vs 5 benign (99.4% malicious).
   - In `LRScorer`, the host string `'google.com'` contributes `+9.5361` to the log-odds decision for `https://www.google.com/`, producing $p_{\text{lr}} = 0.8232$, LightGBM $p_{\text{gbm}} = 0.4291$, combiner $s = 1.5883 \cdot z + 3.9714$, and calibrated $p = 0.9844 \ge \tau_{\text{block}}$.
2. **Policy Mitigation**:
   - The anomaly is **mitigated by policy (STEP_UP instead of BLOCK)** and is **not fixed in model 0.1.0**. Legitimate requests to `google.com` step up to human review with score `0.60`.
   - Recommended v0.2.0 fix: host-aware tokenization separating registered domain, subdomain, and path, and brand regression evaluation sets.

---

## 11. Testing & Verification Commands

```powershell
$env:PYTHONPATH=".;contracts;layer1_heuristics;layer3_fraud_tabular;layer3_fraud_graph"

# 1. Full Test Suite across Contracts, L1, and Orchestration
.\layer1_heuristics\.venv\Scripts\python.exe -m pytest -o rootdir=. --import-mode=importlib contracts/tests layer1_heuristics/tests orchestration/tests -v

# 2. Artifact Integrity and Smoke Validation
.\layer1_heuristics\.venv\Scripts\python.exe layer1_heuristics/scripts/validate_artifacts.py

# 3. Serving Parity Verification against 25k Validation and Test Sets
.\layer1_heuristics\.venv\Scripts\python.exe layer1_heuristics/scripts/verify_serving.py
```
