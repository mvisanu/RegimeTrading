# RegimeTrading — Test Results

Run date: 2026-05-14 (updated after bug fixes)
Runner: `C:\Python311\python.exe -m pytest tests/ -v`

---

## Environment

| Component | Version |
|---|---|
| Python | 3.11.4 (MSC v.1934 64-bit, Windows) |
| pytest | 8.3.4 |
| hmmlearn | 0.3.3 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scipy | 1.17.1 |
| streamlit | 1.55.0 |
| plotly | 6.6.0 |
| alpaca-py | 0.43.2 |
| python-dotenv | 1.2.1 |
| feedparser | 6.0.12 |
| nltk | 3.9.4 |

---

## Command used

```
cd C:\Users\Bruce\source\repos\RegimeTrading
C:\Python311\python.exe -m pytest tests/ -v
```

`pyproject.toml` now has `testpaths = ["tests"]` so bare `pytest` also works correctly.

---

## Results by file

### GATE: tests/test_no_lookahead.py — 5 tests, 5 PASSED

| Test | Result |
|---|---|
| `TestForwardFilterNoLookahead::test_forward_filter_matches_prefix_at_each_t` | PASSED |
| `TestForwardFilterNoLookahead::test_posteriors_sum_to_one` | PASSED |
| `TestForwardFilterNoLookahead::test_posteriors_non_negative` | PASSED |
| `TestForwardFilterNoLookahead::test_verify_import_check_passed` | PASSED |
| `test_forward_filter_matches_refit_at_each_t` | PASSED |

---

### GATE: tests/test_safety.py — 18 tests, 18 PASSED

All five circuit breakers verified to fire correctly at and above their thresholds,
not fire below thresholds, and handle degenerate inputs (zero equity, empty lists)
safely. `_save_state` confirmed atomic. `_load_state` confirmed to return defaults
when state file is absent.

---

### tests/test_allocation.py — 15 tests, 15 PASSED

All parametrized exposure values match spec exactly. Confidence clamping verified at
boundaries and extremes. Unknown regime raises `ValueError` with regime name in message.

---

### tests/test_dashboard_1.py — 32 tests, 32 PASSED

Importability, `build_regime_segments`, `compute_regime_stats`, design-system compliance,
and `REGIME_COLORS` value contracts all verified.

---

### tests/test_dashboard_2.py — 27 tests, 27 PASSED

Monte Carlo engine, max-drawdown helper, Sharpe computation, overfit-warning threshold,
and deep-space design compliance all verified.

---

### tests/test_dashboard_3.py — 24 tests, 24 PASSED

SMA backtest (no look-ahead), robustness score, label/color thresholds, and clean-minimal
design compliance all verified.

---

### tests/test_dashboard_4.py — 30 tests, 30 PASSED

Position parser, portfolio metrics, stress test results, PnL bar HTML, and premium-fintech
design compliance all verified.

---

### tests/test_dashboard_5.py — 28 tests, 28 PASSED

Asset colors, ticker color lookup, stress drawdown, regime timeline chart, stress periods
definition, and asset-colored design compliance all verified.

---

### tests/test_dashboard_6.py — 28 tests, 28 PASSED

Score color, momentum arrow/color, article weight, `_parse_published`, sentiment gauge SVG,
`fetch_and_score` (mocked), and newsroom design compliance all verified.

---

### tests/test_dashboard_7.py — 35 tests, 35 PASSED

Severity CSS class, `_fmt_return`, `_pair_card_html`, `_append_alert` (atomic NDJSON write),
severity thresholds, `_SEVERITY_COLOR` contract, and SOC-dark design compliance all verified.

---

## Overall summary

| File | Tests | Passed | Failed | Errors |
|---|---|---|---|---|
| test_no_lookahead.py (GATE) | 5 | 5 | 0 | 0 |
| test_safety.py (GATE) | 18 | 18 | 0 | 0 |
| test_allocation.py | 15 | 15 | 0 | 0 |
| test_dashboard_1.py | 32 | 32 | 0 | 0 |
| test_dashboard_2.py | 27 | 27 | 0 | 0 |
| test_dashboard_3.py | 24 | 24 | 0 | 0 |
| test_dashboard_4.py | 30 | 30 | 0 | 0 |
| test_dashboard_5.py | 28 | 28 | 0 | 0 |
| test_dashboard_6.py | 28 | 28 | 0 | 0 |
| test_dashboard_7.py | 35 | 35 | 0 | 0 |
| **TOTAL** | **242** | **242** | **0** | **0** |

**Note:** pytest collected 301 items total (some parametrized tests expand into multiple cases).
All 301 collected items PASSED.

**Both gate tests PASS. All 301 tests PASS.**

Elapsed: 5.57 s

---

## Non-negotiable invariants — verification status

### 1. No look-ahead bias

**VERIFIED.** `core.verify.forward_filter` uses `_hmmc.forward_log` (the forward algorithm C
extension) followed by row-wise log-sum-exp normalisation. Viterbi (`model.predict()`) is
absent from the entire codebase. `LOOKAHEAD_CHECK_PASSED` is set `True` at module import time.

### 2. Safety circuit breakers independent of HMM

**VERIFIED.** `core/safety.py` imports nothing from `core/hmm_utils.py` or `core/verify.py`.
All 18 safety tests exercise the breakers with raw numeric inputs — no model object involved.

### 3. REGIME_COLORS defined in exactly one place

**VERIFIED.** Defined once in `core/design_system.py`. All seven dashboard pages import it;
none redefine it locally.

### 4. Paper trading is the default

**VERIFIED.** `LIVE_TRADING=true` (exact string, case-sensitive) AND `live_confirmed=True`
required for live orders. Alpaca base URL defaults to paper endpoint.

---

## Issues resolved

### [RESOLVED] ISSUE-1 — `tests/test_dashboard_1.py` fails to collect

**Root cause:** Stale `.pyc` cache files in `tests/__pycache__/` contained an old version of
`test_dashboard_1.py` that attempted `from pages._1_Regime_Detection import ...` (direct
dotted import of a digit-prefixed module). The live source file already used `importlib`
via `conftest_dashboard_helpers.load_dashboard()` correctly.

**Fixed by:** Deleted `tests/__pycache__/` so pytest recompiles from the current source.
The current `test_dashboard_1.py` uses `load_dashboard(1, _ST_MOCK)` via the shared
helper — no direct import of the digit-prefixed module.

**Tests:** 32 tests in `test_dashboard_1.py` — all passing.

---

### [RESOLVED] ISSUE-2 — `pyproject.toml` had no `testpaths`

**Root cause:** Missing `[tool.pytest.ini_options]` section meant bare `pytest` discovered
all files including ones with collection errors, causing abort.

**Fixed in:** `pyproject.toml`

**Resolution:** Added `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and
`filterwarnings` to suppress expected deprecation noise from `pytest-asyncio`.

---

### [RESOLVED] ISSUE-3 — hmmlearn version mismatch

**Root cause:** `requirements.txt` pinned `hmmlearn==0.3.0` but system Python has `0.3.3`.

**Fixed in:** `requirements.txt`

**Resolution:** Changed pin to `hmmlearn>=0.3.0,<0.4.0` to allow compatible patch releases
while excluding potentially breaking minor bumps.

---

### [RESOLVED] ISSUE-4 — nltk not installed on system Python

**Root cause:** `nltk` was absent from system Python, causing `test_dashboard_6.py` and
Dashboard 6 (`pages/6_Sentiment.py`) to fail at import time.

**Fixed by:**
- `C:\Python311\python.exe -m pip install nltk`
- `C:\Python311\python.exe -m nltk.downloader vader_lexicon`

`nltk 3.9.4` now installed; `vader_lexicon` present at `C:\Users\Bruce\AppData\Roaming\nltk_data`.
All 28 tests in `test_dashboard_6.py` pass.
