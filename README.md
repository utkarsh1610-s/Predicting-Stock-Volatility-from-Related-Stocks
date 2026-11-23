# Market Contagion: Predicting Stock Volatility from Related Stocks

CS 6220 project. See `Project_Proposal_Market_Contagion.md` for the research design.

## Setup

Requires Python 3.13. From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the data (use the shared snapshot, don't re-download)

1. Get `finance_data_snapshot_2026-11-14.zip` from the team Drive and unzip it in the
   project root. It creates `data/` with the raw price CSVs, the constituent list, the
   run log and `MANIFEST.sha256`.
2. `python scripts/verify_data.py` checks every file against the manifest (504 files).
3. `python scripts/clean.py` rebuilds `data/processed/` (about a minute). The output is
   deterministic: the same snapshot always gives the same parquet files.

Steps 0 and 1 below are only for rebuilding the snapshot from scratch, which will
**not** reproduce it exactly (see "The data is a frozen snapshot").

## Data pipeline

Run from the project root with the venv active:

| Step | Script | Output |
|---|---|---|
| 0 | `scripts/collect_constituents.py` | `data/sp500_constituents.csv` (+ raw HTML in `data/raw_wiki/`) |
| 1 | `scripts/download_prices.py` | `data/raw_full/<TICKER>.csv`, one per stock plus `VIX.csv` |
| check | `scripts/verify_data.py` | compares the raw snapshot with `data/MANIFEST.sha256` |
| 2 | `scripts/clean.py` | `data/processed/{stocks,vix}.parquet`, `cleaning_log.csv`, `wide_range_days.csv` |
| 3 | `scripts/features.py` | `data/features/panel.parquet`, `winsor_bounds.json` |
| 4 | `scripts/graphs.py` | `data/graphs/{sector,corr}_edges.parquet`, `data/features/neighbors.parquet` |
| 5 | `scripts/baselines.py` | `results/baseline_*.csv`, `data/predictions/<model>.parquet` |
| 6 | `scripts/trees.py` | `results/trees_*.csv`, `data/predictions/gbt*.parquet` |

Settings shared by all steps live in `scripts/config.py`.

### Analysis scripts

These test whether the neighbor gain comes from specific links or from the market-wide volatility
level. All score **validation only**; the test years stay untouched. Each script states its decision
rule in its docstring, and that rule was committed before the script was run.

| Script | Question | Output |
|---|---|---|
| `scripts/sensitivity.py` | Do the untested design choices (graph window, refresh rate, neighbor count, weighting, winsorizing, pooled vs per-stock HAR) change the neighbor gain? | `results/sensitivity.csv` |
| `scripts/placebo.py` | Does a market-wide average, or a random twin of each graph, do as well as the real graphs (linear HAR)? | `results/placebo.csv` |
| `scripts/placebo_trees.py` | The same placebo with LightGBM | `results/placebo_trees.csv` |
| `scripts/sector_check.py` | Does any single sector show a graph effect beyond the market? | `results/sector_check.csv` |
| `scripts/sector_clustering.py` | Does volatility cluster within sectors beyond the market on the same day? (descriptive, 2010-2021 only) | `results/sector_clustering.csv`, `results/sector_episodes.csv` |
| `scripts/own_history_test.py` | Is the sector's information redundant once the stock's own history is known? | `results/own_history.csv` |

Arguable choices and what each experiment found are logged in `DECISIONS.md`.

## The data is a frozen snapshot

- Constituents scraped from Wikipedia on 2026-11-14; prices downloaded from Yahoo
  Finance the same day (yfinance 1.8.0, 2010-01-01 to 2025-12-31, `auto_adjust=True`).
  See `data/run_info.txt`.
- Do not re-download to "refresh" it. Yahoo rescales adjusted history after every
  dividend or split, and Wikipedia membership changes, so a new pull will not match.
  Step 0 refuses to overwrite the constituent list without `--force`, and step 1
  skips tickers already on disk.

## Cleaning decisions (`scripts/clean.py`)

1. **Listing starts.** Seven tickers have early history from a different security
   (foreign/OTC quotes, a SPAC shell, pre-relisting trading). Each starts at its real
   US listing: SW, FERG, AMCR, VRT, VST, CHTR, HWM. Dates and reasons are in `config.py`.
2. **Row validity.** Drop rows with non-positive or missing prices, zero volume, zero
   high-low range (Garman-Klass would be 0), or a high/low outside the open/close by
   more than float noise.
3. **Minimum history.** Tickers need at least 126 clean rows (drops Q).
4. Days with a high/low ratio above 1.5 are **kept** and written to `wide_range_days.csv`.
   They are real events (6 May 2010 flash crash, 24 Aug 2015, March 2020, BIIB June 2021)
   but produce extreme volatility values; handle them in the feature step (e.g. winsorize).

Result: 499 tickers, 1,878,794 rows; 480 / 490 / 499 stocks in train / val / test.

## Features and target (`scripts/features.py`)

One row per ticker per trading day. Target `y` is **next-day log Garman-Klass volatility**.
Inputs: `har_1/5/22` (log of mean volatility over the last 1/5/22 days), `ret`, `dlogvol`,
`vix`, plus `lv` (today's log volatility) and `sector`.

- **No lookahead.** Every feature at day *t* uses data up to day *t* only; `y` is the one
  forward-looking column. `tests/test_features.py` checks this by perturbing the future.
- **Split by the date of `y`**, not of the features. The 2019-12-31 row has a 2020 label,
  so it is validation and never trains a model.
- **Winsorization** of daily log volatility at the 0.1% / 99.9% quantiles, with bounds taken
  from the **training split only** (`winsor_bounds.json`). `y` is the winsorized value, so
  all models are scored on the same clipped target; QLIKE needs variance, i.e. `exp(2 * y)`.
- **Gaps.** Windows need every day, so a missing trading day removes the rows whose
  features or label touch it (554 rows on this snapshot, on top of the 22 rows per ticker
  lost to warm-up and the unlabeled last day).

Result: 499 tickers, 1,867,262 rows; train 1,125,805, val 245,001, test 496,456.

## Graphs and neighbor features (`scripts/graphs.py`)

Two ways to define "related stocks", each turned into neighbor features aligned row for row
with `panel.parquet` (merge on `ticker`, `date`):

- **Sector graph** (fixed): every other stock in the same GICS sector (20-80 peers).
  Features `sec_har_1/5/22` = mean of the peers' `har_*` on the same day; `sec_n` = how many peers had data.
- **Correlation graph** (rebuilt monthly): the 10 stocks whose daily returns correlated
  most with it over the 60 trading days **before** the month starts. Features `cor_har_1/5/22`, `cor_n`.
  About half of the correlation neighbors are in the same sector, so the two graphs do differ.

Rules, all tested in `tests/test_graphs.py`:

- **No lookahead.** A month's graph never sees any day it is applied to. Neighbor values are
  same-day, never later.
- **Pairwise-complete correlation.** A pair needs 40 shared days in the window. A stock with
  fewer (new listings; every stock in Feb-Mar 2010) has no neighbors that month.
- **Missing neighbor data.** A neighbor counts on a day only if it has `har_*` that day, and
  the neighbor mean is NaN if none do. Neighbor availability does not depend on whether
  tomorrow's label exists.
- **NaNs are not filled.** 1% of rows have no `cor_*` features (early 2010, new listings). When comparing a
  model with and without neighbors, score both on the rows where the neighbor features exist.

## Baselines and evaluation (`scripts/baselines.py`, `scripts/metrics.py`)

`python scripts/baselines.py` fits on the training years and reports train and validation.
The test years are scored only with `--include-test`; run that once, at the end, with all models.

| Model | Forecast of next-day log volatility |
|---|---|
| `naive` | today's value |
| `har` | OLS on `har_1/5/22`, pooled across stocks |
| `har_sector` | `har` + sector peers' averaged `har_*` |
| `har_corr` | `har` + correlation neighbors' averaged `har_*` |
| `har_q`, `har_sector_q`, `har_corr_q` | the same three linear models, trained to minimize QLIKE directly (no smearing) |

- **Same rows for every model.** Fit and scored on rows where both neighbor feature sets
  exist (19,161 of 1,867,262 rows dropped), so differences come from the features only.
- **QLIKE-trained HAR** (`har_q`) is only 0.8% better than OLS with smearing on validation (0.2866 vs 0.2888, p=0.015),
  so the smearing correction is a good approximation for the linear models. Neighbor gains are unchanged on top of it.
- **QLIKE** uses variance `exp(2*yhat)` rescaled by a training-set smearing factor, because a
  log-model forecasts the median variance, which sits below the mean. The unscaled number is
  kept as `qlike_raw`. MSE is on log volatility.
- **Diebold-Mariano** tests use each day's cross-sectional mean QLIKE and a Newey-West variance.

Validation 2020-2021 (QLIKE, lower is better):

| Model | QLIKE | vs HAR | DM p vs HAR |
|---|---|---|---|
| naive | 0.481 | +66.6% | |
| har | 0.289 | | |
| har_sector | 0.268 | -7.0% | 0.012 |
| har_corr | 0.271 | -6.1% | 0.021 |

Read these with care. The gain is concentrated in the COVID crash: with Feb-Apr 2020 removed
it is -2.5% (sector) and -1.9% (correlation), still significant (p < 0.001) but much smaller.
On validation the fixed sector graph beats the correlation graph. Nothing is tuned yet, and
the test years are untouched.

## Gradient-boosted trees (`scripts/trees.py`)

LightGBM on the same rows as the baselines. `gbt` uses `har_1/5/22`, return, volume change and
VIX; `gbt_sector` and `gbt_corr` add the same neighbor features as the HAR models.
`python scripts/trees.py` takes about 3 minutes; `--include-test` scores the test years (once, at the end).

- **Tuning.** Each model gets the same 12-config grid (leaves, learning rate, min leaf size) with early
  stopping on validation MSE. Unlike the baselines, **validation scores here are tuned on validation**,
  so they are mildly optimistic; the test years are the clean number. All three models picked the
  smallest trees (7 leaves, lr 0.03, min leaf 500). Widening the grid improved validation MSE by 0.06%,
  so the choice is not a grid artefact.
- **Over/underfitting.** `results/trees_curves.csv` has train and validation loss per boosting round.
  Validation loss sits below training loss (probably a property of the 2020-2021 period, not checked),
  and stops improving after a few hundred rounds, well before the 3,000-round limit; there is no sign of overfitting.

Validation 2020-2021 (QLIKE, lower is better):

| Model | QLIKE | vs HAR | MSE log-vol | DM p (vs row's control) |
|---|---|---|---|---|
| har | 0.2888 | | 0.1330 | |
| gbt | 0.2744 | -5.0% | 0.1266 | 0.14 vs har |
| har_sector | 0.2685 | -7.0% | 0.1240 | 0.012 vs har |
| gbt_sector | 0.2694 | -6.7% | 0.1243 | 0.005 vs gbt; 0.80 vs har_sector |
| har_corr | 0.2713 | -6.1% | 0.1251 | 0.021 vs har |
| gbt_corr | 0.2693 | -6.7% | 0.1243 | 0.001 vs gbt; 0.61 vs har_corr |

Reading: neighbors help trees too (significantly), but the trees do **not** beat the linear HAR with the
same inputs. Nonlinearity adds little on top of HAR plus neighbors here. The QLIKE ranking between
trees and HAR also depends on the variance correction: on uncorrected `qlike_raw` the neighbor trees are
ahead of the matching HAR by 3-4%, on the corrected number they are level (+0.3% / -0.7%). Plain `gbt` beats
`har` by 5% corrected, but it also has extra inputs (return, volume change, VIX), so the like-for-like
comparisons are `gbt_sector` vs `har_sector` and `gbt_corr` vs `har_corr`. Treat tree-vs-HAR as a tie until
the test years.

## Known limitations (for the report)

- **Survivorship bias.** The universe is the November 2026 S&P 500 list. Firms that left
  the index are missing, including SIVB, FRC, SBNY, ZION and CMA from the March 2023 bank
  failures. FDXF, HONA and VYLR are 2026 spin-offs with no data in the window.
- **Sector labels are from 2026.** GICS reclassifications in 2018 (Communication Services)
  and 2023 (payments companies to Financials) mean early-sample sector labels look slightly ahead.
- **Successor histories.** Some tickers carry a predecessor's prices (e.g. TKO before
  Sept 2023 is WWE). These are kept.

## Layout

- `scripts/`: the data pipeline, features and graphs
- `tests/`: unit tests (`pytest`)
