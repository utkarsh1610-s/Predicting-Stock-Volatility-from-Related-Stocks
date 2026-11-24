# Market Contagion: Predicting Stock Volatility from Related Stocks

CS 6220 project. See `Project_Proposal_Market_Contagion.md` for the research design.

## Setup

Requires Python 3.13. From the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

LightGBM (`scripts/trees.py`) needs the OpenMP runtime. On macOS either `brew install libomp`, or,
without Homebrew, borrow the copy that ships with PyTorch for the session:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/torch/lib
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
| 7 | `scripts/gnnhar.py` | `results/gnnhar*.csv`, `data/predictions/gnn_*.parquet`, `nn_own.parquet` |
| 8 | `scripts/group_analysis.py` | `results/groups*.csv`: neighbor gain by market stress, recent volatility and sector |
| 9 | `scripts/disagreement.py` | `results/disagreement_*_test.csv`: the largest model disagreements, by decile, day and record |
| 10 | `scripts/cold_start.py`, `spillover.py`, `competitor.py` | follow-up tests: new stocks, market-free target, shocks, direct rivals |
| figures | `scripts/figures.py` | `figures/fig01-14_*.png` from the files above |
| clustering | `scripts/stock_clusters.py` | `results/stock_clusters.csv`, `cluster_quality.csv`, `cluster_summary.csv`, `cluster_pairs.csv` |

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

## GNNHAR (`scripts/gnnhar.py`)

One-layer GNNHAR after Zhang et al. (2025): HAR on the stock's own `har_1/5/22` plus one hidden ReLU
layer over [own `har_*`, the neighbors' mean `har_*`]. With mean aggregation a day's graph layer is exactly
that neighbor mean, so the networks see the same neighbor information as the linear and tree models.
`python scripts/gnnhar.py` scores validation; `--include-test` scores the test years (once, at the end).

| Model | Graph input |
|---|---|
| `nn_own` | none: the same network on own history only, to separate nonlinearity from graph information |
| `gnn_sector`, `gnn_corr` | the real graphs |
| `gnn_market` | every other stock (the placebo's market average) |
| `gnn_market+sector`, `gnn_market+corr` | the graph on top of the market average |
| `gnn_rand_sector_s*`, `gnn_rand_corr_s*` | random twins of the graphs (same neighbor counts), one per seed |

- **Training.** QLIKE loss, inputs standardized on the training years, HAR part starting at the QLIKE-trained
  linear fit. Epochs are chosen by early stopping on 2019 (fit on 2010-2018), then the model is refit on all
  training years. Validation is never used to fit or tune, so it stays a clean score.
- **Fixed, untuned architecture:** 16 hidden units, Adam lr 1e-3, batch 8192, at most 30 epochs, patience 3,
  10 seeds averaged. The verdict rule is in the script's docstring and was written before running.
- Runtime about 4 minutes on a laptop CPU. Early stopping picked 1 to 18 epochs.

Validation 2020-2021 (QLIKE, lower is better; forecasts averaged over 10 seeds, each random twin is one graph and one seed):

| Model | QLIKE | MSE log-vol | Linear counterpart (QLIKE) |
|---|---|---|---|
| nn_own | 0.2842 | 0.1448 | har_q 0.2866 |
| gnn_sector | 0.2645 | 0.1373 | har_sector_q 0.2663 |
| gnn_corr | 0.2668 | 0.1381 | har_corr_q 0.2689 |
| gnn_market | 0.2616 | 0.1333 | |
| gnn_market+sector | 0.2609 | 0.1338 | |
| gnn_market+corr | 0.2605 | 0.1337 | |
| gnn_rand_sector (10 twins) | 0.2620 to 0.2645 | | |
| gnn_rand_corr (10 twins) | 0.2649 to 0.2674 | | |

Comparisons (`results/gnnhar.csv`, DM test, negative = first model better):

| Comparison | Change | DM p | Without Feb-Apr 2020 |
|---|---|---|---|
| nn_own vs har_q | -0.83% | 0.15 | -0.17% |
| gnn_sector vs nn_own | -6.93% | 0.006 | -2.72% |
| gnn_corr vs nn_own | -6.12% | 0.010 | -2.12% |
| gnn_market vs nn_own | -7.95% | 0.002 | -3.63% |
| gnn_sector vs har_sector_q | -0.69% | 0.008 | -0.35% |
| gnn_corr vs har_corr_q | -0.77% | 0.013 | -0.34% |
| gnn_sector vs gnn_market | +1.11% | 0.001 | +0.95% |
| gnn_corr vs gnn_market | +1.99% | <0.001 | +1.57% |
| gnn_market+sector vs gnn_market | -0.27% | 0.14 | -0.08% |
| gnn_market+corr vs gnn_market | -0.43% | 0.11 | -0.09% |
| gnn_sector vs its 10 random twins | -0.01% to +0.95% (mean +0.41%) | 0.0008 to 0.95 | +0.17% to +0.59% |
| gnn_corr vs its 10 random twins | -0.22% to +0.72% (mean +0.23%) | 0.045 to 0.92 | -0.05% to +0.27% |

Reading: the neural network repeats the linear placebo. Neighbors help a lot over own history alone, but the
market average helps significantly more than either real graph (by 1.1% and 2.0%), and the real graphs are, if
anything, *worse* than random twins: never better by more than 0.22%, and worse in 16 of 20 comparisons. This holds
even though the comparison favors the real graphs (a 10-seed average against single-seed twins). On top of the market
average the graphs add 0.27% and 0.43%, not significant, under 0.1% without COVID, and slightly worse on MSE. The
pre-written verdict is **market-wide only** for both graphs. Without a graph the network is no better than linear
HAR (p=0.15); with a graph it beats the linear model with the same inputs by under 1%.

Seeds matter: a first run with 3 seeds gave a gain beyond the market of 0.84% and 0.92% (verdict "mixed"). With 10
seeds it fell to 0.27% and 0.43%, so most of that gain was seed noise. Single seeds still differ by up to 2.3% QLIKE
(gnn_market 0.2598 to 0.2657), which is why the average of many seeds is the number to read. Test years untouched.

## Stock clustering (`scripts/stock_clusters.py`)

Clusters stocks by how their volatility moves together **beyond the market**, on the training years only:
each stock's daily log-vol is regressed on the mean of all other stocks that day, and stocks are clustered
(average linkage) on 1 - correlation of those residuals. Compared with GICS sectors by adjusted Rand index
and silhouette (sectors get a silhouette too, as a reference), plus pairs to inspect by hand: correlated
cross-sector pairs that share a cluster, and same-sector pairs the clustering splits.
Runs in about 5 seconds.

Results (471 stocks with at least 500 training days, 2010-2019):

- **Weak structure overall.** Silhouette is 0.04 to 0.06 for every k from 5 to 25, and 0.046 for the GICS sectors
  themselves: neither the data-driven groups nor the sectors separate residual co-movement much. Agreement with the
  sectors is low to moderate, ARI 0.18 to 0.26 (0.25 at k = 11).
- **A few real groups** (k = 11, `results/cluster_summary.csv`), by mean residual correlation inside the cluster:
  Utilities with Real Estate (52 stocks, 0.19), banks and asset managers (40, 0.16), managed care and hospitals
  (CI, CNC, ELV, HUM, HCA and others, 11, 0.15), cable and media (CHTR, CMCSA, DIS, 4, 0.14), Energy with
  materials and agriculture (41, 0.10). The two largest clusters (146 and 80 stocks, 10 sectors each) are catch-alls
  with mean correlation 0.05.
- **Pairs** (`results/cluster_pairs.csv`). Cross-sector pairs that share a cluster are economically sensible: oil
  producers with copper (DVN, COP, APA with FCX, about 0.45), homebuilders with Masco (PHM, LEN, DHI, about 0.42),
  AMZN with META (0.43). Same-sector pairs the clustering splits are negatively correlated after the market, for
  example airlines vs GE (-0.20 to -0.24) and insurers vs Interactive Brokers.
- This matches the same-day clustering finding (DECISIONS.md): co-movement beyond the market is concentrated in
  Utilities, Real Estate and Energy, and small elsewhere; banks also stand out here. Descriptive only; groups named
  after looking.

## Test years 2022-2025 (scored once)

Every model was fit exactly as on validation (trees keep the hyper-parameters chosen on validation;
GNNHAR is fit on the training years, 10 seeds) and scored once on 2022-2025. Files: `results/*_with_test.csv`,
`results/groups_test.csv`. These are the proposal's success numbers.

**Success measure (proposal 4b): met.** Every model family beats its no-neighbor version on the test
years, significantly (DM test on daily mean QLIKE):

| With neighbors vs without | Sector graph | Correlation graph |
|---|---|---|
| Linear HAR (`har_sector`/`har_corr` vs `har`) | -2.72% (p=0.002) | -2.38% (p=0.002) |
| Trees (`gbt_sector`/`gbt_corr` vs `gbt`) | -1.54% (p<0.001) | -1.06% (p<0.001) |
| GNNHAR (`gnn_sector`/`gnn_corr` vs `nn_own`) | -2.89% (p=0.001) | -2.51% (p=0.001) |

QLIKE, test years (lower is better):

| Model | QLIKE | | Model | QLIKE |
|---|---|---|---|---|
| naive | 0.5374 | | nn_own | 0.2957 |
| har | 0.2966 | | gnn_sector | 0.2871 |
| har_sector | 0.2885 | | gnn_corr | 0.2883 |
| har_corr | 0.2895 | | gnn_market | 0.2857 |
| gbt | 0.2903 | | gnn_market+sector | 0.2851 |
| gbt_sector | 0.2859 | | gnn_market+corr | 0.2846 |
| gbt_corr | 0.2873 | | random twins (sector / corr) | 0.2865-0.2872 / 0.2882-0.2890 |

What the test years add to the validation picture:
- **The gain is about 40% of its validation size** (-2.4% to -2.9% vs -6% to -7%): the test years have no
  COVID-scale crash. That matches the validation numbers without Feb-Apr 2020.
- **Still market-wide.** The market average beats both real graphs (sector +0.50% worse, p=0.01;
  correlation +0.89%, p=0.004), and the real graphs equal their random twins (sector +0.09% on average,
  correlation -0.15%; no twin comparison has p<0.17). GNNHAR verdict: **market-wide only** for both graphs.
- **A small, real extra on top of the market.** Graph on top of the market average: -0.23% (sector, p<0.001)
  and -0.38% (correlation, p<0.001). Below the 0.5% line of the verdict rule, but unlike on validation it is
  statistically clear: with four calm years the estimate is precise. It is concentrated in quiet stocks (see below).
- **The fixed sector graph beats the monthly correlation graph** for linear HAR (p=0.03) and trees (p<0.001).
- **Nonlinearity adds little.** Trees vs HAR with the same inputs: -0.9% and -0.8% (p=0.15, 0.26). GNNHAR vs
  the QLIKE-trained linear model with the same inputs: -0.46% and -0.41%. Without a graph, GNNHAR = HAR (p=0.20).

## Per-group analysis (`scripts/group_analysis.py`)

Proposal 4c.1: is the neighbor gain bigger on stressed days, in tightly linked sectors, or only in
already-volatile stocks? Groups use information known on the forecast day: VIX > 25 (stressed) vs <= 25;
the stock's 22-day volatility against its own quartiles from before the scored period; GICS sector.
Each with/without pair is scored on the group's rows, with a DM test per group and Newey-West tests for
"the gain differs between stressed and calm days" and "between top and bottom quartile". The reading rule
(an expectation counts only if two of the three model families meet it with p < 0.05) was fixed on
validation before the test years were scored. `python scripts/group_analysis.py [--split test]`, seconds.

| QLIKE change, with vs without neighbors (sector graph) | Validation 2020-21 | Test 2022-25 |
|---|---|---|
| HAR, calm / stressed days | -2.2% / -15.4% | -1.9% / -7.5% |
| GNNHAR, calm / stressed days | -2.4% / -15.1% | -2.2% / -7.2% |
| Trees, calm / stressed days | -1.9% / -1.7% | -1.6% / -1.3% |
| HAR, quiet / volatile stocks (bottom / top quartile) | -2.7% / -9.7% | -1.9% / -3.4% |
| Stressed days in the period | 175 of 505 | 157 of 1003 |

- **(1) Bigger gain on stressed days: not supported by the rule.** For HAR and GNNHAR the gain on stressed days
  is 3 to 7 times the calm-day gain, but the difference is not significant (validation p=0.05 to 0.07, test
  p=0.3 to 0.45): stressed days come in a few long episodes (COVID, 2022), so there are few independent ones.
  The trees show no stress pattern at all; they already get VIX as an input, which is the market-level
  information the neighbors carry for the other models. The market-only network shows the same stress
  pattern as the graphs (-16.3% / -7.5% on stressed days).
- **(3) Quiet stocks gain too: supported** on both splits, in all three families (test: -1.4% to -2.4% in the
  bottom quartile, mostly significant). Volatile stocks gain more, but not significantly more.
  The small graph-beyond-market extra is the exception: it is larger for **quiet** stocks (-0.48% / -0.84% vs
  -0.20% / -0.29% in the top quartile, difference p<0.002 on test).
- **(2) Bigger gain in banks and chip makers: not supported.** Financials and Real Estate do gain most on test
  (-3.8% and -3.9% for HAR with sector peers, vs -2.7% overall), but Information Technology gains least
  (-1.6%, not significant), and Energy is not significant either. Validation is similar: Utilities, Real Estate
  and Financials on top, Information Technology near the bottom. Sector results are many tests; few clear the
  Bonferroni line (marked `significant_bonferroni` in the CSV).

## Largest disagreements (`scripts/disagreement.py`)

Proposal 4c.2: on the stock-days where HAR with sector peers (`har_sector`) and plain HAR disagree most, who
was right, and what did the peers do? Test years, both forecasts with their smearing correction.
`python scripts/disagreement.py [--split val]`, seconds. Descriptive.

- **The gain comes from the extremes.** Split the 496,188 test stock-days into ten equal bins by disagreement:
  the peer model cuts QLIKE by 6.4% in the bin where peers say "lower" most and by 11.8% where they say
  "higher" most; in the middle six bins the change is 0% to -1% (`results/disagreement_deciles_test.csv`).
  So the neighbor gain comes from about a fifth of stock-days.
- **"Higher" is an insurance call.** When peers say lower, the peer model is closer 69-72% of the time. When
  they say higher, it is closer only 33-39% of the time, yet that bin has the biggest QLIKE gain: QLIKE
  punishes under-predicting risk, and the rare real spikes it catches outweigh the many small misses.
- **The big disagreements are market days, not sector stories** (`results/disagreement_days_test.csv`). The
  busiest days are the April 2025 tariff sell-off: on 4, 7 and 8 April the peer model was closer on 91-100%
  of large disagreements, but on 9 and 10 April, after the rebound, only 44% and 8%; the peers kept calling
  for high volatility after it had passed. Next come January 2022 (tech sell-off), October 2022, August 2024
  and the day after the 2024 election. In the March 2023 bank crisis (13 and 16 March) the disagreements
  were mostly Financials, but the peer model was closer on only 28-29% of them.
- **Quiet days too.** The largest "peers say lower" days are holiday sessions (day after Thanksgiving
  2022 and 2023, late December), where the peer model was closer on 63-95%.
- In the top records (`results/disagreement_top_test.csv`) the "peers moving" are a sector's biggest movers
  that day, and on the busiest days they are the same for every stock in the sector: the peers stand in for
  the market. This matches the placebo and GNNHAR findings.

## Follow-up: which links carry contagion? (`cold_start.py`, `spillover.py`, `competitor.py`)

Three scripts, each with its decision rule written in its docstring before it was run. All fit on 2010-2019
and report validation (2020-21) and test (2022-25). Volatility differences are log-vol differences turned
into percent.

| Script | Question | Output |
|---|---|---|
| `scripts/cold_start.py` | Can a stock with little or no history of its own borrow its sector's volatility? | `results/cold_start*.csv` |
| `scripts/spillover.py` | With the market removed, do sector peers predict a stock's volatility? When one company has its own shock, do its sector peers follow the next day? | `results/excess_vol*.csv`, `results/shock_*.csv` |
| `scripts/competitor.py` | Does the single closest competitor carry information the whole sector does not? | `results/competitor*.csv` |

**Story in one line:** the whole market drives volatility; contagion between companies travels through direct
rivals when shocks hit, mostly on the same day; the sector is useful only as a starting guess for a new stock.

**Ordinary days (stock with full history): market first, everything else ~0.**
- With the market removed from the target (a stock's log-vol minus the market's), its own history explains
  43% of the variation; sector peers add 0.05% (MSE -0.08%, p=0.01 validation; -0.02%, p=0.33 test). Rule
  (>1% on both splits) not met. Random groups add nothing; the small gain is the fixed sector label.
- A single competitor (closest rival, rival of similar size, industry leader, or top correlation neighbor)
  adds ~0.05% over the market, the same as the sector average.

**Shock days: the direct rival reacts, the sector barely does.** A shock is a stock's own jump (log-vol minus
its 22-day mean, minus the market's) in the top 1% of training days while the market is calm (1,281 shocks in
validation, 4,031 in test). Exposed stocks are compared with unexposed stocks on the same day.

| Linked company had a shock | Same day, val / test | Next day beyond HAR + market, val / test |
|---|---|---|
| Any sector peer | +2.7% / +4.0% (p<0.001) | +0.1% (p=0.85) / +0.3% (p=0.28) |
| Closest rival (same sub-industry, most similar market-free returns) | +17% / +23% (p<0.001) | +3.9% (p=0.009) / +1.5% (p=0.07) |
| Industry leader (largest in the sub-industry) | +17% / +19% (p<0.001) | **+4.2% (p=0.02) / +2.5% (p=0.01)** |
| Random stock (5 draws) | ~0 (p>0.6) | ~0 (p>0.6) |

- A rival's shock moves a stock about five times more on the same day than a typical sector peer's: averaging
  over 20-70 peers washes the signal out.
- The industry leader is the only link with a next-day effect on both splits, beyond the stock's own history
  and the market. With 4 link types tested it misses a Bonferroni cut (0.0125) narrowly, so it is consistent
  but borderline. Sector-wide exposure has no next-day effect: its same-day news is already in each peer's
  own volatility by the close.
- Two share classes of one company (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) are never linked. Example pairs (latest
  month): DAL-UAL, HD-LOW, KO-PEP, XOM-CVX, UNH-ELV, WMT-COST, JPM-BAC.

**New stocks: the sector on day 1, the closest rival once it has traded.** Each stock is given only its last
k days of its own volatility (peers keep full history).

| Own history | Sector vs market only, val / test | Live sector vs fixed sector label |
|---|---|---|
| 0 days | **-8.0% / -5.5%** (p<0.001) | -0.9% (p=0.005) / -0.06% (p=0.72) |
| 1 day | -2.4% / -1.6% | -0.5% / -0.2% (p<0.01) |
| 5 days | -0.35% / -0.3% | ~0 |
| 22 days (full HAR) | -0.06% / -0.08% | ~0 |

- Random groups of the sector's size add nothing at any k, so for a new stock the sector beats random for
  the first time in the project. But a fixed label per sector gets almost all of it: what helps is knowing
  the sector's usual level (utilities calm, energy wild), not its daily news. The rule (live sector beats
  the label on both splits) is not met at k=0. The gain is gone after about 5 days of own history.
- With no own history, the closest rival (-15.7% / -10.4%) and the top correlation neighbor (-17.3% / -17.5%)
  beat the sector average, but choosing them needs ~60 days of the stock's own returns. The industry leader,
  the only rival usable on day 1, is worse than the sector average (-6.0% / +0.5%).
- Real new listings (83 stocks first seen after March 2010, their first 60 rows): with no own history the
  sector beats the market by 9.8%; once their own 22-day HAR exists the sector adds nothing (+0.25%). A row
  needs 22 days of history, so listing days 1-21 can only be studied by the simulation above.

**What this adds to the reference papers.** Zhang et al. and Kumar et al. credit their gains to the stock
graph. With a market-average input and random graphs (controls neither paper has) most of that gain is the
market. What is specific to links sits in single direct rivals on shock days, which a graph that averages
many neighbors dilutes.

## Figures (`scripts/figures.py`)

`python scripts/figures.py` redraws every figure from files the pipeline already wrote (seconds, nothing is
refit) into `figures/`, at 1920 x 1080 for slides and video. In story order:

| File | Shows |
|---|---|
| `fig01_volatility_timeline.png` | 16 years of volatility with the train / validation / test periods and the big episodes |
| `fig02_graphs.png` | The sector graph vs the correlation graph for 30 random stocks (March 2023) |
| `fig03_main_result.png` | Neighbors improve every model, validation and test (the success measure) |
| `fig04_placebo.png` | Real neighbors vs random twins vs the whole market (GNNHAR) |
| `fig05_seed_check.png` | 3 vs 10 training runs, and how much single runs differ |
| `fig06_learning_curves.png` | Training vs validation error of the trees (overfitting check) |
| `fig07_calm_vs_stressed.png` | Gain on calm vs stressed days (VIX > 25) |
| `fig08_sector_heatmap.png` | Gain by sector and model, test years |
| `fig09_disagreement.png` | The gain comes from the days the models disagree most |
| `fig10_clustering.png` | Market-residual correlation matrix ordered by clustering, with sectors |
| `fig11_april_2025.png` | Case: the April 2025 tariff sell-off, actual vs both forecasts |
| `fig12_new_stock.png` | A new stock: gain from the sector vs days of its own history, live average vs fixed label vs random |
| `fig13_shock_rival_vs_sector.png` | After a company's own shock: sector peers vs closest rival vs industry leader vs random, same day and next day |
| `fig14_new_stock_stand_in.png` | Who a new stock should copy: sector, industry leader, closest rival, top correlation neighbor |

## Known limitations (for the report)

- **Survivorship bias.** The universe is the November 2026 S&P 500 list. Firms that left
  the index are missing, including SIVB, FRC, SBNY, ZION and CMA from the March 2023 bank
  failures. FDXF, HONA and VYLR are 2026 spin-offs with no data in the window.
- **Sector labels are from 2026.** GICS reclassifications in 2018 (Communication Services)
  and 2023 (payments companies to Financials) mean early-sample sector labels look slightly ahead.
- **Successor histories.** Some tickers carry a predecessor's prices (e.g. TKO before
  Sept 2023 is WWE). These are kept.

## Layout

- `scripts/`: the data pipeline, features, graphs, models (HAR, trees, GNNHAR) and analyses
- `tests/`: unit tests (`pytest`)
