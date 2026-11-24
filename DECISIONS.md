# Decision log

Choices that could reasonably have gone another way. For each: who set it, what the alternatives are,
and whether we have checked that it matters. Add a row whenever a new one is made.

**Set by**: `proposal` = in the project proposal; `team` = pre-existing pipeline code (steps 1-2);
`claude` = chosen while building steps 3-6, open to challenge;
`utkarsh` = chosen while building GNNHAR (step 7) and the stock clustering, open to challenge.
**Checked?**: `yes` = sensitivity tested; `partly`; `no` = never tested, so we do not know if it matters.

## Data and cleaning (steps 1-2)

| # | Decision | Set by | Alternatives | Checked? |
|---|---|---|---|---|
| 1 | Universe is the **current (Nov 2026) S&P 500 list** | proposal | Historical index membership (removes survivorship bias; much more work) | no. Known limitation: SIVB, FRC etc. are missing |
| 2 | Data is a **frozen snapshot**, never re-downloaded | team | Refresh periodically (but Yahoo rescales history, so results would shift) | n/a |
| 3 | **`auto_adjust=True`** prices | team | Raw prices | no |
| 4 | Drop rows with zero volume, zero high-low range, bad ticks, non-positive prices (32 rows) | team | Keep and repair | partly (counts are tiny) |
| 5 | 7 tickers start at their "real" US listing date (9,233 rows dropped) | team | Keep stitched history | partly (checked against volume) |
| 6 | Minimum **126 clean rows** per ticker (drops only Q) | team | Other cutoff | no, but only 1 ticker affected |
| 7 | Days with high/low > 1.5 are **kept** (41 days) | team | Drop them | no |

## Features and target (step 3)

| # | Decision | Set by | Alternatives | Checked? |
|---|---|---|---|---|
| 8 | Volatility measure is **Garman-Klass** | proposal | Realized vol from intraday data (not free), Parkinson, close-to-close | no |
| 9 | Model the **log** of volatility | claude | Model volatility or variance directly | no (standard in the HAR literature, but not tested here) |
| 10 | **Winsorize** daily log-vol at the 0.1% / 99.9% quantiles, bounds from training years only; **the target is the clipped value** | claude (README suggested winsorizing; the quantiles are mine) | Other quantiles, no clipping, clip only inputs | **partly** (sweep): the headline neighbor gain moves with clipping (sector: 1/99% -5.2%, default -7.0%, none -8.0%) but the ex-COVID gain is stable (-2.0% to -2.6%). Absolute losses are on different targets, so levels are not comparable |
| 11 | HAR windows **1, 5, 22** days | proposal | Other windows | no |
| 12 | Train 2010-19 / val 2020-21 / test 2022-25 | proposal | Rolling or expanding windows | n/a |
| 13 | Rows are assigned to a split by the **date of the label**, not the inputs | claude | Assign by input date (leaks one day of validation into training) | yes (tested; the alternative is a planted-bug test) |
| 14 | A missing trading day **drops** every row whose window or label touches it | claude | Fill the gap, or allow shorter windows | no (cost is 554 rows) |
| 15 | Extra tree inputs: `ret`, `dlogvol` (log change in volume vs yesterday), same-day `vix` | proposal names them; exact definitions are claude | e.g. absolute return, volume vs 22-day average | no |

## Graphs (step 4)

| # | Decision | Set by | Alternatives | Checked? |
|---|---|---|---|---|
| 16 | Correlation graph rebuilt **monthly** | proposal | Weekly, quarterly, daily | **yes**: weekly and quarterly change the gain by 0.1% or less |
| 17 | Correlation window **60 trading days** | proposal | 30, 120, 250 days. Longer is more stable, shorter reacts faster; I have not seen evidence for either here | **yes**: windows of 30 to 250 days change the gain by 0.3% or less (30 days is 0.3% better, p=0.004, far under the 1% rule) |
| 18 | **10 neighbors** per stock | proposal | 5, 20, or a correlation threshold | **yes**: more neighbors help steadily: K=5 gives -5.1%, 10 gives -6.1%, 20 gives -6.8%. K=20 is 0.8% better than the default (p=0.002), under the 1% rule margin. Not tested beyond 20 |
| 19 | Correlation computed on **log returns** (close to close) | claude | Volatility series, absolute returns. NB: neighbors are chosen by return correlation but then used to predict volatility | no |
| 20 | **Pairwise-complete** correlation, needing **at least 40 shared days** of the 60 | claude (Option A chosen with you); the 40 is mine | Require all 60 days (Option B); another minimum | **yes**: 30 and 50 change nothing (0.1% or less). Option B itself not tested |
| 21 | Neighbor features are an **equal-weight mean** of neighbors' `har_*` | proposal says "averaged" | Weight by correlation, use the max, use a median | **yes** for correlation weighting: no change (0.0%). Others untested |
| 22 | Sector graph links **all** same-sector stocks (20-80 peers, equal weight) | proposal | Sub-industry (tighter groups) | **yes**: sub-industry is much worse (-3.5% vs -7.0% for sector), consistent with more neighbors helping |
| 23 | Sector labels are the **2026** labels throughout | proposal | Historical GICS | no. Known limitation |
| 24 | Models only compare on rows where **both** neighbor feature sets exist (1% dropped) | claude | Let each model use all its rows | no |

## Models and evaluation (steps 5-6)

| # | Decision | Set by | Alternatives | Checked? |
|---|---|---|---|---|
| 25 | HAR is **one pooled regression across all stocks** | claude | One regression per stock (Corsi's original form) | **yes**: per-stock HAR is 2.6% worse in absolute QLIKE (0.2961 vs 0.2888). It makes the neighbor gain look bigger (-8.3%) only because the baseline is weaker, so pooled is the stricter baseline |
| 26 | QLIKE and MSE-on-log-vol as metrics | proposal | RMSE on volatility, MAE, etc. | n/a |
| 27 | QLIKE forecast variance is rescaled by a **training-set smearing factor** (~1.37) | claude | No correction, or fit with a QLIKE loss | **yes, and it matters**: without it trees lead HAR by ~3% on QLIKE; with it they tie |
| 28 | Diebold-Mariano on each day's **cross-sectional mean** loss, Newey-West variance with lag floor(T^(1/3)) | claude (DM test itself is proposal) | Panel DM with clustered errors; stock-level tests | no. Treats all stocks as equal and ignores that stocks move together within a day beyond the averaging |
| 29 | Fit on **train only**; never refit on train+val for the test run | claude | Refit on train+val before scoring test | no |
| 30 | **Test years untouched** until a final run with all models | proposal | n/a | n/a |
| 31 | Trees: LightGBM, **L2 loss**, small 12-config grid, early stopping on val | claude | Tune QLIKE loss directly, bigger grid, XGBoost | partly (a wider grid gained 0.06%) |
| 32 | Validation scores for trees are **tuned on validation** | claude | Nested tuning | n/a. Flagged in the README |
| 45 | Sensitivity rule: a variant replaces a default only if it is >1% better on rows both score, DM p<0.05, and still >0.5% better without Feb-Apr 2020. Written before the sweep ran | claude | A different margin; correcting for the ~30 comparisons | n/a. No variant met it, so all defaults stay |
| 47 | Placebo design: market feature = same-day mean `har_*` of all other stocks; random twins keep each stock's neighbor count (and, for the correlation graph, its monthly availability); 5 seeds; linear HAR only. Reading rule written beforehand | claude | Nonlinear placebo (trees / GNNHAR with a market input); volatility-weighted or sector-neutral market averages | n/a. See the finding below |
| 48 | Tree placebo uses ONE fixed hyper-parameter setting (7 leaves, lr 0.03, 500 rows per leaf) for all models, 3 seeds for the random twins | claude | Tune each model separately (more optimism, tuning noise in the comparison) | partly: all earlier tuning runs chose this setting and a wider grid gained 0.06% |
| 46 | When varying the window, `min_common` scales with it (2/3, as 40 of 60) | claude | A fixed minimum | no |
| 49 | **Done (PR #10):** a QLIKE-trained HAR baseline (`har_q`, plus sector and correlation versions) alongside the OLS HAR | claude, from Zhang et al. (their HAR_Q beats OLS HAR by 1.9% QLIKE and 7.3% MSE at 1 day, so their headline gain over OLS HAR overstates the gain over a fair baseline) | Keep OLS HAR plus the smearing correction (#27) only | **yes**: `har_q` is only 0.8% better than OLS+smearing (QLIKE 0.2866 vs 0.2888, p=0.015); neighbor gains unchanged on top of it (-7.8% sector, -6.9% correlation). Smearing was a good approximation for linear models. Trees (L2 loss, smeared) are now 1.2% worse than `har_sector_q` and level with `har_corr_q`; a QLIKE objective for trees is untried |
| 50 | **Built (`scripts/gnnhar.py`), not yet run on the data:** GNNHAR with **one graph layer** only | claude, from Zhang et al. (2- and 3-hop add no clear value; 3 layers over-smooth; on S&P 100, 4-5 layers are worse) | 2-3 layers | n/a |
| 51 | **Built, not yet run:** run GNNHAR on the real graph, a random twin graph and the market (all-stocks) graph | claude | Run the real graph only, as the paper does | no. Neither reference paper has this control |
| 52 | **Planned:** consider the **Model Confidence Set** (as in Zhang et al.) for multi-model comparisons, in place of or alongside pairwise DM tests (#28, #40) | claude | Keep pairwise DM, correct for multiple comparisons | no (not built yet) |
| 53 | GNNHAR has no per-stock identity embedding: its hidden values are computed each day from that day's inputs. "Cluster the learned embeddings" (#43) therefore needs a definition. **Decided:** cluster stocks on the correlation of their **market-residual** daily log-vol over the training years (`scripts/stock_clusters.py`), the co-movement the graphs are meant to carry | utkarsh | Average each stock's hidden state over days; cluster averaged HAR features | no |
| 54 | Per-sector check: each sector gets its own QLIKE-trained regression; flag a sector/graph only if market+graph beats market by >1% with DM p<0.002 (about 0.05/22 tests) and >0.5% without Feb-Apr 2020. Written before running | claude | Pooled fit only (what the placebo did); a looser p | n/a. See the finding below |
| 55 | Own-history test: without own history, the stock's mean log-vol over the TRAINING years replaces it (a constant per stock); QLIKE-trained linear models; pooled verdict rule written beforehand; per-sector results are hypothesis-generating (44 tests, p<0.001 and >1%) | claude | A different stand-in for "no own history" (e.g. a short history window); nonlinear models | n/a. See the finding below |
| 56 | Same-day clustering analysis is restricted to 2010-2021 by construction, uses the mean log-vol of other stocks as the market factor (own series excluded), and the sector "excess" is the peer mean minus the market factor | claude | Include the test years (they are reserved); a volatility-weighted market factor | no. Exploratory, no decision rule |
| 57 | GNNHAR aggregates neighbors by their **mean** (row-normalized adjacency), so a day's graph layer equals a hidden layer over [own `har_*`, neighbor-mean `har_*`] | utkarsh | The paper's symmetric normalization D^-1/2 A D^-1/2 (identical for the sector graph, whose sectors are cliques; close for the correlation graph, out-degree 10) | no |
| 58 | GNNHAR adds an **`nn_own` control**: the same network with no graph input, to separate nonlinearity in a stock's own history from information that comes through the graph | utkarsh | Compare only with linear HAR, as the paper does | n/a. Not in either reference paper |
| 59 | GNNHAR trains on **QLIKE** with inputs standardized on training years; HAR part starts at the QLIKE-trained linear fit; epochs chosen by early stopping on **2019** (fit on 2010-2018), then refit on all training years. Validation is never used to fit or tune | utkarsh | Early-stop on validation (as the trees tune on it); MSE loss plus smearing | no |
| 60 | GNNHAR architecture is fixed and untuned: 16 hidden units, Adam lr 1e-3, batch 8192, at most 30 epochs, patience 3; 3 seeds, forecast = mean of the seeds' log-vol forecasts | utkarsh | Tune on 2019; more seeds | no |
| 61 | GNNHAR verdict rule (in the script's docstring, written before running): graph-specific only if gnn_G beats every random twin by >1% QLIKE with DM p<0.05 AND gnn_market+G beats gnn_market by >0.5% with p<0.05 and still better without Feb-Apr 2020; market-wide only if gnn_market+G improves on gnn_market by <0.5% | utkarsh | Same rule as the linear placebo (#47) with its 1 pp market condition | n/a |
| 62 | Stock clustering: average linkage on 1 - residual correlation, k = 5..25 with k = 11 (number of GICS sectors) as the main comparison; stocks need 500 training days and pairs 250 shared days (pairs with less count as uncorrelated) | utkarsh | Ward or spectral clustering; choose k by silhouette; drop pairs with little overlap | no. Descriptive, no decision rule |

## Research design (all from the proposal)

Proposal-owned choices are logged on the same terms as ours: they can be argued and have not been tested.

| # | Decision | Set by | Alternatives | Checked? |
|---|---|---|---|---|
| 33 | Predict **volatility, not price direction** | proposal | Direction, returns, or both. The proposal's reason (direction is near a coin flip; volatility clusters) is sound but not tested here | n/a |
| 34 | **Horizon is 1 trading day** ahead (5-day is out of scope) | proposal | 5 or 22 days. Spillovers could look different at longer horizons | no. Zhang et al. Table 1: the nonlinear GNN's QLIKE gain over a QLIKE-trained HAR is 2.0% at 1 day and 3.4% at 1 week, and gone at 1 month. So the optional 5-day horizon looks at least as promising |
| 35 | **Daily** OHLC data, free sources only | proposal | Intraday realized volatility (the GNNHAR paper's data). The proposal expects smaller gains because of this | n/a |
| 36 | Sample is **2010-2025**, S&P 500 stocks | proposal | Longer history, other markets or indices | no |
| 37 | Yahoo Finance via `yfinance` as the price source | proposal | Stooq (named as backup), paid data | no |
| 38 | Baseline set: **naive, HAR, LightGBM, GNNHAR**; GARCH and LSTM are out of scope | proposal | Add GARCH (the textbook volatility baseline) | no |
| 39 | Neighbor information is added as **averaged features** for HAR and trees, and via message passing for GNNHAR | proposal | Other ways to inject neighbors (e.g. attention, cross-stock regression) | no |
| 40 | **Success** = neighbor models beat their no-neighbor versions on test years, with statistical significance. No significance level is stated | proposal (the level is open; I have been reading p < 0.05) | State the level and a correction for the many comparisons we run | no. We run ~10 DM tests per split without adjusting |
| 41 | Stress is defined as **VIX > 25** (calm: VIX <= 25) | proposal | Another threshold, or VIX terciles | no |
| 42 | "Recent volatility" groups are the **top vs bottom quartile** of each stock's past-month volatility | proposal | Terciles, deciles | no |
| 43 | Second task: **cluster GNNHAR's learned embeddings** and compare to sectors (silhouette, adjusted Rand index) | proposal | Cluster raw correlations or HAR features instead of embeddings | no |
| 44 | Case study is the **March 2023 bank failures** | proposal | Another stress episode (COVID, 2022 bear market). SIVB itself is not in our data | no |

**Per-sector finding (`results/sector_check.csv`, validation; PR #10).** No sector shows a graph effect beyond the market:
0 of 22 sector/graph pairs met the rule. Adding sector peers or correlation neighbors on top of the market average
moves QLIKE by between -0.64% and +0.54% in every sector, mostly slightly *worse* (extra parameters). The market
average itself helps every sector, by -7.3% (Information Technology) to -12.0% (Utilities), so the effect is universal and
not driven by a few sectors. Three of 22 raw p-values fell below 0.05 (one in the wrong direction), about what chance
gives. This answers the heterogeneity worry about the pooled placebo: sector-by-sector models change nothing.

**Same-day clustering finding (`results/sector_clustering.csv`, 2010-2021; PR #11).** Volatility does cluster by sector beyond
the market, modestly overall and strongly in three sectors. The market factor explains 39.5% of a stock's daily log-volatility
variation; the sector excess adds 4.5 points overall, but +17.8 in Utilities, +11.5 in Energy and +10.5 in Real Estate (+1 to +4
elsewhere). After removing the market, same-sector residual correlation is +0.074 vs -0.010 for different sectors (Utilities +0.32,
Energy +0.23, Real Estate +0.21). Sector excess is persistent (autocorrelation 0.46 at 1 day, 0.28 at 22 days). Largest sector-vs-market
gaps: Energy 2014-2016 and April 2020 (1.6x to 1.85x the market), Utilities calm in 2011 and 2012 (about 0.6x). The dates fit the
oil price collapse; that link is an inference, not something the data shows. Utilities, Energy and Real Estate were named after
looking, so tests on them are hypotheses.

**Own-history finding (`results/own_history.csv`, validation; PR #11).** The sector information is real but redundant given the
stock's own recent volatility. Pooled, without own history: sector peers beyond the market -2.00% QLIKE (p=4e-25), correlation neighbors
-6.30% (p=4e-27). With own history: -0.06% (p=0.06) and -0.10% (p=0.14). Own history itself is worth 21.9%. The pre-registered verdict
is "redundant" for both graphs. Per sector, without own history: Utilities -16.6%, Energy -12.2%, Real Estate -7.4% (sector peers); with it,
+0.30%, +0.36%, -0.43%. So the sector helps exactly where it clusters most, and only until the stock's own history is known. The
correlation graph is worth far more than the sector graph without own history; untested guess: correlated stocks stand in for the stock's
own current level. A possible use this points to, untested: stocks with little or no history.

## What the two reference papers do and do not tell us

Read by a delegated agent, then checked against pasted text from paper 1 (Table 1, Eq. 7, Section 5.1, Section 6.2, Conclusion). Paper 2 could only be read as an arXiv preprint; its published version was blocked.

- **Zhang et al. (GNNHAR), arXiv v1, May 2023:** 27 Dow stocks, intraday data, graph from Graphical LASSO on returns. Headline gain at 1 day is 13% (MSE) and 4% (QLIKE) over OLS HAR, but 6.5% and 2.0% over the QLIKE-trained HAR. Gains hold to 1 week (QLIKE vs HAR_Q: -2.0% at 1 day, -3.4% at 1 week) and vanish at 1 month. GNNHAR's layer is a graph convolution with fixed normalized weights (Eq. 7). The paper has **no market-volatility term, no random-graph control, no comparison of graph types, and no nonlinear model without a graph**, so its "nonlinear spillover" gain is mixed up with plain nonlinearity in a stock's own volatility.
- **Kumar et al. (preprint of Mathematics 14(2), 289):** 8 stock indices, daily, a temporal graph attention network, correlation vs Diebold-Yilmaz graphs. No significance tests, no market input, no random-graph control, internal inconsistencies in the tables. The abstract of the published version (seen only in a search summary, unverified) reports no systematic gain from dynamic over fixed graphs, which matches our refresh-rate sweep.
- **Neither paper answers whether sector information helps beyond the market level.** Our placebo (`results/placebo*.csv`) is the only evidence so far.

## What I would test first, if time allows
Done in the sensitivity sweep (`results/sensitivity.csv`, PR #8): #10, #16-18, #20-22, #25. None changed the picture enough to replace a default.

**Placebo finding (`results/placebo.csv`, validation, linear HAR; PR #9).** The neighbor gain is a market-wide
volatility level, not specific links. A feature averaging *all* stocks gains -8.7% QLIKE vs plain HAR, more than
the sector graph (-7.0%) or the correlation graph (-6.1%). Random twins of the graphs with the same neighbor counts
gain -8.0% and -6.7%, i.e. the real graphs are 1.1% and 0.7% *worse* than random. Adding either graph on top of the
market average improves QLIKE by 0.06%. The same holds without Feb-Apr 2020. One wording note: the rule said the
market had to be "within 1.0 pp" of the graphs, but it beat them by 1.7 to 2.6 pp. The rule did not anticipate that;
the other two conditions point the same way, so the verdict does not depend on it. Limits: linear model only, and an
equal-weight average of 10 specific stocks is a noisier market proxy than 75 or 499, which may explain why the
random and market versions win.

**Placebo finding for trees (`results/placebo_trees.csv`).** Same verdict, noisier. Under the pre-written rule no
graph-specific structure exists for trees: market+graph beats market by only 0.56% (sector, p=0.18) and 0.75%
(correlation, p=0.06), and no real graph beats its random twin by more than 1% with p<0.05 in every seed (sector
-1.1% on average but max p=0.42; correlation -0.1%). Without Feb-Apr 2020, every neighbor-style feature gives about
-2% (real sector -1.95%, real correlation -1.50%, market -2.12%, random twins -2.2% and -2.1%), so the real graphs
are no better than a market average or random stocks. Over the full validation period the tree numbers are erratic
(the market feature alone is +0.47% *worse* than none, p=0.84, while -2.1% without COVID). One untested guess:
trees cannot extrapolate, and March 2020 is far outside the 2010-2019 training range.

Still open, by how much each could change the headline result:
- **Cold start:** do peers help when a stock has little or no history (new listings)? The own-history test points there; not run.
- **Whether GNNHAR finds structure that our averaged-neighbor trees did not.** Correction to an earlier version of this line: GNNHAR does not learn per-neighbor weights. Its layer is a graph convolution, `ReLU(D^-1/2 A D^-1/2 H Theta)`, with a fixed degree-normalized adjacency and one shared transform of each stock's 1/5/22-day volatilities (Zhang et al., Eq. 7). Its neighbor information is close to what the tree placebo already saw, so a null result is the expectation. Test: real graph vs random twin vs market graph.
- **Trees and the COVID extrapolation guess** (above).
- **Specific episodes** where links should matter more than the market (sector shocks, the March 2023 banks, which sits in the test years). Sector-by-sector averages over 2020-2021 found nothing.
- #40: how many tests we run versus how we call something significant.
- #34: the 1-day horizon.
- #27: already known to matter, so decide on one primary QLIKE.
- #18: K beyond 20.
