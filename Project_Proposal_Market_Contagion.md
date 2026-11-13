# Market Contagion: Predicting Stock Volatility from Related Stocks

**CS 6220 Data Mining · Project Proposal**

> **In one sentence:** We predict how volatile each S&P 500 stock will be tomorrow, and test whether looking at *related* stocks makes that prediction better than looking at the stock alone.

---

## Contents

1. [Team Members](#1-team-members)
2. [Code Repository](#2-code-repository)
3. [Dataset](#3-dataset)
4. [The Data Mining Problem](#4-the-data-mining-problem)
   - 4a. [Type of Problem](#4a-type-of-problem)
   - 4b. [How We Measure Success](#4b-how-we-measure-success)
   - 4c. [How We Check the Results Are Meaningful](#4c-how-we-check-the-results-are-meaningful)
5. [Scope and Work Split](#5-scope-and-work-split)
6. [References](#6-references)

---

## 1. Team Members

| Role | Name |
| :--- | :--- |
| Corresponding author | [Name 1] |
| Member | [Name 2] |

---

## 2. Code Repository

**Link:** [github.com/&lt;team&gt;/market-contagion](https://github.com/)

- Private repository; access granted to the instructor and TAs only.
- All model code is written by us.
- Anything borrowed, including ideas from the GNNHAR paper, is cited in the report.

---

## 3. Dataset

### 3.1 Overview

We use daily price data for about **500 S&P 500 stocks from 2010 to 2025**, plus the VIX index and official sector labels. All sources are free. We will download and save everything locally in week 1.

### 3.2 Sources

| Source | Content | Size |
| :--- | :--- | :--- |
| Yahoo Finance (`yfinance` package); backup: Stooq | Daily open, high, low, close, volume per stock | ~2 million rows (≈ 500 stocks × 4,000 trading days), 7 attributes |
| Yahoo Finance | VIX (market fear index), daily close | ~4,000 rows |
| Wikipedia: List of S&P 500 companies | Ticker, company name, GICS sector and sub-industry | ~500 rows |

### 3.3 Format and Derived Target

- Stored as one CSV / Parquet table: **one row per stock per trading day**.
- From each row we compute the stock's **daily volatility** with the **Garman-Klass estimator**, which uses the day's open, high, low and close.
- A day with a wide high-to-low range gets a high volatility value.

### 3.4 Sample Records

*Illustrative of the format; replaced with rows from our download before submission.*

| date | ticker | open | high | low | close | volume | sector |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| 2023-03-09 | SIVB | 267.83 | 268.00 | 100.00 | 106.04 | 26,067,000 | Financials |
| 2023-03-09 | JPM | 138.10 | 138.95 | 133.21 | 133.89 | 19,240,000 | Financials |
| 2023-03-09 | AAPL | 153.56 | 154.54 | 150.23 | 150.59 | 53,833,600 | Information Technology |

### 3.5 Known Limitation

We use the **current** S&P 500 member list, so companies that dropped out of the index are missing (**survivorship bias**). We state this clearly in the report rather than rebuild historical member lists.

---

## 4. The Data Mining Problem

### 4a. Type of Problem

#### Main task: Prediction (regression)

Predict each stock's **volatility for the next trading day**.

We predict volatility rather than price direction because:

- Price direction is close to a coin flip.
- Volatility clusters: calm days follow calm days, and wild days follow wild days.

#### The question we test

> Does information from **related stocks** improve the forecast beyond the stock's own history?

*Example:* if Nvidia drops 8% today, AMD may swing more tomorrow even with no news of its own. A model that only looks at AMD's past cannot see this.

#### How we link related stocks

| Graph | How stocks are linked | Changes over time? |
| :--- | :--- | :--- |
| **Sector graph** | Stocks in the same GICS sector | No (fixed) |
| **Correlation graph** | Each stock links to the 10 stocks whose returns moved most like it over the past 60 trading days | Yes, rebuilt monthly from past data only |

Comparing the two graphs answers a question recent papers disagree on: **does a graph that changes over time help, or is a fixed one enough?**

#### Models

Each model (except the naive baseline) is compared **without and with neighbor information**, so the only difference within a pair is whether the model sees related stocks.

| # | Model | Course topic | Without neighbors | With neighbors |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Naive:** tomorrow = today | Baseline | ✓ | — |
| 2 | **HAR:** linear regression on the stock's average volatility over the last 1, 5 and 22 days | Linear regression (L4) | ✓ | Adds neighbors' averaged 1/5/22-day volatility |
| 3 | **Gradient-boosted trees** (LightGBM / XGBoost): HAR features + returns, volume change, VIX | Trees (L5) | ✓ | Same neighbor features added |
| 4 | **GNNHAR:** HAR + a small graph neural network over neighbors, re-implemented from Zhang et al. (2025) | Neural networks (L7) | HAR is its no-neighbor version | ✓ once per graph |

#### Second task: Clustering

Cluster the **stock embeddings** that GNNHAR learns and compare the clusters with official sectors. The goal is to find groups of stocks that behave alike but sit in different official sectors.

---

### 4b. How We Measure Success

**Success** = models with neighbor information beat their no-neighbor versions on the test years, and the gain is statistically significant.

#### Train / validation / test split (by time, never shuffled)

| Split | Years | Purpose |
| :--- | :---: | :--- |
| Train | 2010–2019 | Long stretch of normal history |
| Validation | 2020–2021 | Includes the COVID crash; used to tune hyper-parameters (number of neighbors, tree depth, learning rate, network size) |
| Test | 2022–2025 | Touched once for the final score; includes the 2022 bear market and the March 2023 bank failures |

#### Prediction metrics

| Metric | What it tells us |
| :--- | :--- |
| **QLIKE** | Standard loss for volatility forecasts; punishes under-predicting risk more than over-predicting it |
| **MSE on log volatility** | Easy-to-read average error |
| **Diebold-Mariano test** | Whether a model's win over HAR is real or could be luck |

#### Overfitting and underfitting checks

- Plot **training vs. validation loss** for every model.
- Use **early stopping** for the neural network and the trees.
- Compare every model against the **naive and HAR baselines**:
  - Does not beat HAR → underfitting, or the model adds nothing.
  - Large train–validation gap → overfitting.

#### Clustering quality

| Measure | What it tells us |
| :--- | :--- |
| **Silhouette score** | How well separated the embedding clusters are |
| **Adjusted Rand index** | How much our clusters agree with the official sectors |

---

### 4c. How We Check the Results Are Meaningful

#### 1. Per-group prediction quality

We split the test data into groups and compare each with/without-neighbor pair within each group.

| Grouping | Groups | What we expect if the model behaves sensibly |
| :--- | :--- | :--- |
| Market stress | Calm days (VIX ≤ 25) vs. stressed days (VIX > 25) | Neighbors help most on stressed days, when shocks spread |
| Sector | 11 GICS sectors | Bigger gains in tightly linked sectors, e.g. banks and chip makers |
| Recent volatility | Top vs. bottom quartile of each stock's past-month volatility | Shows whether the gain comes only from already-volatile stocks |

#### 2. Inspect individual records

Look at test days where the with-neighbor model and HAR **disagree most**, and check which neighbors had large moves the day before.

#### 3. Case study: the March 2023 bank failures

- Walk through **8–17 March 2023** day by day.
- Check whether the with-neighbor models raised their forecasts for other banks (large and regional banks still in the index) the next day, while HAR did not.
- Silicon Valley Bank itself left the index, so it may be missing from our data; the case study focuses on the banks around it.

#### 4. Inspect cluster pairs

- Hand-check pairs of stocks in the **same** cluster and in **different** clusters.
- Ask whether a cross-sector pair makes economic sense (e.g. a chip maker grouped with a data-center real-estate company) or looks random.

---

## 5. Scope and Work Split

The core result is ready by **week 2** (HAR with vs. without neighbor features). Weeks 3–4 add the GNN, clustering and analysis.

### 5.1 Weekly plan

| Week | [Name 1] | [Name 2] |
| :---: | :--- | :--- |
| 1 | Download and save all data; compute volatility targets; set up splits | Build sector and correlation graphs; set up repo |
| | *Both:* read the GNNHAR paper | |
| 2 | Naive and HAR, with and without neighbors; metrics and Diebold-Mariano test | Gradient-boosted trees with and without neighbors; start GNNHAR |
| 3 | Per-group analysis; record inspection; March 2023 case study | Finish and tune GNNHAR on both graphs; cluster the learned embeddings |
| 4 | Final report | Video and figures |

### 5.2 Out of scope

Cut to fit a two-person, one-month project:

- GARCH and LSTM baselines
- A graph that rewires based on VIX
- Clustering trading days into market regimes
- Splits by earnings dates or company size
- 5-day-ahead forecasts *(added only if time allows)*

---

## 6. References

1. Zhang, Pu, Cucuringu, Dong. *Graph Neural Networks for Forecasting Realized Volatility with Nonlinear Spillover Effects.* International Journal of Forecasting, 2025. [arXiv:2308.01419](https://arxiv.org/abs/2308.01419)
   - We implement from the paper, not from its public code.
   - Our version uses free **daily** data instead of the paper's intraday data, so we expect smaller gains and will report that honestly.
2. Corsi. *A Simple Approximate Long-Memory Model of Realized Volatility* (the HAR model). Journal of Financial Econometrics, 2009.
