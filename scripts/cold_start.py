"""Cold start: can a stock with little or no history of its own borrow its sector's volatility?

  python scripts/cold_start.py

Simulated cold start (every stock, every day): pretend the stock only has its last k days of its own
volatility, k = 0 (brand new), 1, 5, 10, and 22 (full HAR, the reference). Own features for k days are
today's log-vol and the mean of the last k days' log-vol (k = 0: none). Peers always have full history.
Each own-history level is crossed with a peer input:
  none           own history only (k = 0: a constant)
  market         mean har_1/5/22 of all other stocks that day
  market+sector_level   market plus a fixed dummy per sector (the sector's static level, no live data)
  sector         mean har_1/5/22 of the other stocks in the same sector that day
  market+sector  both live averages
  random, market+random   a fixed random group of the same size as the stock's sector, 5 seeds
All models are pooled OLS on log-vol, fit on the training years, scored by QLIKE with smearing.

Real new listings (check): the 83 stocks whose first row is after March 2010, in their first 60 rows
(their own HAR already exists there, since a row needs 22 days of history). Models are fit on training
rows with those early rows removed and scored on the early rows of every year.

Reading the result (written before running): the sector helps a new stock if market+sector beats
market by >1% QLIKE with DM p<0.05 on validation AND test, AND beats market+random in every seed, AND
beats market+sector_level (otherwise the gain is the static sector level, not live sector news).
Outputs: results/cold_start.csv, results/cold_start_new_listings.csv.
"""
import numpy as np
import pandas as pd

from config import FEATURES_DIR, RESULTS_DIR
from metrics import dm_test, qlike_loss, smearing_factor

HAR = ["har_1", "har_5", "har_22"]
KS = [0, 1, 5, 10, 22]
SEEDS = range(5)
EARLY_ROWS, LATE_START = 60, "2010-03-01"
SPLITS = ["val", "test"]


def peer_means(panel, groups, prefix):
    """Mean har_* over each stock's group (excluding itself), per date. groups: ticker -> array of tickers."""
    tickers = sorted(panel["ticker"].unique())
    idx = {t: i for i, t in enumerate(tickers)}
    a = np.zeros((len(tickers), len(tickers)))
    for t, g in groups.items():
        a[idx[t], [idx[n] for n in g if n != t]] = 1.0
    out = []
    for c in HAR:
        w = panel.pivot(index="date", columns="ticker", values=c).reindex(columns=tickers)
        m = w.notna().to_numpy(float)
        x = np.nan_to_num(w.to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (x @ a.T) / (m @ a.T)
        out.append(pd.DataFrame(mean, index=w.index, columns=tickers).stack().rename(f"{prefix}_{c}"))
    return pd.concat(out, axis=1)


def build(panel):
    sec = panel.drop_duplicates("ticker").set_index("ticker")["sector"]
    tickers = np.array(sorted(sec.index))
    sector_groups = {t: sec.index[sec == s].to_numpy() for t, s in sec.items()}
    feats = [peer_means(panel, {t: tickers for t in tickers}, "mkt"), peer_means(panel, sector_groups, "sec")]
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rnd = {t: rng.choice(tickers[tickers != t], size=len(sector_groups[t]) - 1, replace=False) for t in tickers}
        feats.append(peer_means(panel, rnd, f"rnd{seed}"))
    f = pd.concat(feats, axis=1).rename_axis(["date", "ticker"]).reset_index()
    df = panel.merge(f, on=["date", "ticker"], how="left").sort_values(["ticker", "date"]).reset_index(drop=True)
    for k in KS[1:-1]:
        df[f"own_mean{k}"] = df.groupby("ticker")["lv"].transform(lambda s: s.rolling(k, min_periods=k).mean())
    for s in sorted(sec.unique())[1:]:
        df[f"is_{s}"] = (df["sector"] == s).astype(float)
    df["row_no"] = df.groupby("ticker").cumcount()
    first = df.groupby("ticker")["date"].transform("min")
    df["early_new"] = (first > LATE_START) & (df["row_no"] < EARLY_ROWS)
    return df


def own_cols(k):
    return [] if k == 0 else HAR if k == 22 else ["lv"] if k == 1 else ["lv", f"own_mean{k}"]


def peer_sets(df):
    sec_dummies = [c for c in df.columns if c.startswith("is_")]
    mkt, sec = [f"mkt_{c}" for c in HAR], [f"sec_{c}" for c in HAR]
    sets = {"none": [], "market": mkt, "market+sector_level": mkt + sec_dummies, "sector": sec,
            "market+sector": mkt + sec}
    for s in SEEDS:
        rnd = [f"rnd{s}_{c}" for c in HAR]
        sets[f"random_s{s}"], sets[f"market+random_s{s}"] = rnd, mkt + rnd
    return sets


def fit_predict(df, cols, fit_mask):
    x = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    y = df["y"].to_numpy(float)
    b = np.linalg.lstsq(x[fit_mask], y[fit_mask], rcond=None)[0]
    yhat = x @ b
    return yhat, smearing_factor(y[fit_mask], yhat[fit_mask])


def losses(df, yhat, smear):
    return qlike_loss(np.exp(2 * df["y"].to_numpy()), np.exp(2 * yhat) * smear)


def daily(df, loss, mask):
    return pd.Series(loss[mask]).groupby(df["date"].to_numpy()[mask]).mean()


def compare(df, la, lb, mask):
    """QLIKE change of a vs b and DM p on daily mean losses."""
    a, b = daily(df, la, mask), daily(df, lb, mask)
    _, _, p = dm_test(a.to_numpy(), b.to_numpy())
    return la[mask].mean() / lb[mask].mean() - 1, p


def simulated(df):
    sets = peer_sets(df)
    rows, comps = [], []
    for k in KS:
        need = own_cols(k) + sorted({c for v in sets.values() for c in v})
        ok = df[need + ["y"]].notna().all(axis=1).to_numpy()
        fit = ok & (df["split"] == "train").to_numpy()
        L = {}
        for name, cols in sets.items():
            yhat, sm = fit_predict(df, own_cols(k) + cols, fit)
            L[name] = losses(df, yhat, sm)
        for split in SPLITS:
            m = ok & (df["split"] == split).to_numpy()
            for name in sets:
                rows.append({"own_days": k, "split": split, "peers": name, "rows": int(m.sum()), "qlike": L[name][m].mean()})
            for a, b in [("market", "none"), ("sector", "market"), ("market+sector", "market"),
                         ("market+sector", "market+sector_level"), ("market+sector_level", "market")] + \
                        [("market+sector", f"market+random_s{s}") for s in SEEDS] + \
                        [("sector", f"random_s{s}") for s in SEEDS]:
                ch, p = compare(df, L[a], L[b], m)
                comps.append({"own_days": k, "split": split, "model": a, "vs": b, "qlike_change": ch, "dm_p": p})
    return pd.DataFrame(rows), pd.DataFrame(comps)


def new_listings(df):
    sets = {k: v for k, v in peer_sets(df).items() if k in ("none", "market", "market+sector_level", "sector", "market+sector")
            or k.startswith("market+random")}
    rows = []
    for k in [0, 22]:
        need = own_cols(k) + sorted({c for v in sets.values() for c in v})
        ok = df[need + ["y"]].notna().all(axis=1).to_numpy()
        fit = ok & (df["split"] == "train").to_numpy() & ~df["early_new"].to_numpy()
        L = {n: losses(df, *fit_predict(df, own_cols(k) + c, fit)) for n, c in sets.items()}
        for grp, m in [("new listings, first 60 rows", ok & df["early_new"].to_numpy()),
                       ("all other rows, val+test", ok & ~df["early_new"].to_numpy() & df["split"].isin(SPLITS).to_numpy())]:
            for a, b in [("market", "none"), ("market+sector", "market"), ("market+sector", "market+sector_level")] + \
                        [("market+sector", f"market+random_s{s}") for s in SEEDS]:
                ch, p = compare(df, L[a], L[b], m)
                rows.append({"own_days": k, "rows_group": grp, "stocks": df.loc[m, "ticker"].nunique(), "rows": int(m.sum()),
                             "model": a, "vs": b, "qlike_change": ch, "dm_p": p})
    return pd.DataFrame(rows)


def main():
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet")
    df = build(panel)
    levels, comps = simulated(df)
    newl = new_listings(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    comps.to_csv(RESULTS_DIR / "cold_start.csv", index=False, float_format="%.6g")
    levels.to_csv(RESULTS_DIR / "cold_start_levels.csv", index=False, float_format="%.6g")
    newl.to_csv(RESULTS_DIR / "cold_start_new_listings.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    lv = levels[~levels["peers"].str.contains("random")].pivot_table(index=["own_days", "split"], columns="peers", values="qlike")
    print("QLIKE BY OWN-HISTORY DAYS AND PEER INPUT (lower is better)")
    print(lv.round(4).to_string())
    c = comps.copy()
    c["vs"] = c["vs"].str.replace(r"_s\d", "_seeds", regex=True)
    agg = c.groupby(["own_days", "split", "model", "vs"]).agg(change=("qlike_change", "mean"), worst_change=("qlike_change", "max"),
                                                              max_p=("dm_p", "max")).reset_index()
    print("\nCOMPARISONS (random: mean over 5 seeds, worst seed, max p)")
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n = newl.copy()
    n["vs"] = n["vs"].str.replace(r"_s\d", "_seeds", regex=True)
    print("\nREAL NEW LISTINGS")
    print(n.groupby(["own_days", "rows_group", "model", "vs"]).agg(stocks=("stocks", "first"), rows=("rows", "first"),
          change=("qlike_change", "mean"), max_p=("dm_p", "max")).reset_index().to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
