"""Two tests of sector information with the market taken out.

  python scripts/spillover.py

PART 1, excess volatility. Target: a stock's log-vol tomorrow minus the average log-vol of all other stocks
tomorrow (its position relative to the market). Inputs, all as excess over the market average today:
  own            the stock's own har_1/5/22 minus the market's
  own+sector_level   plus a fixed dummy per sector
  own+sector     plus the sector peers' average excess har_1/5/22 (stock itself left out)
  own+random     plus a fixed random group of the same size as the sector, 5 seeds
Pooled OLS fit on 2010-2019, scored by MSE of the excess log-vol on 2020-21 and 2022-25.
Rule (written before running): sector information predicts excess volatility if own+sector beats own by
>1% MSE with DM p<0.05 on both splits, beats own+random in every seed, and beats own+sector_level.

PART 2, shock spillover (event study). A stock has a SHOCK on day t when its own jump (log-vol today minus
its 22-day mean) minus the market's average jump is in the top 1% of training days, and the market itself
is not jumping (market average jump below its training 90th percentile). For every other stock on day t,
EXPOSED = at least one sector peer had a shock (stocks with their own shock are left out). The response is
the HAR surprise for tomorrow: y minus a pooled OLS forecast from own har_1/5/22 plus market har_1/5/22.
Per day: mean surprise of exposed stocks minus mean surprise of unexposed stocks (same day, so the market
cancels). Averaged over days, Newey-West p. Placebo: the same with random groups of the sector's size.
Also: the excess-vol path from day t to t+5 (log-vol minus own har_22 on day t, day-demeaned), by sector.
Rule: a spillover exists if exposed minus unexposed > 0 with p<0.05 on val and test and is larger than in
every random seed.
Outputs: results/excess_vol.csv, results/excess_vol_by_sector.csv, results/shock_spillover.csv,
results/shock_path.csv, results/shock_by_sector.csv.
"""
import numpy as np
import pandas as pd

from config import FEATURES_DIR, RESULTS_DIR
from metrics import dm_test

HAR = ["har_1", "har_5", "har_22"]
SEEDS = range(5)
SPLITS = ["val", "test"]
SHOCK_Q, MKT_CALM_Q, PATH_DAYS = 0.99, 0.90, 5


def group_mean(panel, groups, cols, tickers):
    """Per date and ticker, the mean of `cols` over the ticker's group (excluding itself)."""
    idx = {t: i for i, t in enumerate(tickers)}
    a = np.zeros((len(tickers), len(tickers)))
    for t, g in groups.items():
        a[idx[t], [idx[n] for n in g if n != t]] = 1.0
    out = {}
    for c in cols:
        w = panel.pivot(index="date", columns="ticker", values=c).reindex(columns=tickers)
        m, x = w.notna().to_numpy(float), np.nan_to_num(w.to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = pd.DataFrame((x @ a.T) / (m @ a.T), index=w.index, columns=tickers).stack()
    return pd.DataFrame(out)


def ols(df, x_cols, y_col, fit):
    x = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in x_cols])
    b = np.linalg.lstsq(x[fit], df[y_col].to_numpy(float)[fit], rcond=None)[0]
    return x @ b


def build(panel):
    sec = panel.drop_duplicates("ticker").set_index("ticker")["sector"]
    tickers = sorted(sec.index)
    tk = np.array(tickers)
    sector_groups = {t: sec.index[sec == s].to_numpy() for t, s in sec.items()}
    panel = panel.assign(jump=panel["lv"] - panel["har_22"])
    cols = HAR + ["y", "jump"]
    groups = {"mkt": {t: tk for t in tickers}, "sec": sector_groups}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        groups[f"rnd{s}"] = {t: rng.choice(tk[tk != t], size=len(sector_groups[t]) - 1, replace=False) for t in tickers}
    parts = [group_mean(panel, g, cols, tickers).add_prefix(f"{name}_") for name, g in groups.items()]
    f = pd.concat(parts, axis=1).rename_axis(["date", "ticker"]).reset_index()
    df = panel.merge(f, on=["date", "ticker"], how="left")
    # tomorrow's market log-vol: the mkt mean of y on date t is the mean of other stocks' log-vol on t+1
    df["ey"] = df["y"] - df["mkt_y"]
    for c in HAR:
        df[f"own_x_{c}"] = df[c] - df[f"mkt_{c}"]
        for g in ["sec"] + [f"rnd{s}" for s in SEEDS]:
            df[f"{g}_x_{c}"] = df[f"{g}_{c}"] - df[f"mkt_{c}"]
    for s in sorted(sec.unique())[1:]:
        df[f"is_{s}"] = (df["sector"] == s).astype(float)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True), groups


def daily_compare(df, la, lb, m):
    a = pd.Series(la[m]).groupby(df["date"].to_numpy()[m]).mean()
    b = pd.Series(lb[m]).groupby(df["date"].to_numpy()[m]).mean()
    return la[m].mean() / lb[m].mean() - 1, dm_test(a.to_numpy(), b.to_numpy())[2]


def part1(df):
    own = [f"own_x_{c}" for c in HAR]
    dummies = [c for c in df.columns if c.startswith("is_")]
    sets = {"own": own, "own+sector_level": own + dummies, "own+sector": own + [f"sec_x_{c}" for c in HAR]}
    for s in SEEDS:
        sets[f"own+random_s{s}"] = own + [f"rnd{s}_x_{c}" for c in HAR]
    need = sorted({c for v in sets.values() for c in v}) + ["ey"]
    ok = df[need].notna().all(axis=1).to_numpy()
    fit = ok & (df["split"] == "train").to_numpy()
    ey = df["ey"].to_numpy()
    L = {n: (ey - ols(df, c, "ey", fit)) ** 2 for n, c in sets.items()}
    rows, by_sec = [], []
    for split in SPLITS:
        m = ok & (df["split"] == split).to_numpy()
        var = ey[m].var()
        for n in sets:
            rows.append({"split": split, "model": n, "vs": "", "mse": L[n][m].mean(), "r2_of_excess": 1 - L[n][m].mean() / var,
                         "change": np.nan, "dm_p": np.nan})
        for a, b in [("own+sector", "own"), ("own+sector", "own+sector_level"), ("own+sector_level", "own")] + \
                    [("own+sector", f"own+random_s{s}") for s in SEEDS] + [(f"own+random_s{s}", "own") for s in SEEDS]:
            ch, p = daily_compare(df, L[a], L[b], m)
            rows.append({"split": split, "model": a, "vs": b, "mse": np.nan, "r2_of_excess": np.nan, "change": ch, "dm_p": p})
        for s, g in df[m].groupby("sector"):
            mm = m & (df["sector"] == s).to_numpy()
            ch, p = daily_compare(df, L["own+sector"], L["own"], mm)
            ch2, p2 = daily_compare(df, L["own+sector"], L["own+sector_level"], mm)
            by_sec.append({"split": split, "sector": s, "stocks": g["ticker"].nunique(), "vs_own": ch, "p_vs_own": p,
                           "vs_sector_level": ch2, "p_vs_sector_level": p2})
    return pd.DataFrame(rows), pd.DataFrame(by_sec)


def part2(df, groups):
    train = (df["split"] == "train").to_numpy()
    zjump = df["jump"] - df["mkt_jump"]
    thr = zjump[train].quantile(SHOCK_Q)
    calm_thr = df.loc[train, "mkt_jump"].quantile(MKT_CALM_Q)
    df["shock"] = (zjump >= thr) & (df["mkt_jump"] < calm_thr)
    x = HAR + [f"mkt_{c}" for c in HAR]
    ok = df[x + ["y"]].notna().all(axis=1).to_numpy()
    df["surprise"] = df["y"] - ols(df, x, "y", ok & train)
    df.loc[~ok, "surprise"] = np.nan
    # excess-vol path: log-vol on day t+h minus own har_22 on day t
    g = df.groupby("ticker")
    for h in range(0, PATH_DAYS + 1):
        df[f"path_{h}"] = g["lv"].shift(-h) - df["har_22"]
    tickers = sorted(df["ticker"].unique())
    shock_w = df.pivot(index="date", columns="ticker", values="shock").reindex(columns=tickers).fillna(False).to_numpy(float)
    dates = df.pivot(index="date", columns="ticker", values="shock").index
    idx = {t: i for i, t in enumerate(tickers)}

    def exposure(gr):
        a = np.zeros((len(tickers), len(tickers)))
        for t, mem in gr.items():
            a[idx[t], [idx[n] for n in mem if n != t]] = 1.0
        e = pd.DataFrame(shock_w @ a.T, index=dates, columns=tickers).stack().rename("n_exposed")
        return df[["date", "ticker"]].merge(e.rename_axis(["date", "ticker"]).reset_index(), how="left")["n_exposed"].to_numpy()

    out, paths, by_sec = [], [], []
    for name in ["sec"] + [f"rnd{s}" for s in SEEDS]:
        exp = exposure(groups[name]) > 0
        for split in SPLITS:
            m = (df["split"] == split).to_numpy() & ~df["shock"].to_numpy() & df["surprise"].notna().to_numpy()
            d = pd.DataFrame({"date": df["date"].to_numpy()[m], "e": exp[m], "r": df["surprise"].to_numpy()[m]})
            per = d.groupby(["date", "e"])["r"].mean().unstack().dropna()
            diff = (per[True] - per[False]).to_numpy()
            p = dm_test(diff, np.zeros_like(diff))[2]
            out.append({"split": split, "groups": name, "shock_days": len(per), "exposed_rows": int(d["e"].sum()),
                        "exposed_minus_unexposed": diff.mean(), "nw_p": p})
            if name == "sec":
                for h in range(0, PATH_DAYS + 1):
                    v = df[f"path_{h}"].to_numpy()[m]
                    dd = pd.DataFrame({"date": d["date"], "e": d["e"], "v": v}).dropna()
                    pp = dd.groupby(["date", "e"])["v"].mean().unstack().dropna()
                    dh = (pp[True] - pp[False]).to_numpy()
                    paths.append({"split": split, "day": h, "exposed_minus_unexposed": dh.mean(),
                                  "nw_p": dm_test(dh, np.zeros_like(dh))[2]})
                # within a sector every non-shocked stock is exposed, so compare exposed stocks of sector S
                # with the unexposed stocks of all other sectors on the same day
                # the sector's usual gap (its unexposed days) is subtracted, so a sector-wide bias of the
                # pooled forecast is not counted as spillover
                sd = d.assign(sector=df["sector"].to_numpy()[m])
                unexp = sd[~sd["e"]].groupby("date")["r"].mean()
                sd["gap"] = sd["r"] - sd["date"].map(unexp)
                for s, gs in sd.groupby("sector"):
                    usual = gs.loc[~gs["e"], "gap"].mean()
                    ds = (gs[gs["e"]].groupby("date")["gap"].mean() - usual).dropna()
                    if len(ds) < 10:
                        continue
                    per_s, ds = ds, ds.to_numpy()
                    by_sec.append({"split": split, "sector": s, "shock_days": len(per_s), "exposed_minus_unexposed": ds.mean(),
                                   "nw_p": dm_test(ds, np.zeros_like(ds))[2]})
    info = {"shock_threshold": thr, "market_calm_threshold": calm_thr,
            "shocks_per_split": df.groupby("split")["shock"].sum().to_dict()}
    return pd.DataFrame(out), pd.DataFrame(paths), pd.DataFrame(by_sec), info


def main():
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet")
    df, groups = build(panel)
    p1, p1s = part1(df)
    p2, path, p2s, info = part2(df, groups)
    RESULTS_DIR.mkdir(exist_ok=True)
    for frame, name in [(p1, "excess_vol"), (p1s, "excess_vol_by_sector"), (p2, "shock_spillover"),
                        (path, "shock_path"), (p2s, "shock_by_sector")]:
        frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    f = lambda v: f"{v:.4f}"
    print("PART 1: EXCESS VOLATILITY (target = own log-vol minus market log-vol tomorrow)")
    print(p1.to_string(index=False, float_format=f))
    print("\nby sector: own+sector vs own, and vs own+sector_level")
    print(p1s.to_string(index=False, float_format=f))
    print("\nPART 2: SHOCK SPILLOVER", info)
    print(p2.to_string(index=False, float_format=f))
    print("\nexcess-vol path after a peer shock (day 0 = shock day)")
    print(path.to_string(index=False, float_format=f))
    print("\nby sector (next-day surprise)")
    print(p2s.to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
