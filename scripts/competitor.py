"""Does the single closest competitor carry information that the whole sector does not?

  python scripts/competitor.py

Each stock gets ONE linked stock per month, chosen only from data before that month:
  comp_sub    same GICS sub-industry, highest correlation of market-removed daily returns over the 60
              trading days before the month (the closest rival by how it trades)
  comp_size   same sub-industry, closest in average dollar volume over those 60 days (rival of similar size)
  comp_leader same sub-industry, largest average dollar volume over those 60 days (the industry leader;
              needs no data of the stock itself, so it is the only one usable for a brand-new listing)
  comp_corr   top neighbor of the existing correlation graph (any sector, raw returns)
Two share classes of one company (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) are never linked to each other.
  random      one random other stock per month (5 seeds)
plus the sector average for comparison. Stocks alone in their sub-industry have no comp_sub / comp_size;
every comparison uses only rows where all inputs exist.

Three tests, on 2020-21 (val) and 2022-25 (test), fit on 2010-2019:
  A forecast    QLIKE of HAR + market + X vs HAR + market (pooled OLS, smearing); also with NO own history
  B excess vol  MSE of (own log-vol - market log-vol) tomorrow: own excess HAR + X's excess HAR vs own only
  C shock       the linked stock has its own shock today (as in spillover.py); next-day HAR surprise of
                exposed vs unexposed stocks on the same day, and the same-day jump
Rule (written before running): a competitor matters if, on val AND test, it beats the market-only model by
>1% QLIKE in A or >1% MSE in B with p<0.05, or has a next-day effect in C with p<0.05, AND beats every
random seed AND beats the sector average.
Outputs: results/competitor.csv, results/competitor_shock.csv, results/competitor_pairs_sample.csv
"""
import numpy as np
import pandas as pd

from config import FEATURES_DIR, PROCESSED_DIR, RESULTS_DIR
from metrics import dm_test, qlike_loss, smearing_factor
from spillover import HAR, MKT_CALM_Q, SHOCK_Q, group_mean, ols

WINDOW, MIN_COMMON = 60, 40
SEEDS = range(5)
SPLITS = ["val", "test"]


def company_base(stocks):
    """Company name without the share class, so GOOG and GOOGL map to the same company."""
    c = stocks.drop_duplicates("ticker").set_index("ticker")["company"]
    return c.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()


def competitor_maps(stocks, panel_tickers):
    s = stocks[stocks["ticker"].isin(panel_tickers)]
    base = company_base(s)
    close = s.pivot(index="date", columns="ticker", values="close").sort_index()
    dvol = (s.assign(dv=s["close"] * s["volume"]).pivot(index="date", columns="ticker", values="dv")
            .reindex(index=close.index, columns=close.columns))
    ret = np.log(close / close.shift(1))
    resid = ret.sub(ret.mean(axis=1), axis=0)
    sub = s.drop_duplicates("ticker").set_index("ticker")["sub_industry"].reindex(close.columns).to_numpy()
    tick = np.array(close.columns)
    b = base.reindex(tick).to_numpy()
    same = (sub[:, None] == sub[None, :]) & ~np.eye(len(tick), dtype=bool) & (b[:, None] != b[None, :])
    cal = close.index
    first = cal.to_series().groupby(cal.to_period("M")).first()
    rows = []
    for month, d0 in first.items():
        pos = cal.get_loc(d0)
        if pos < WINDOW:
            continue
        c = resid.iloc[pos - WINDOW:pos].corr(min_periods=MIN_COMMON).to_numpy(copy=True)
        size = np.log(dvol.iloc[pos - WINDOW:pos].mean().to_numpy())
        cs = np.where(same & np.isfinite(c), c, -np.inf)
        dz = np.where(same & np.isfinite(size)[None, :] & np.isfinite(size)[:, None], np.abs(size[:, None] - size[None, :]), np.inf)
        lead = np.where(same & np.isfinite(size)[None, :], size[None, :], -np.inf)
        bi, bs, bl = cs.argmax(1), dz.argmin(1), lead.argmax(1)
        for i, t in enumerate(tick):
            rows.append((month.to_timestamp(), t, tick[bi[i]] if np.isfinite(cs[i, bi[i]]) else None,
                         tick[bs[i]] if np.isfinite(dz[i, bs[i]]) else None,
                         tick[bl[i]] if np.isfinite(lead[i, bl[i]]) else None))
    m = pd.DataFrame(rows, columns=["month", "ticker", "comp_sub", "comp_size", "comp_leader"])
    cg = pd.read_parquet("data/graphs/corr_edges.parquet")
    cg = cg[cg["ticker"].map(base) != cg["neighbor"].map(base)]
    top = cg.sort_values("rank").drop_duplicates(["month", "ticker"])
    m = m.merge(top[["month", "ticker", "neighbor"]].rename(columns={"neighbor": "comp_corr"}),
                on=["month", "ticker"], how="left")
    pt = np.array(sorted(panel_tickers))
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(pt) - 1, len(m))
        own = np.searchsorted(pt, m["ticker"].to_numpy())
        m[f"random_s{seed}"] = pt[draw + (draw >= own)]          # any stock except itself
    return m


def build(panel, maps):
    sec = panel.drop_duplicates("ticker").set_index("ticker")["sector"]
    tickers = sorted(sec.index)
    tk = np.array(tickers)
    panel = panel.assign(jump=panel["lv"] - panel["har_22"])
    grp = {"mkt": {t: tk for t in tickers}, "sec": {t: sec.index[sec == s].to_numpy() for t, s in sec.items()}}
    f = pd.concat([group_mean(panel, g, HAR + ["y", "jump"], tickers).add_prefix(f"{n}_") for n, g in grp.items()], axis=1)
    df = panel.merge(f.rename_axis(["date", "ticker"]).reset_index(), on=["date", "ticker"], how="left")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(maps, on=["month", "ticker"], how="left")
    train = (df["split"] == "train").to_numpy()
    zj = df["jump"] - df["mkt_jump"]
    df["shock"] = (zj >= zj[train].quantile(SHOCK_Q)) & (df["mkt_jump"] < df.loc[train, "mkt_jump"].quantile(MKT_CALM_Q))
    look = df[["date", "ticker"] + HAR + ["jump", "shock"]]
    links = ["comp_sub", "comp_size", "comp_leader", "comp_corr"] + [f"random_s{s}" for s in SEEDS]
    for L in links:
        add = look.rename(columns={"ticker": L, **{c: f"{L}_{c}" for c in HAR + ["jump", "shock"]}})
        df = df.merge(add, on=["date", L], how="left")
    df["ey"] = df["y"] - df["mkt_y"]
    for c in HAR:
        for p in ["own", "sec"] + links:
            src = c if p == "own" else f"{p}_{c}"
            df[f"{p}_x_{c}"] = df[src] - df[f"mkt_{c}"]
    return df, links


def qlike_losses(df, cols, fit):
    yhat = ols(df, cols, "y", fit)
    y = df["y"].to_numpy()
    return qlike_loss(np.exp(2 * y), np.exp(2 * yhat) * smearing_factor(y[fit], yhat[fit]))


def cmp(df, la, lb, m):
    d = df["date"].to_numpy()[m]
    a, b = pd.Series(la[m]).groupby(d).mean(), pd.Series(lb[m]).groupby(d).mean()
    return la[m].mean() / lb[m].mean() - 1, dm_test(a.to_numpy(), b.to_numpy())[2]


def tests_ab(df, links):
    mkt = [f"mkt_{c}" for c in HAR]
    xs = {p: [f"{p}_{c}" for c in HAR] for p in ["sec"] + links}
    need = HAR + mkt + [c for v in xs.values() for c in v] + ["y", "ey"]
    ok = df[need].notna().all(axis=1).to_numpy()
    fit = ok & (df["split"] == "train").to_numpy()
    rows = []
    for own_days, own in [(22, HAR), (0, [])]:
        base = qlike_losses(df, own + mkt, fit)
        L = {p: qlike_losses(df, own + mkt + v, fit) for p, v in xs.items()}
        for split in SPLITS:
            m = ok & (df["split"] == split).to_numpy()
            for p in L:
                ch, pv = cmp(df, L[p], base, m)
                rows.append({"test": f"A forecast, own history {own_days}d", "split": split, "link": p, "vs": "market only",
                             "rows": int(m.sum()), "change": ch, "p": pv})
            for p in ["comp_sub", "comp_size", "comp_leader", "comp_corr"]:
                ch, pv = cmp(df, L[p], L["sec"], m)
                rows.append({"test": f"A forecast, own history {own_days}d", "split": split, "link": p, "vs": "sector average",
                             "rows": int(m.sum()), "change": ch, "p": pv})
    ey = df["ey"].to_numpy()
    own = [f"own_x_{c}" for c in HAR]
    se = lambda cols: (ey - ols(df, cols, "ey", fit)) ** 2
    base = se(own)
    L = {p: se(own + [f"{p}_x_{c}" for c in HAR]) for p in ["sec"] + links}
    for split in SPLITS:
        m = ok & (df["split"] == split).to_numpy()
        for p in L:
            ch, pv = cmp(df, L[p], base, m)
            rows.append({"test": "B excess vol", "split": split, "link": p, "vs": "own only", "rows": int(m.sum()), "change": ch, "p": pv})
        for p in ["comp_sub", "comp_size", "comp_leader", "comp_corr"]:
            ch, pv = cmp(df, L[p], L["sec"], m)
            rows.append({"test": "B excess vol", "split": split, "link": p, "vs": "sector average", "rows": int(m.sum()), "change": ch, "p": pv})
    return pd.DataFrame(rows)


def test_c(df, links):
    x = HAR + [f"mkt_{c}" for c in HAR]
    ok = df[x + ["y"]].notna().all(axis=1).to_numpy()
    df["surprise"] = df["y"] - ols(df, x, "y", ok & (df["split"] == "train").to_numpy())
    df["dayjump"] = df["jump"] - df["mkt_jump"]
    rows = []
    for L in links:
        has = df[f"{L}_shock"].notna().to_numpy() & ok & ~df["shock"].to_numpy()
        exp = df[f"{L}_shock"].fillna(False).astype(bool).to_numpy()
        for split in SPLITS:
            m = has & (df["split"] == split).to_numpy()
            for resp, label in [("dayjump", "same day"), ("surprise", "next day, beyond HAR")]:
                d = pd.DataFrame({"date": df["date"].to_numpy()[m], "e": exp[m], "r": df[resp].to_numpy()[m]})
                per = d.groupby(["date", "e"])["r"].mean().unstack()
                if True not in per.columns:
                    continue
                per = per.dropna()
                diff = (per[True] - per[False]).to_numpy()
                rows.append({"split": split, "link": L, "response": label, "shock_days": len(per),
                             "exposed_rows": int(d["e"].sum()), "exposed_minus_unexposed": diff.mean(),
                             "p": dm_test(diff, np.zeros_like(diff))[2]})
    return pd.DataFrame(rows)


def summarize(t, col):
    t = t.copy()
    t["link"] = t["link"].str.replace(r"_s\d", "", regex=True)
    keys = [c for c in t.columns if c not in (col, "p", "rows", "exposed_rows", "shock_days")]
    return t.groupby(keys, sort=False).agg(change=(col, "mean"), worst=(col, "max"), max_p=("p", "max")).reset_index()


def main():
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet")
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet", columns=["date", "ticker", "close", "volume", "sub_industry", "company"])
    maps = competitor_maps(stocks, panel["ticker"].unique())
    df, links = build(panel, maps)
    ab, c = tests_ab(df, links), test_c(df, links)
    RESULTS_DIR.mkdir(exist_ok=True)
    ab.to_csv(RESULTS_DIR / "competitor.csv", index=False, float_format="%.6g")
    c.to_csv(RESULTS_DIR / "competitor_shock.csv", index=False, float_format="%.6g")
    last = maps[maps["month"] == maps["month"].max()][["ticker", "comp_sub", "comp_size", "comp_leader", "comp_corr"]]
    last.to_csv(RESULTS_DIR / "competitor_pairs_sample.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    f = lambda v: f"{v:.4f}"
    cov = maps[["comp_sub", "comp_size", "comp_leader", "comp_corr"]].notna().mean()
    print("coverage of stock-months:", cov.round(3).to_dict())
    print("sample pairs (latest month):")
    print(last[last["ticker"].isin(["JPM", "KO", "XOM", "HD", "NVDA", "UNH", "MSFT", "DAL", "WMT", "GOOGL"])].to_string(index=False))
    print("\nTESTS A, B (change: negative = better; random: mean over 5 seeds, worst seed, max p)")
    print(summarize(ab, "change").to_string(index=False, float_format=f))
    print("\nTEST C: SHOCK IN THE LINKED STOCK (positive = exposed stocks more volatile)")
    cc = c.rename(columns={"exposed_minus_unexposed": "change"})
    print(summarize(cc, "change").rename(columns={"worst": "max_seed"}).to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
