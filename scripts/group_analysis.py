"""Per-group prediction quality (proposal 4c.1): does the neighbor gain depend on market stress,
on how volatile the stock has been lately, or on the sector?

  python scripts/group_analysis.py              # validation (2020-2021), from data/predictions/<model>.parquet
  python scripts/group_analysis.py --split test # test (2022-2025), from <model>_with_test.parquet

Run baselines.py, trees.py and gnnhar.py first (with --include-test for the test split).

PAIRS (treatment vs control; each pair is scored on the rows both models forecast):
  with vs without neighbors, one pair per model family and graph
    har_sector / har_corr vs har          (linear, OLS + smearing)
    gbt_sector / gbt_corr vs gbt          (trees)
    gnn_sector / gnn_corr vs nn_own       (GNNHAR vs the same network without a graph)
  market controls (GNNHAR), because the validation results say the gain is market-wide
    gnn_market vs nn_own                  market average alone
    gnn_market+sector / +corr vs gnn_market  graph beyond the market

GROUPS (all known on the forecast date t, before the predicted day; no look-ahead)
  stress      VIX close on day t: calm (<= 25) vs stressed (> 25)
  recent_vol  the stock's har_22 (mean log-vol over the last 22 days) against ITS OWN quartiles,
              computed from its rows BEFORE the scored split begins (train for validation,
              train + validation for test; features only, never labels). Bottom vs top quartile;
              the middle half is reported too. Stocks with fewer than MIN_HISTORY such rows are left out.
  sector      the 11 GICS sectors

NUMBERS per pair and group: QLIKE of each model (row mean, each model's own smearing factor),
relative change (negative = treatment better), DM p on the group's daily mean loss, and the change
without Feb-Apr 2020 (validation only; the test years have no such window).
Difference tests: stressed minus calm gain, and top minus bottom quartile gain, each a Newey-West
t-test on the daily loss differences (does the gain really differ between the two groups?). The
changes shown beside them are row-weighted, as in the per-group table; the tests weight each day equally.

READING, written before the test years were scored. The proposal's expectations (4c.1):
  (1) neighbors help most on stressed days, (2) more in tightly linked sectors (e.g. banks,
  chip makers), (3) the gain is not only in already-volatile stocks.
Each is called "supported" only if the difference is in the expected direction with p < 0.05 for
the with/without pairs of at least two of the three model families, on the split being read.
Per-group p-values are many (11 sectors x 9 pairs); a per-group result counts as significant only
below BONFERRONI = 0.05 / number of per-group tests in that grouping. Everything else is descriptive.

Outputs: results/groups{_test}.csv (pair x group), results/groups_diff{_test}.csv (difference tests).
"""
import argparse
import math

import numpy as np
import pandas as pd

from config import FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, SPLITS
from metrics import dm_test, qlike_loss

VIX_STRESS = 25.0
MIN_HISTORY = 250
CSV_FMT = "%.8g"
PAIRS = [("har_sector", "har"), ("har_corr", "har"),
         ("gbt_sector", "gbt"), ("gbt_corr", "gbt"),
         ("gnn_sector", "nn_own"), ("gnn_corr", "nn_own"),
         ("gnn_market", "nn_own"),
         ("gnn_market+sector", "gnn_market"), ("gnn_market+corr", "gnn_market")]
FAMILY = {"har": "linear", "gbt": "trees", "nn_own": "gnnhar"}
# the with/without-neighbor pairs the verdicts read (the market controls are reported, not judged)
NEIGHBOR_PAIRS = [(a, b) for a, b in PAIRS if b in FAMILY and a != "gnn_market"]
COVID = ("2020-02-01", "2020-04-30")


def load_predictions(name, split):
    tag = "_with_test" if split == "test" else ""
    p = pd.read_parquet(PREDICTIONS_DIR / f"{name.replace('+', '_plus_')}{tag}.parquet")
    p = p[p["split"] == split]
    p = p.assign(loss=qlike_loss(np.exp(2 * p["y"].to_numpy()), np.exp(2 * p["yhat"].to_numpy()) * p["smear"].to_numpy()))
    return p[["ticker", "date", "y_date", "loss"]]


def vol_quartiles(panel, split):
    """Per-stock har_22 quartile cut points from rows before `split` starts."""
    start = SPLITS[split][0]
    hist = panel[panel["y_date"] < start]
    g = hist.groupby("ticker")["har_22"]
    q = pd.DataFrame({"n": g.size(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)})
    return q[q["n"] >= MIN_HISTORY]


def add_groups(rows, panel, split):
    """rows: ticker, date, ... Adds stress, recent_vol, sector from the panel (info known on date t)."""
    info = panel[["ticker", "date", "vix", "har_22", "sector"]]
    out = rows.merge(info, on=["ticker", "date"], how="left", validate="1:1")
    out["stress"] = np.where(out["vix"] > VIX_STRESS, "stressed (VIX>25)", "calm (VIX<=25)")
    q = vol_quartiles(panel, split)
    out = out.merge(q[["q25", "q75"]], left_on="ticker", right_index=True, how="left")
    out["recent_vol"] = np.select([out["har_22"] <= out["q25"], out["har_22"] >= out["q75"]],
                                  ["bottom quartile", "top quartile"], "middle half")
    out.loc[out["q25"].isna(), "recent_vol"] = np.nan
    return out.drop(columns=["q25", "q75"])


def daily(frame, col):
    return frame.groupby("y_date")[col].mean()


def rel(a, b):
    return float(a.mean() / b.mean() - 1)


def nw_mean_test(d, lag=None):
    """Newey-West t-test that the mean of series d is zero. Returns (mean, p)."""
    d = np.asarray(d, float)
    if len(d) < 10:
        return float(d.mean()) if len(d) else np.nan, np.nan
    return dm_test(d, np.zeros_like(d), lag)[0::2]


def nw_dummy_test(d, dummy, lag=None):
    """OLS of d on [1, dummy] with Newey-West standard errors; returns (coef on dummy, two-sided p).
    The coefficient is mean(d | dummy=1) - mean(d | dummy=0)."""
    d, x = np.asarray(d, float), np.column_stack([np.ones(len(d)), np.asarray(dummy, float)])
    t = len(d)
    if t < 10 or x[:, 1].min() == x[:, 1].max():
        return np.nan, np.nan
    lag = int(t ** (1 / 3)) if lag is None else lag
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ d
    u = x * (d - x @ beta)[:, None]
    s = u.T @ u
    for k in range(1, lag + 1):
        g = u[k:].T @ u[:-k]
        s += (1 - k / (lag + 1)) * (g + g.T)
    se = math.sqrt((xtx_inv @ s @ xtx_inv)[1, 1])
    z = beta[1] / se if se > 0 else 0.0
    return float(beta[1]), float(math.erfc(abs(z) / math.sqrt(2)))


def pair_frame(a, b, preds, panel, split):
    j = preds[a].merge(preds[b], on=["ticker", "date", "y_date"], suffixes=("_a", "_b"))
    return add_groups(j, panel, split)


def group_rows(j, a, b, grouping, split):
    rows = []
    groups = [("all", j)] if grouping == "all" else list(j.dropna(subset=[grouping]).groupby(grouping))
    for g, f in groups:
        da, db = daily(f, "loss_a"), daily(f, "loss_b")
        p = dm_test(da.to_numpy(), db.to_numpy())[2] if len(da) >= 10 else np.nan
        ex = f[(f["y_date"] < COVID[0]) | (f["y_date"] > COVID[1])]
        rows.append({"model": a, "versus": b, "grouping": grouping, "group": g,
                     "rows": len(f), "days": len(da), "stocks": f["ticker"].nunique(),
                     "qlike_model": f["loss_a"].mean(), "qlike_versus": f["loss_b"].mean(),
                     "qlike_change": rel(f["loss_a"], f["loss_b"]), "dm_p": p,
                     "qlike_change_exCOVID": rel(ex["loss_a"], ex["loss_b"]) if split == "val" and len(ex) else np.nan})
    return rows


def diff_rows(j, a, b):
    """Does the gain differ between stressed and calm days, and between top and bottom quartile?"""
    out = []
    j = j.assign(d=j["loss_a"] - j["loss_b"])
    # stress is a property of the day: regress the daily mean difference on a stressed-day dummy
    day = j.groupby("y_date").agg(d=("d", "mean"), b=("loss_b", "mean"), stressed=("vix", lambda v: v.iloc[0] > VIX_STRESS))
    coef, p = nw_dummy_test(day["d"], day["stressed"])
    hi = j["vix"] > VIX_STRESS
    out.append({"model": a, "versus": b, "test": "stressed minus calm",
                "change_group1": rel(j.loc[hi, "loss_a"], j.loc[hi, "loss_b"]) if hi.any() else np.nan,
                "change_group2": rel(j.loc[~hi, "loss_a"], j.loc[~hi, "loss_b"]) if (~hi).any() else np.nan,
                "days_group1": int(day["stressed"].sum()), "days_group2": int((~day["stressed"]).sum()),
                "diff_in_mean_loss_gap": coef, "p": p})
    # recent volatility varies within a day: compare the daily mean difference of the two quartiles
    v = j.dropna(subset=["recent_vol"])
    rt, rb = v[v["recent_vol"] == "top quartile"], v[v["recent_vol"] == "bottom quartile"]
    top, bot = rt.groupby("y_date").agg(d=("d", "mean")), rb.groupby("y_date").agg(d=("d", "mean"))
    both = top.join(bot, how="inner", suffixes=("_top", "_bot"))
    mean, p = nw_mean_test(both["d_top"] - both["d_bot"])
    out.append({"model": a, "versus": b, "test": "top minus bottom quartile",
                "change_group1": rel(rt["loss_a"], rt["loss_b"]) if len(rt) else np.nan,
                "change_group2": rel(rb["loss_a"], rb["loss_b"]) if len(rb) else np.nan,
                "days_group1": len(top), "days_group2": len(bot), "diff_in_mean_loss_gap": mean, "p": p})
    return out


def analyse(preds, panel, split):
    rows, diffs = [], []
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        j = pair_frame(a, b, preds, panel, split)
        for grouping in ["all", "stress", "recent_vol", "sector"]:
            rows += group_rows(j, a, b, grouping, split)
        diffs += diff_rows(j, a, b)
    res, diff = pd.DataFrame(rows), pd.DataFrame(diffs)
    n_tests = res[res["grouping"] != "all"].groupby("grouping").size()
    res["significant_bonferroni"] = res.apply(
        lambda r: bool(r.dm_p < 0.05 / n_tests[r.grouping]) if r.grouping in n_tests else bool(r.dm_p < 0.05), axis=1)
    return res, diff


def verdicts(res, diff):
    """The three proposal expectations, read by the rule in the docstring."""
    neighbor = NEIGHBOR_PAIRS
    out = {}
    # (1) stress: the gain is larger (more negative) on stressed days
    d = diff[(diff["test"] == "stressed minus calm") & diff.apply(lambda r: (r.model, r.versus) in neighbor, axis=1)]
    fams = {FAMILY[r.versus] for r in d.itertuples() if r.diff_in_mean_loss_gap < 0 and r.p < 0.05}
    out["(1) bigger gain on stressed days"] = ("supported" if len(fams) >= 2 else "not supported", sorted(fams))
    # (3) not only volatile stocks: the bottom quartile also gains, significantly, in 2+ families
    r = res[(res["grouping"] == "recent_vol") & (res["group"] == "bottom quartile")
            & res.apply(lambda x: (x.model, x.versus) in neighbor, axis=1)]
    fams = {FAMILY[x.versus] for x in r.itertuples() if x.qlike_change < 0 and x.dm_p < 0.05}
    out["(3) quiet stocks gain too"] = ("supported" if len(fams) >= 2 else "not supported", sorted(fams))
    # (2) tightly linked sectors: Financials and Information Technology gain more than the pair's overall gain
    s = res[res.apply(lambda x: (x.model, x.versus) in neighbor, axis=1)]
    fams = set()
    for (a, b), g in s.groupby(["model", "versus"]):
        overall = g.loc[g["group"] == "all", "qlike_change"].iloc[0]
        sec = g[(g["grouping"] == "sector") & g["group"].isin(["Financials", "Information Technology"])]
        if len(sec) == 2 and (sec["qlike_change"] < overall).all() and (sec["dm_p"] < 0.05).all():
            fams.add(FAMILY[b])
    out["(2) bigger gain in banks and tech"] = ("supported" if len(fams) >= 2 else "not supported", sorted(fams))
    return out


def main(split):
    names = sorted({m for pair in PAIRS for m in pair})
    preds = {}
    for n in names:
        try:
            preds[n] = load_predictions(n, split)
        except FileNotFoundError:
            print(f"  missing predictions for {n}; its pairs are skipped")
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet")
    res, diff = analyse(preds, panel, split)

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "_test" if split == "test" else ""
    res.to_csv(RESULTS_DIR / f"groups{tag}.csv", index=False, float_format=CSV_FMT)
    diff.to_csv(RESULTS_DIR / f"groups_diff{tag}.csv", index=False, float_format=CSV_FMT)

    pct = lambda v: f"{v:+.2%}" if pd.notna(v) else ""
    print(f"PER-GROUP QLIKE CHANGE ({split}; negative = first model better; * = DM p<0.05, ** = Bonferroni)")
    show = res.assign(cell=res.apply(lambda r: pct(r.qlike_change) + ("**" if r.significant_bonferroni and r.grouping != "all"
                                                                     else "*" if r.dm_p < 0.05 else ""), axis=1),
                      pair=res["model"] + " vs " + res["versus"])
    for grouping in ["all", "stress", "recent_vol", "sector"]:
        t = show[show["grouping"] == grouping].pivot(index="group", columns="pair", values="cell")
        t = t[[f"{a} vs {b}" for a, b in PAIRS if f"{a} vs {b}" in t.columns]]
        print(f"\n[{grouping}]")
        print(t.T.to_string())
    print("\nDIFFERENCE TESTS (relative gain in each group; p for 'the gain differs')")
    print(diff.assign(change_group1=diff["change_group1"].map(pct), change_group2=diff["change_group2"].map(pct))
          [["model", "versus", "test", "change_group1", "change_group2", "days_group1", "days_group2", "p"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nPROPOSAL EXPECTATIONS (rule in the docstring)")
    for k, (v, fams) in verdicts(res, diff).items():
        print(f"  {k}: {v}  (families meeting it: {', '.join(fams) or 'none'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    main(ap.parse_args().split)
