"""Sensitivity of the neighbor gain to the untested design choices (DECISIONS.md #10, 16-18, 20-22, 25).

  python scripts/sensitivity.py        # validation only; the test years are never scored

One setting changes at a time around the proposal's defaults. For each variant we fit the three HAR
models (none / sector / correlation neighbors) on the training years and score validation.

DECISION RULE (written before any variant was run):
  A graph variant REPLACES the default for a graph (sector or correlation) only if, on the rows
  both versions score, ALL of:
    (a) its neighbor model's QLIKE is lower than the default's by more than 1.0% (relative),
    (b) a Diebold-Mariano test of variant vs default gives p < 0.05,
    (c) the improvement holds with Feb-Apr 2020 removed, at more than 0.5%.
  Otherwise the default stays and the report says the choice was tested and did not matter.
  Per-stock HAR (#25) and the winsorizing variants (#10) change the baseline model or the target
  itself, so they are REPORTED ONLY: they are design questions for the team, not tuning. Their
  neighbor gain is compared within the variant, because absolute losses are on different targets.
  We run ~15 variants, so one 'significant' result is expected by chance (see #40); the 1% margin
  and condition (c) are there to guard against that.

Outputs: results/sensitivity.csv.
"""
import time

import numpy as np
import pandas as pd

import baselines as bl
from config import (CORR_K, CORR_MIN_COMMON, CORR_WINDOW, PROCESSED_DIR, RESULTS_DIR, WINSOR_Q)
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from metrics import dm_test, qlike_loss, smearing_factor

DEFAULTS = dict(window=CORR_WINDOW, k=CORR_K, min_common=CORR_MIN_COMMON, freq="M", weight=None,
                sector_level="sector", winsor_q=WINSOR_Q, per_stock=False)
# min_common scales with the window (2/3, as 40 of 60) unless that is the thing being varied
VARIANTS = [
    dict(name="default"),
    dict(name="window_30", window=30, min_common=20),
    dict(name="window_120", window=120, min_common=80),
    dict(name="window_250", window=250, min_common=167),
    dict(name="k_5", k=5),
    dict(name="k_20", k=20),
    dict(name="refresh_weekly", freq="W"),
    dict(name="refresh_quarterly", freq="Q"),
    dict(name="min_common_30", min_common=30),
    dict(name="min_common_50", min_common=50),
    dict(name="weight_by_corr", weight="corr"),
    dict(name="sector_is_sub_industry", sector_level="sub_industry"),
    dict(name="winsor_1_99", winsor_q=(0.01, 0.99)),
    dict(name="winsor_none", winsor_q=(0.0, 1.0)),
    dict(name="har_per_stock", per_stock=True),
]
COVID = ("2020-02-15", "2020-04-30")
MIN_TRAIN_ROWS = 100      # per-stock fits need this many training rows, else pooled coefficients are used
MODELS = {"har": bl.HAR, "har_sector": bl.HAR + bl.SEC, "har_corr": bl.HAR + bl.COR}


def fit_models(df, per_stock):
    """Returns (per-model yhat, smearing factor, number of tickers that fell back to pooled)."""
    train = (df["split"] == "train").to_numpy()
    y = df["y"].to_numpy()
    groups = df.groupby("ticker").indices
    yhat, smear, fallback = {}, {}, 0
    for name, cols in MODELS.items():
        x = df[cols].to_numpy()
        pooled = bl.fit_ols(x[train], y[train])
        pred = bl.predict_ols(pooled, x)
        if per_stock:
            fallback = 0
            for _, idx in groups.items():
                tr = idx[train[idx]]
                if len(tr) >= MIN_TRAIN_ROWS:
                    pred[idx] = bl.predict_ols(bl.fit_ols(x[tr], y[tr]), x[idx])
                else:
                    fallback += 1
        yhat[name] = pred
        smear[name] = smearing_factor(y[train], pred[train])
    return yhat, smear, fallback


def val_losses(df, yhat, smear):
    """Per-row QLIKE on validation rows, keyed by (ticker, y_date)."""
    m = (df["split"] == "val").to_numpy()
    out = df.loc[m, ["ticker", "y_date"]].reset_index(drop=True)
    v = np.exp(2 * df["y"].to_numpy()[m])
    for name in MODELS:
        out[name] = qlike_loss(v, np.exp(2 * yhat[name][m]) * smear[name])
    return out


def daily(frame, col):
    return frame.groupby("y_date")[col].mean()


def rel(a, b):
    return float(a.mean() / b.mean() - 1)


def excl_covid(frame):
    d = frame["y_date"]
    return frame[~((d >= COVID[0]) & (d <= COVID[1]))]


def compare(frame_a, frame_b, col):
    """Model `col` in A vs the same model in B on rows both score: (relative change, DM p, ex-COVID change)."""
    j = frame_a.merge(frame_b, on=["ticker", "y_date"], suffixes=("_a", "_b"))
    a, b = daily(j, f"{col}_a"), daily(j, f"{col}_b")
    _, _, p = dm_test(a.to_numpy(), b.to_numpy())
    jc = excl_covid(j)
    return rel(j[f"{col}_a"], j[f"{col}_b"]), p, rel(jc[f"{col}_a"], jc[f"{col}_b"]), len(j)


def main():
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    calendar = pd.DatetimeIndex(sorted(vix["date"]))
    returns = return_matrix(stocks, calendar)

    feat_cache, sec_cache, cor_cache, records, frames = {}, {}, {}, [], {}
    for v in VARIANTS:
        t0 = time.time()
        p = {**DEFAULTS, **v}
        if p["winsor_q"] not in feat_cache:
            rows, _ = compute_features(stocks, vix, p["winsor_q"])
            feat_cache[p["winsor_q"]] = (rows, panel_from_rows(rows))
        rows, panel = feat_cache[p["winsor_q"]]
        if p["sector_level"] not in sec_cache:
            sec_cache[p["sector_level"]] = sector_edges(stocks, p["sector_level"])
        key = (p["k"], p["window"], p["min_common"], p["freq"])
        if key not in cor_cache:
            cor_cache[key] = corr_edges(returns, p["k"], p["window"], p["min_common"], p["freq"])
        nbr = neighbor_features(rows, sec_cache[p["sector_level"]], cor_cache[key], p["freq"], p["weight"])
        df = bl.common_sample(panel.merge(nbr, on=["ticker", "date"], how="left"))
        df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)

        yhat, smear, fallback = fit_models(df, p["per_stock"])
        loss = val_losses(df, yhat, smear)
        frames[v["name"]] = loss
        rec = {"variant": v["name"], "val_rows": len(loss), "dropped_vs_panel_pct": 100 * (1 - len(df) / len(
            panel[panel["split"].isin(["train", "val"])])), "perstock_fallback_tickers": fallback,
               "har_qlike": loss["har"].mean()}
        lc = excl_covid(loss)
        for g, col in [("sector", "har_sector"), ("corr", "har_corr")]:
            _, _, pv = dm_test(daily(loss, col).to_numpy(), daily(loss, "har").to_numpy())
            rec |= {f"{g}_qlike": loss[col].mean(), f"{g}_gain": rel(loss[col], loss["har"]),
                    f"{g}_gain_p": pv, f"{g}_gain_exCOVID": rel(lc[col], lc["har"])}
        records.append(rec)
        print(f"{v['name']:24s} {time.time() - t0:5.0f}s  sector gain {rec['sector_gain']:+.1%}  "
              f"corr gain {rec['corr_gain']:+.1%}  rows {len(loss):,}", flush=True)

    # variant vs default, where the target and rows are comparable
    base = frames["default"]
    for rec, v in zip(records, VARIANTS):
        same_target = {**DEFAULTS, **v}["winsor_q"] == WINSOR_Q
        for g, col in [("sector", "har_sector"), ("corr", "har_corr")]:
            if v["name"] == "default" or not same_target:
                continue
            chg, p, chg_cov, n = compare(frames[v["name"]], base, col)
            rec |= {f"{g}_vs_default": chg, f"{g}_vs_default_p": p, f"{g}_vs_default_exCOVID": chg_cov,
                    "rows_in_common": n}
            rec[f"{g}_replaces_default"] = bool(chg < -0.01 and p < 0.05 and chg_cov < -0.005)
        if same_target and v["name"] != "default":
            rec["har_vs_default"] = compare(frames[v["name"]], base, "har")[0]

    res = pd.DataFrame(records)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "sensitivity.csv", index=False, float_format=bl.CSV_FMT)
    pd.set_option("display.width", 250)
    pct = lambda c: res[c].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
    show = pd.DataFrame({"variant": res["variant"], "rows": res["val_rows"],
                         "sec gain": pct("sector_gain"), "sec exCOV": pct("sector_gain_exCOVID"),
                         "sec vs def": pct("sector_vs_default") if "sector_vs_default" in res else "",
                         "sec p": res.get("sector_vs_default_p", pd.Series(dtype=float)).map(lambda x: f"{x:.3f}"),
                         "sec repl": res.get("sector_replaces_default", ""),
                         "cor gain": pct("corr_gain"), "cor exCOV": pct("corr_gain_exCOVID"),
                         "cor vs def": pct("corr_vs_default") if "corr_vs_default" in res else "",
                         "cor p": res.get("corr_vs_default_p", pd.Series(dtype=float)).map(lambda x: f"{x:.3f}"),
                         "cor repl": res.get("corr_replaces_default", "")})
    print("\n" + show.to_string(index=False))
    print("\nHAR itself vs default HAR (same target): ")
    print(res[["variant", "har_qlike", "har_vs_default"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
