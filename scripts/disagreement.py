"""Inspect individual records (proposal 4c.2): on the stock-days where the with-neighbor model and HAR
disagree most, who was right, and what did the neighbors do the day before?

  python scripts/disagreement.py            # test years 2022-2025 (needs baselines.py --include-test)
  python scripts/disagreement.py --split val

Pair: har_sector (HAR + sector peers' averaged volatility) vs har (own history only), both linear, so the
disagreement is exactly the sector peers' contribution. Forecasts are compared as log volatility with each
model's smearing correction applied (log-vol + 0.5 * log(smear)), the level its QLIKE is scored at.

  disagreement = har_sector forecast - har forecast   (positive = peers say "more volatile tomorrow")

Outputs (results/):
  disagreement_deciles{tag}.csv  disagreement in 10 equal-count bins: QLIKE of both models, share of rows
                                 where the peer model was closer, mean disagreement and actual surprise
  disagreement_top{tag}.csv      the TOP_N largest disagreements in each direction, with the 3 sector peers
                                 whose volatility jumped most on the forecast day (log-vol minus their 22-day mean)
  disagreement_days{tag}.csv     the days with the most large disagreements (top 1% of |disagreement|)
Descriptive, no decision rule.
"""
import argparse

import numpy as np
import pandas as pd

from config import FEATURES_DIR, PREDICTIONS_DIR, RESULTS_DIR
from metrics import qlike_loss

TOP_N, N_PEERS = 15, 3
CSV_FMT = "%.6g"


def load(name, split):
    tag = "_with_test" if split == "test" else ""
    p = pd.read_parquet(PREDICTIONS_DIR / f"{name}{tag}.parquet")
    p = p[p["split"] == split].copy()
    p["f"] = p["yhat"] + 0.5 * np.log(p["smear"])            # smeared log-vol forecast
    return p[["ticker", "date", "y_date", "y", "f"]]


def pair(split):
    a, b = load("har_sector", split), load("har", split)
    j = a.merge(b[["ticker", "date", "f"]], on=["ticker", "date"], suffixes=("_peer", "_own"))
    j["disagreement"] = j["f_peer"] - j["f_own"]
    j["err_peer"], j["err_own"] = (j["y"] - j["f_peer"]).abs(), (j["y"] - j["f_own"]).abs()
    j["peer_closer"] = j["err_peer"] < j["err_own"]
    v = np.exp(2 * j["y"])
    j["qlike_peer"], j["qlike_own"] = qlike_loss(v, np.exp(2 * j["f_peer"])), qlike_loss(v, np.exp(2 * j["f_own"]))
    j["surprise"] = j["y"] - j["f_own"]                      # what HAR missed
    return j


def deciles(j):
    j = j.assign(bin=pd.qcut(j["disagreement"], 10, labels=False) + 1)
    g = j.groupby("bin")
    out = pd.DataFrame({"rows": g.size(), "disagreement_mean": g["disagreement"].mean(),
                        "disagreement_min": g["disagreement"].min(), "disagreement_max": g["disagreement"].max(),
                        "har_missed_by": g["surprise"].mean(), "peer_closer_share": g["peer_closer"].mean(),
                        "qlike_har": g["qlike_own"].mean(), "qlike_har_sector": g["qlike_peer"].mean()})
    out["qlike_change"] = out["qlike_har_sector"] / out["qlike_har"] - 1
    return out.reset_index()


def top_records(j, panel):
    info = panel[["ticker", "date", "sector", "lv", "har_22", "ret"]].copy()
    info["jump"] = info["lv"] - info["har_22"]                # today's log-vol vs its own 22-day mean
    meta = j.merge(info[["ticker", "date", "sector", "jump", "ret"]], on=["ticker", "date"], how="left")
    picks = pd.concat([meta.nlargest(TOP_N, "disagreement").assign(direction="peers say higher"),
                       meta.nsmallest(TOP_N, "disagreement").assign(direction="peers say lower")])
    by_day = {d: g for d, g in info[info["date"].isin(picks["date"].unique())].groupby("date")}
    rows = []
    for r in picks.itertuples():
        peers = by_day[r.date]
        peers = peers[(peers["sector"] == r.sector) & (peers["ticker"] != r.ticker)].nlargest(N_PEERS, "jump") \
            if r.direction == "peers say higher" else \
            peers[(peers["sector"] == r.sector) & (peers["ticker"] != r.ticker)].nsmallest(N_PEERS, "jump")
        rows.append({"direction": r.direction, "ticker": r.ticker, "sector": r.sector,
                     "forecast_day": r.date.date(), "predicted_day": r.y_date.date(),
                     "own_jump_today": r.jump, "own_return_today": r.ret,
                     "har": r.f_own, "har_sector": r.f_peer, "actual": r.y, "disagreement": r.disagreement,
                     "closer": "har_sector" if r.peer_closer else "har",
                     "peers_moving": ", ".join(f"{t} {jv:+.2f}" for t, jv in zip(peers["ticker"], peers["jump"]))})
    return pd.DataFrame(rows)


def busiest_days(j, panel):
    cut = j["disagreement"].abs().quantile(0.99)
    big = j[j["disagreement"].abs() >= cut]
    sec = panel[["ticker", "date", "sector"]]
    big = big.merge(sec, on=["ticker", "date"], how="left")
    g = big.groupby("date")
    out = pd.DataFrame({"large_disagreements": g.size(),
                        "mean_disagreement": g["disagreement"].mean(),
                        "peer_closer_share": g["peer_closer"].mean(),
                        "main_sector": g["sector"].agg(lambda s: s.value_counts().index[0]),
                        "main_sector_share": g["sector"].agg(lambda s: s.value_counts().iloc[0] / len(s))})
    return out.sort_values("large_disagreements", ascending=False).head(20).reset_index()


def main(split):
    j = pair(split)
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet", columns=["ticker", "date", "sector", "lv", "har_22", "ret"])
    dec, top, days = deciles(j), top_records(j, panel), busiest_days(j, panel)
    tag = "_test" if split == "test" else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    dec.to_csv(RESULTS_DIR / f"disagreement_deciles{tag}.csv", index=False, float_format=CSV_FMT)
    top.to_csv(RESULTS_DIR / f"disagreement_top{tag}.csv", index=False, float_format=CSV_FMT)
    days.to_csv(RESULTS_DIR / f"disagreement_days{tag}.csv", index=False, float_format=CSV_FMT)

    pd.set_option("display.width", 250)
    print(f"har_sector vs har, {split}: {len(j):,} stock-days; peer model closer on {j['peer_closer'].mean():.1%}")
    print("\nBY DISAGREEMENT DECILE (1 = peers say much lower, 10 = peers say much higher)")
    print(dec.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDAYS WITH THE MOST LARGE DISAGREEMENTS (top 1%)")
    print(days.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nLARGEST DISAGREEMENTS")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    main(ap.parse_args().split)
