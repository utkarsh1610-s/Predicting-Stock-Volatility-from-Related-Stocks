"""Stage 7: GNNHAR with one graph layer, on the real graphs and on their controls.

  python scripts/gnnhar.py                 # fit on train, score validation (test years untouched)
  python scripts/gnnhar.py --include-test  # ALSO score the test years (once, at the end, with all models)

MODEL (after Zhang et al. 2025, Eq. 7, one layer; see DECISIONS.md #50)
For stock i on day t, with h_i = its own har_1/5/22 and m_i = the mean of its neighbors' har_* that day:

    z_i = b0 + beta . h_i  +  w . ReLU(W [h_i, m_i] + c)        forecast of next-day log volatility

The first part is HAR. The second is a one-layer graph convolution: m_i is the graph's message
(row-normalized adjacency times every stock's features, i.e. the neighbor mean), passed with the
stock's own features through one shared hidden layer. With mean aggregation the paper's symmetric
normalization is identical for the sector graph (every sector is a clique) and close for the
correlation graph (out-degree 10 for every stock). Because a day's graph layer reduces to the
neighbor means, the inputs are the same neighbor features the linear and tree models use.

Models (each fit with 3 seeds; the reported forecast is the mean of the seeds' log-vol forecasts):
  nn_own              the network without any graph input (h_i only). Separates nonlinearity in the
                      stock's own history from information coming through the graph. Neither
                      reference paper has this control.
  gnn_sector, gnn_corr  the real graphs
  gnn_market          every other stock is a neighbor (the placebo's market average)
  gnn_market+sector, gnn_market+corr   the graph's message on top of the market message
  gnn_rand_sector_s, gnn_rand_corr_s   random twins of the graphs (same neighbor counts and monthly
                      availability as the real graph; scripts/placebo.py), one twin per seed
Linear references, trained with the same QLIKE loss: har_q, har_sector_q, har_corr_q.

TRAINING
  * Loss is QLIKE on the variance forecast exp(2z), as for har_q, so no smearing is needed.
  * Inputs are standardized with means and standard deviations from the training years only.
  * The HAR part starts at the QLIKE-trained linear fit; the hidden layer starts near zero, so every
    network starts at har_q and can only move away from it if the training loss improves.
  * Epochs are chosen by early stopping on the LAST training year (2019) after fitting on 2010-2018.
    The model is then refit on all training years for that many epochs. The validation years are
    never used to fit or tune anything, so they stay a clean out-of-sample score (unlike the trees).
  * Fixed architecture: 16 hidden units, Adam (lr 1e-3), batch 8192, at most 30 epochs, patience 3.
    Not tuned.

READING THE RESULT (written before the script was run on the data):
  Graph-specific nonlinear spillover: for a graph G, gnn_G beats every random twin gnn_rand_G_s by more
    than 1% QLIKE with DM p < 0.05, AND gnn_market+G beats gnn_market by more than 0.5% with p < 0.05
    and is still better with Feb-Apr 2020 removed.
  Market-wide only: for both graphs, gnn_market+G improves on gnn_market by less than 0.5%.
  Anything else is reported as mixed.
  Nonlinearity is descriptive, no rule: nn_own vs har_q (own history), gnn_G vs har_G_q (same inputs).

Outputs: results/gnnhar.csv (comparisons), results/gnnhar_models.csv (per model and seed),
data/predictions/gnn_*.parquet (seed-averaged forecasts, for the per-group analysis).
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

import baselines as bl
from config import PREDICTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, SPLITS
from features import compute_features, panel_from_rows
from graphs import corr_edges, neighbor_features, return_matrix, sector_edges
from metrics import mse_logvol, qlike, qlike_loss
from placebo import MKT, RCOR, RSEC, market_edges, random_corr_edges, random_sector_edges, renamed
from sensitivity import compare

SEEDS = range(3)
HIDDEN, LR, BATCH, MAX_EPOCHS, PATIENCE = 16, 1e-3, 8192, 30, 3
ES_START = "2019-01-01"     # last training year, used only to choose the number of epochs
NETS = {"nn_own": [],
        "gnn_sector": bl.SEC, "gnn_corr": bl.COR, "gnn_market": MKT,
        "gnn_market+sector": MKT + bl.SEC, "gnn_market+corr": MKT + bl.COR}
LINEAR = {"har_q": [], "har_sector_q": bl.SEC, "har_corr_q": bl.COR}


class GNNHAR(nn.Module):
    """HAR on the stock's own inputs plus one hidden ReLU layer over [own, neighbor messages]."""

    def __init__(self, n_own, n_msg, hidden=HIDDEN):
        super().__init__()
        self.n_own = n_own
        self.har = nn.Linear(n_own, 1)
        self.hidden = nn.Linear(n_own + n_msg, hidden)
        self.out = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, x):
        linear = self.har(x[:, :self.n_own]).squeeze(1)
        return linear + self.out(torch.relu(self.hidden(x))).squeeze(1)


def qlike_objective(y, z):
    """Mean QLIKE of variance forecast exp(2z) against exp(2y), minus its constant.

    v/h - log(v/h) - 1 with v = exp(2y), h = exp(2z) equals v*exp(-2z) + 2z - (2y + 1); the
    dropped term does not depend on the model.
    """
    return torch.mean(torch.exp(2 * y - 2 * z) + 2 * z)


def standardize(x, train):
    mu, sd = x[train].mean(axis=0), x[train].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd


def _fit(x, y, init_har, seed, epochs, x_stop=None, y_stop=None):
    """Train one network. With x_stop, stop when its loss has not improved for PATIENCE epochs and
    return (model, best epoch count); otherwise train exactly `epochs` epochs."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_own = len(init_har) - 1
    model = GNNHAR(n_own, x.shape[1] - n_own)
    with torch.no_grad():
        model.har.bias.fill_(float(init_har[0]))
        model.har.weight.copy_(torch.tensor(init_har[1:], dtype=torch.float32).unsqueeze(0))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    best, best_epoch, best_state, wait = np.inf, 0, None, 0
    if x_stop is not None:
        if len(x_stop) == 0:
            raise ValueError("no early-stopping rows: the training split must include the year from ES_START")
        xs, ys = torch.tensor(x_stop, dtype=torch.float32), torch.tensor(y_stop, dtype=torch.float32)
        with torch.no_grad():
            best = float(qlike_objective(ys, model(xs)))           # epoch 0: the linear starting point
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(BATCH):
            opt.zero_grad()
            qlike_objective(yt[idx], model(xt[idx])).backward()
            opt.step()
        if x_stop is None:
            continue
        model.eval()
        with torch.no_grad():
            loss = float(qlike_objective(ys, model(xs)))
        if loss < best - 1e-7:
            best, best_epoch, wait = loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if x_stop is not None:
        model.load_state_dict(best_state)
        return model, best_epoch
    return model, epochs


def fit_network(df, msg_cols, seed):
    """Fit on the training split; returns (log-vol forecast for every row of df, epochs used).

    Only training rows (y_date in the train split) influence the result: standardization,
    the starting linear fit, early stopping (on 2019) and the final fit all use them alone.
    """
    train = (df["split"] == "train").to_numpy()
    cols = bl.HAR + msg_cols
    x = standardize(df[cols].to_numpy(), train)
    y = df["y"].to_numpy()
    n_own = len(bl.HAR)
    init = bl.fit_qlike_linear(x[train, :n_own], y[train])
    early = train & (df["y_date"] < ES_START).to_numpy()
    stop = train & (df["y_date"] >= ES_START).to_numpy()
    _, epochs = _fit(x[early], y[early], init, seed, MAX_EPOCHS, x[stop], y[stop])
    epochs = max(epochs, 1)
    model, _ = _fit(x[train], y[train], init, seed, epochs)
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32)).numpy().astype(float)
    return z, epochs


def fit_linear(df, msg_cols):
    train = (df["split"] == "train").to_numpy()
    x = df[bl.HAR + msg_cols].to_numpy()
    return bl.predict_ols(bl.fit_qlike_linear(x[train], df["y"].to_numpy()[train]), x)


def row_losses(df, z, split):
    m = (df["split"] == split).to_numpy()
    out = df.loc[m, ["ticker", "y_date"]].reset_index(drop=True)
    out["loss"] = qlike_loss(np.exp(2 * df["y"].to_numpy()[m]), np.exp(2 * z[m]))
    return out


def model_row(name, seed, df, z, epochs, split):
    m = (df["split"] == split).to_numpy()
    y = df["y"].to_numpy()
    return {"model": name, "seed": seed, "split": split, "n": int(m.sum()), "epochs": epochs,
            "qlike": qlike(y[m], z[m]), "mse_logvol": mse_logvol(y[m], z[m])}


def build_sample(include_test):
    """Panel plus real, market and random-twin neighbor features, on the placebo's base rows."""
    stocks = pd.read_parquet(PROCESSED_DIR / "stocks.parquet")
    vix = pd.read_parquet(PROCESSED_DIR / "vix.parquet")
    calendar = pd.DatetimeIndex(sorted(vix["date"]))
    rows, _ = compute_features(stocks, vix)
    panel = panel_from_rows(rows)
    sec = sector_edges(stocks)
    cor = corr_edges(return_matrix(stocks, calendar))
    nbr = neighbor_features(rows, sec, cor)
    mkt = renamed(neighbor_features(rows, market_edges(stocks["ticker"].unique()), cor), "sec", "mkt")
    df = panel.merge(nbr, on=["ticker", "date"], how="left").merge(mkt, on=["ticker", "date"], how="left")
    splits = ["train", "val"] + (["test"] if include_test else [])
    df = df[df["split"].isin(splits)]
    df = df.dropna(subset=bl.HAR + bl.SEC + bl.COR + MKT).reset_index(drop=True)
    twins = {}
    for s in SEEDS:
        rn = neighbor_features(rows, random_sector_edges(sec, s), random_corr_edges(cor, s))
        twins[s] = (df[["ticker", "date"]]
                    .merge(renamed(rn, "sec", "rsec"), on=["ticker", "date"], how="left", validate="1:1")
                    .merge(renamed(rn, "cor", "rcor"), on=["ticker", "date"], how="left", validate="1:1"))
        assert (twins[s][["ticker", "date"]].to_numpy() == df[["ticker", "date"]].to_numpy()).all()
    return df, twins


def main(include_test):
    df, twins = build_sample(include_test)
    split = "test" if include_test else "val"
    print(f"Sample: {len(df):,} rows; scoring {split}", flush=True)

    losses, rows, preds = {}, [], {}
    for name, cols in LINEAR.items():
        z = fit_linear(df, cols)
        losses[name] = row_losses(df, z, split)
        rows.append(model_row(name, -1, df, z, 0, split))

    for name, cols in NETS.items():
        zs = []
        for s in SEEDS:
            z, ep = fit_network(df, cols, s)
            zs.append(z)
            rows.append(model_row(name, s, df, z, ep, split))
            print(f"  {name:18s} seed {s}: {ep} epochs, {split} QLIKE {rows[-1]['qlike']:.4f}", flush=True)
        preds[name] = np.mean(zs, axis=0)
        losses[name] = row_losses(df, preds[name], split)
        rows.append(model_row(name, "mean", df, preds[name], 0, split))

    for kind, cols in [("rand_sector", RSEC), ("rand_corr", RCOR)]:
        for s in SEEDS:
            d = pd.concat([df, twins[s][cols]], axis=1).dropna(subset=cols).reset_index(drop=True)
            z, ep = fit_network(d, cols, s)
            name = f"gnn_{kind}_s{s}"
            losses[name] = row_losses(d, z, split)
            rows.append(model_row(name, s, d, z, ep, split))
            print(f"  {name:18s}: {ep} epochs, {split} QLIKE {rows[-1]['qlike']:.4f}", flush=True)

    out = []

    def add(a, b):
        chg, p, chg_cov, n = compare(losses[a], losses[b], "loss")
        out.append({"comparison": f"{a} vs {b}", "model": a, "versus": b, "qlike_change": chg,
                    "dm_p": p, "qlike_change_exCOVID": chg_cov, "rows": n})

    add("nn_own", "har_q")
    for g in ["sector", "corr"]:
        add(f"gnn_{g}", f"har_{g}_q")
        add(f"gnn_{g}", "nn_own")
        add(f"gnn_{g}", "gnn_market")
        add(f"gnn_market+{g}", "gnn_market")
        for s in SEEDS:
            add(f"gnn_{g}", f"gnn_rand_{g}_s{s}")
    add("gnn_market", "nn_own")

    RESULTS_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_with_test" if include_test else ""
    res = pd.DataFrame(out)
    res.to_csv(RESULTS_DIR / f"gnnhar{tag}.csv", index=False, float_format=bl.CSV_FMT)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"gnnhar_models{tag}.csv", index=False, float_format=bl.CSV_FMT)
    for name, z in preds.items():
        p = df[["ticker", "date", "y_date", "split", "y"]].copy()
        p["yhat"], p["smear"] = z, 1.0
        p.to_parquet(PREDICTIONS_DIR / f"{name.replace('+', '_plus_')}{tag}.parquet", index=False)

    print(f"\nCOMPARISONS ({split}, QLIKE; negative = first model better)")
    for _, r in res.iterrows():
        print(f"  {r.comparison:34s} {r.qlike_change:+.2%}  p={r.dm_p:.3g}  exCOVID {r.qlike_change_exCOVID:+.2%}")
    print("\nVERDICT (rule in the docstring)")
    for g in ["sector", "corr"]:
        twin = res[res["comparison"].str.fullmatch(f"gnn_{g} vs gnn_rand_{g}_s\\d")]
        beyond = res[res["comparison"] == f"gnn_market+{g} vs gnn_market"].iloc[0]
        specific = (bool(((twin.qlike_change < -0.01) & (twin.dm_p < 0.05)).all())
                    and beyond.qlike_change < -0.005 and beyond.dm_p < 0.05 and beyond.qlike_change_exCOVID < 0)
        market_only = beyond.qlike_change > -0.005
        verdict = "graph-specific" if specific else "market-wide only" if market_only else "mixed"
        print(f"  {g}: {verdict}  (beyond market {beyond.qlike_change:+.2%}, p={beyond.dm_p:.3g}; "
              f"vs random twins {twin.qlike_change.mean():+.2%} mean, max p {twin.dm_p.max():.3g})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    main(ap.parse_args().include_test)
