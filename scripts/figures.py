"""Figures for the report and video, from files the pipeline already wrote (nothing is refit).

  python scripts/figures.py            # writes figures/fig01_*.png ... (1920 x 1080)

Needs: data/features/panel.parquet, data/graphs/*.parquet, data/predictions/har*_with_test.parquet and the
results/*.csv files from steps 5-9 and the clustering. Colors: the first three slots of a colorblind-checked
categorical palette (blue, orange, aqua), gray for context, a blue-gray-red diverging scale for signed values.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from config import FEATURES_DIR, GRAPHS_DIR, PREDICTIONS_DIR, RESULTS_DIR, SPLITS
from pathlib import Path

OUT = Path("figures")
R = RESULTS_DIR
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT = "#b5b3ac", "#e1e0d9"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE = "#fcfcfb"
DIVERGE = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a8302f"])
SIZE, DPI = (12, 6.75), 160

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def figure(title, subtitle, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=SIZE, **kw)
    fig.text(0.04, 0.955, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.895, subtitle, fontsize=13, color=INK2, va="top")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.12, wspace=0.25, hspace=0.45)
    return fig, axes


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=DPI)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%" if v else "0%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def ann_vol(logvol):
    """Daily log volatility -> annualized volatility in %."""
    return np.exp(logvol) * np.sqrt(252) * 100


# ---------------------------------------------------------------- 1. the data
def fig_timeline():
    p = pd.read_parquet(FEATURES_DIR / "panel.parquet", columns=["date", "lv"])
    v = ann_vol(p.groupby("date")["lv"].median()).rolling(5, min_periods=1).mean()
    fig, ax = figure("Sixteen years of daily volatility, 499 S&P 500 stocks",
                     "Median stock's volatility (annualized, 5-day average). Models learn on 2010-19, are checked on 2020-21, and scored once on 2022-25.")
    for (name, (a, b)), shade in zip(SPLITS.items(), ["#f3f2ee", "#e6eef9", "#fbe9e1"]):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=shade, zorder=0, lw=0)
        ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, 0.97,
                {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}[name], transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK2, fontsize=11, fontweight="bold")
    ax.plot(v.index, v.values, color=BLUE, lw=1.6)
    for d, label, off, ha in [("2020-03-20", "COVID crash", (14, -4), "left"), ("2022-05-10", "2022 bear market", (0, 80), "center"),
                              ("2023-03-13", "Bank failures", (45, 40), "left"), ("2025-04-08", "Tariff sell-off", (-14, -4), "right")]:
        d = pd.Timestamp(d)
        win = v.loc[d - pd.Timedelta(days=12): d + pd.Timedelta(days=12)]
        y, d = win.max(), win.idxmax()
        ax.annotate(label, (d, y), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=11, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=2, shrinkB=2))
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_ylim(0, v.max() * 1.25)
    ax.margins(x=0.01)
    save(fig, "fig01_volatility_timeline")


# ---------------------------------------------------------------- 2. the graphs
def fig_graphs(month="2023-03-01", per=10):
    sec = pd.read_parquet(GRAPHS_DIR / "sector_edges.parquet")
    cor = pd.read_parquet(GRAPHS_DIR / "corr_edges.parquet")
    cor = cor[cor["month"] == pd.Timestamp(month)]
    panel = pd.read_parquet(FEATURES_DIR / "panel.parquet", columns=["ticker", "sector"]).drop_duplicates("ticker")
    groups = {"Financials": BLUE, "Utilities": ORANGE, "Energy": AQUA}
    nodes = []
    for s in groups:
        cand = panel.loc[panel["sector"] == s, "ticker"]
        cand = cand[cand.isin(cor["ticker"])].sort_values()
        nodes += list(cand.sample(per, random_state=0).sort_values())      # random, so the picture is not cherry-picked
    sector_of = panel.set_index("ticker").loc[nodes, "sector"]
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False) + np.pi / 2
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(nodes, ang)}
    keep = set(nodes)

    fig, axes = figure("Two ways to say which stocks are related",
                       f"30 random stocks from three sectors. Left: linked if in the same sector (fixed). Right: linked if among each other's\n10 most correlated stocks over the past 60 days ({pd.Timestamp(month):%B %Y}; rebuilt every month).",
                       ncols=2)
    for ax, edges, title in [(axes[0], sec, "Sector graph"), (axes[1], cor, "Correlation graph")]:
        e = edges[edges["ticker"].isin(keep) & edges["neighbor"].isin(keep)]
        for a, b in zip(e["ticker"], e["neighbor"]):
            cross = sector_of[a] != sector_of[b]
            ax.plot(*zip(pos[a], pos[b]), color=INK2 if cross else GRAY, lw=1.1 if cross else 0.6,
                    alpha=0.9 if cross else 0.55, zorder=1)
        for t in nodes:
            ax.scatter(*pos[t], s=260, color=groups[sector_of[t]], edgecolor=SURFACE, linewidth=2, zorder=2)
            x, y = pos[t]
            ax.text(x * 1.16, y * 1.16, t, ha="center", va="center", fontsize=9, color=INK2)
        n_cross = int((e["ticker"].map(sector_of) != e["neighbor"].map(sector_of)).sum())
        ax.set_title(f"{title}   ({n_cross} links cross sectors)" if title.startswith("Corr") else title)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    fig.legend(handles=[Patch(color=c, label=s) for s, c in groups.items()] +
               [plt.Line2D([], [], color=INK2, lw=1.1, label="link across sectors")],
               loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11.5)
    fig.subplots_adjust(top=0.76, bottom=0.1)
    save(fig, "fig02_graphs")


# ---------------------------------------------------------------- 3. main result
def neighbor_gains():
    m = pd.read_csv(R / "trees_metrics_with_test.csv")
    dm = pd.read_csv(R / "trees_dm_with_test.csv")
    q = m.set_index(["model", "split"])["qlike"]
    rows = []
    for split in ["val", "test"]:
        for fam, base, models in [("Linear HAR", "har", ["har_sector", "har_corr"]),
                                  ("Trees (LightGBM)", "gbt", ["gbt_sector", "gbt_corr"])]:
            for mod, graph in zip(models, ["sector", "corr"]):
                p = dm[(dm["split"] == split) & (dm["model"] == mod) & (dm["vs"] == base)]["p_value"].iloc[0]
                rows.append((split, fam, graph, 100 * (q[mod, split] / q[base, split] - 1), p))
        g = pd.read_csv(R / ("gnnhar.csv" if split == "val" else "gnnhar_with_test.csv")).set_index("comparison")
        for graph in ["sector", "corr"]:
            r = g.loc[f"gnn_{graph} vs nn_own"]
            rows.append((split, "Neural net (GNNHAR)", graph, 100 * r["qlike_change"], r["dm_p"]))
    return pd.DataFrame(rows, columns=["split", "family", "graph", "change", "p"])


def fig_main():
    d = neighbor_gains()
    fams = ["Linear HAR", "Trees (LightGBM)", "Neural net (GNNHAR)"]
    fig, axes = figure("Adding related stocks makes every model better",
                       "Change in forecast error (QLIKE) from adding neighbor information; lower is better. * p<0.05, ** p<0.01 (Diebold-Mariano).",
                       ncols=2, sharey=True)
    x, w = np.arange(len(fams)), 0.36
    for ax, split, title in [(axes[0], "val", "Validation 2020-21 (includes COVID)"), (axes[1], "test", "Test 2022-25 (scored once)")]:
        for k, (graph, color, label) in enumerate([("sector", BLUE, "Sector peers"), ("corr", ORANGE, "Correlated stocks")]):
            s = d[(d["split"] == split) & (d["graph"] == graph)].set_index("family").loc[fams]
            xs = x + (k - 0.5) * (w + 0.03)
            ax.bar(xs, s["change"], w, color=color, label=label)
            for xi, (c, p) in zip(xs, zip(s["change"], s["p"])):
                ax.text(xi, c - 0.15, f"{c:.1f}%{stars(p)}", ha="center", va="top", fontsize=10.5, color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(x, fams); ax.set_title(title); ax.grid(axis="x", visible=False)
        pct_axis(ax)
    axes[0].set_ylabel("Change in QLIKE vs same model without neighbors")
    axes[0].set_ylim(min(d["change"]) * 1.3, 0.8)
    axes[1].legend(loc="lower right")
    save(fig, "fig03_main_result")


# ---------------------------------------------------------------- 4. placebo
def fig_placebo():
    fig, axes = figure("The twist: random stocks work as well as real neighbors",
                       "Neural net (GNNHAR): change in QLIKE vs the same network with no neighbors. The average of the whole market helps most.",
                       ncols=2, sharey=True)
    for ax, split, f, title in [(axes[0], "val", "gnnhar_models.csv", "Validation 2020-21"),
                                (axes[1], "test", "gnnhar_models_with_test.csv", "Test 2022-25")]:
        m = pd.read_csv(R / f)
        m = m[m["split"] == split]
        mean = m[m["seed"].astype(str) == "mean"].set_index("model")["qlike"]
        base = mean["nn_own"]
        rel = lambda v: 100 * (v / base - 1)
        twins = {g: m[m["model"].str.startswith(f"gnn_rand_{g}_s")]["qlike"] for g in ["sector", "corr"]}
        rows = [("Sector peers (real)", rel(mean["gnn_sector"]), BLUE, None),
                ("Random twin of sector graph", rel(twins["sector"].mean()), GRAY, (rel(twins["sector"].min()), rel(twins["sector"].max()))),
                ("Correlated stocks (real)", rel(mean["gnn_corr"]), BLUE, None),
                ("Random twin of correlation graph", rel(twins["corr"].mean()), GRAY, (rel(twins["corr"].min()), rel(twins["corr"].max()))),
                ("Whole market (all other stocks)", rel(mean["gnn_market"]), ORANGE, None)]
        y = np.arange(len(rows))[::-1]
        for yi, (label, v, c, rng) in zip(y, rows):
            ax.barh(yi, v, 0.6, color=c)
            if rng:
                ax.plot(rng, [yi, yi], color=INK2, lw=1.2)
                ax.plot([rng[0]] * 2, [yi - 0.12, yi + 0.12], color=INK2, lw=1.2)
                ax.plot([rng[1]] * 2, [yi - 0.12, yi + 0.12], color=INK2, lw=1.2)
            left = min(v, rng[0]) if rng else v
            ax.text(left - 0.12, yi, f"{v:.1f}%", ha="right", va="center", fontsize=10.5, color=INK2)
        ax.set_yticks(y, [r[0] for r in rows]); ax.axvline(0, color="#c3c2b7", lw=0.8)
        ax.set_title(title); ax.grid(axis="y", visible=False)
        ax.set_xlim(min(r[1] for r in rows) * 1.3, 0.3); pct_axis(ax, "x")
    fig.subplots_adjust(left=0.22, wspace=0.08)
    axes[0].text(0, -0.13, "Random twins: same number of neighbors per stock, picked at random. Bar = mean of 10, whisker = range.",
                 transform=axes[0].transAxes, fontsize=10, color=MUTED)
    save(fig, "fig04_placebo")


# ---------------------------------------------------------------- 5. seeds
SEED3 = {"sector": -0.84, "corr": -0.92}      # first validation run, 3 seeds (DECISIONS.md, GNNHAR finding)


def fig_seeds():
    g = pd.read_csv(R / "gnnhar.csv").set_index("comparison")
    m = pd.read_csv(R / "gnnhar_models.csv")
    fig, axes = figure("Checking for luck: 10 training runs instead of 3",
                       "Left: extra gain from adding a graph on top of the market average. Right: each training run's error (validation).",
                       ncols=2, gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    x, w = np.arange(2), 0.36
    s10 = [100 * g.loc[f"gnn_market+{k} vs gnn_market", "qlike_change"] for k in ["sector", "corr"]]
    for k, (vals, color, label) in enumerate([([SEED3["sector"], SEED3["corr"]], GRAY, "3 runs (first try)"),
                                              (s10, BLUE, "10 runs")]):
        xs = x + (k - 0.5) * (w + 0.03)
        ax.bar(xs, vals, w, color=color, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}%", ha="center", va="bottom", fontsize=10.5, color="#ffffff", fontweight="bold")
    ax.axhline(-0.5, color=INK2, lw=1)
    ax.text(1.88, -0.52, "rule: 'market-wide only'\nif smaller than 0.5%", ha="right", va="top", fontsize=10, color=INK2)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(x, ["+ sector peers", "+ correlated stocks"]); ax.set_ylim(-1.15, 0.1); ax.set_xlim(-0.6, 1.9)
    ax.set_title("Gain beyond the market shrank"); ax.grid(axis="x", visible=False); pct_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.legend(loc="lower left")

    ax = axes[1]
    models = [("gnn_market", "Market"), ("gnn_market+sector", "Market + sector"), ("gnn_market+corr", "Market + correlated")]
    per = m[(m["split"] == "val") & m["seed"].astype(str).str.fullmatch(r"\d+")]
    for i, (mod, label) in enumerate(models):
        v = per[per["model"] == mod]["qlike"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=64, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        mean = m[(m["model"] == mod) & (m["seed"].astype(str) == "mean") & (m["split"] == "val")]["qlike"].iloc[0]
        ax.plot([i - 0.25, i + 0.25], [mean, mean], color=ORANGE, lw=2.5, zorder=3)
    ax.plot([], [], color=ORANGE, lw=2.5, label="forecast averaged over 10 runs")
    ax.scatter([], [], s=64, color=BLUE, label="one training run")
    ax.set_xticks(range(3), [l for _, l in models]); ax.set_ylabel("QLIKE (validation)")
    ax.set_title("Single runs differ more than the effect we test"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=10.5)
    save(fig, "fig05_seed_check")


# ---------------------------------------------------------------- 6. learning curves
def fig_curves():
    c = pd.read_csv(R / "trees_curves.csv")
    params = pd.read_csv(R / "trees_params.csv").set_index("model")
    fig, axes = figure("No sign of overfitting in the trees",
                       "Training vs validation error as boosting rounds are added (MSE on log volatility). Early stopping picks the dashed round.",
                       ncols=3, sharey=True)
    for ax, (mod, title) in zip(axes, [("gbt", "Trees, own history"), ("gbt_sector", "+ sector peers"), ("gbt_corr", "+ correlated stocks")]):
        d = c[c["model"] == mod]
        ax.plot(d["round"], d["train_mse"], color=BLUE, lw=2, label="training (2010-19)")
        ax.plot(d["round"], d["val_mse"], color=ORANGE, lw=2, label="validation (2020-21)")
        best = int(params.loc[mod, "best_iteration"])
        ax.axvline(best, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(best, ax.get_ylim()[1] if False else d["train_mse"].max(), f" round {best}", fontsize=10, color=INK2, va="top")
        ax.set_title(title); ax.set_xlabel("Boosting round")
    axes[0].set_ylabel("MSE (log volatility)")
    axes[0].legend(loc="upper right")
    save(fig, "fig06_learning_curves")


# ---------------------------------------------------------------- 7. stress
def fig_stress():
    fams = [("har_sector", "Linear HAR"), ("gbt_sector", "Trees"), ("gnn_sector", "Neural net"), ("gnn_market", "Market-only network")]
    fig, axes = figure("Neighbors help most when the market is scared",
                       "Change in QLIKE from adding sector peers: calm days (VIX at most 25) vs stressed days (VIX above 25).\nThe market-only network is shown for comparison.",
                       ncols=2, sharey=True)
    x, w = np.arange(len(fams)), 0.36
    for ax, f, title in [(axes[0], "groups.csv", "Validation 2020-21"), (axes[1], "groups_test.csv", "Test 2022-25")]:
        g = pd.read_csv(R / f)
        g = g[g["grouping"] == "stress"]
        for k, (grp, color, label) in enumerate([("calm (VIX<=25)", BLUE, "Calm days"), ("stressed (VIX>25)", ORANGE, "Stressed days")]):
            vals = [100 * g[(g["model"] == m) & (g["group"] == grp)]["qlike_change"].iloc[0] for m, _ in fams]
            days = g[g["group"] == grp]["days"].iloc[0]
            xs = x + (k - 0.5) * (w + 0.03)
            ax.bar(xs, vals, w, color=color, label=f"{label} ({days} days)")
            for xi, v in zip(xs, vals):
                ax.text(xi, v - 0.3, f"{v:.1f}%", ha="center", va="top", fontsize=10, color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(x, [l for _, l in fams], fontsize=11); ax.set_title(title); ax.grid(axis="x", visible=False)
        ax.legend(loc="lower left", fontsize=10.5); pct_axis(ax)
    axes[0].set_ylabel("Change in QLIKE with neighbors")
    axes[0].set_ylim(-20, 1)
    fig.subplots_adjust(top=0.76, left=0.09)
    fig.text(0.04, 0.03, "The calm-vs-stressed difference is not statistically significant (p = 0.05-0.07 validation, 0.3-0.45 test): stressed days come in a few long episodes.",
             fontsize=10, color=MUTED)
    save(fig, "fig07_calm_vs_stressed")


# ---------------------------------------------------------------- 8. sectors
def fig_sectors():
    g = pd.read_csv(R / "groups_test.csv")
    g = g[g["grouping"] == "sector"]
    cols = [("har_sector", "HAR\n+ sector"), ("gbt_sector", "Trees\n+ sector"), ("gnn_sector", "Neural net\n+ sector"),
            ("gnn_market", "Neural net\n+ market"), ("gnn_market+sector", "Sector on top\nof market")]
    t = g.pivot(index="group", columns="model", values="qlike_change")[[c for c, _ in cols]] * 100
    sig = g.pivot(index="group", columns="model", values="dm_p")[[c for c, _ in cols]]
    t = t.loc[t["har_sector"].sort_values().index]
    fig, ax = figure("Which sectors gain most? (test years)",
                     "Change in QLIKE by sector. Blue = forecasts improve. Financials and Real Estate gain most; tech least. Dot = DM p<0.05.")
    lim = np.nanmax(np.abs(t.values))
    im = ax.imshow(t.values, cmap=DIVERGE, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:+.1f}%" + (" •" if sig.loc[t.index[i]].iloc[j] < 0.05 else ""), ha="center", va="center",
                    fontsize=10.5, color="#ffffff" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(len(cols)), [l for _, l in cols]); ax.set_yticks(range(len(t)), t.index)
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.76, bottom=0.04)
    save(fig, "fig08_sector_heatmap")


# ---------------------------------------------------------------- 9. disagreement
def fig_disagreement():
    d = pd.read_csv(R / "disagreement_deciles_test.csv")
    fig, axes = figure("The gain comes from the days the models disagree most",
                       "Test stock-days split into 10 equal groups by how much HAR + sector peers disagrees with plain HAR.",
                       nrows=1, ncols=2)
    fig.subplots_adjust(bottom=0.17)
    ax = axes[0]
    colors = [BLUE if abs(v) > 0.015 else GRAY for v in d["qlike_change"]]
    ax.bar(d["bin"], 100 * d["qlike_change"], 0.7, color=colors)
    for b, v in zip(d["bin"], 100 * d["qlike_change"]):
        if abs(v) > 1.5:
            ax.text(b, v - 0.3, f"{v:.1f}%", ha="center", va="top", fontsize=10.5, color=INK2)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_title("Change in QLIKE with sector peers"); pct_axis(ax); ax.grid(axis="x", visible=False)
    ax.set_xticks(range(1, 11)); ax.set_ylim(-14, 1)
    ax.set_xlabel("peers say much LOWER  <--   group   -->  peers say much HIGHER")
    ax = axes[1]
    ax.bar(d["bin"], 100 * d["peer_closer_share"], 0.7, color=[AQUA if v > 0.5 else GRAY for v in d["peer_closer_share"]])
    ax.axhline(50, color=INK2, lw=1)
    ax.text(10.5, 51, "coin flip", ha="right", va="bottom", fontsize=10, color=INK2)
    ax.set_title("How often the peer model is closer to the truth"); ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xticks(range(1, 11)); ax.grid(axis="x", visible=False)
    ax.set_xlabel("peers say much LOWER  <--   group   -->  peers say much HIGHER")
    fig.text(0.04, 0.025, "When peers warn of higher volatility they are usually wrong (right panel) but save a lot when a real spike comes (left panel): QLIKE punishes missed risk.",
             fontsize=10, color=MUTED)
    save(fig, "fig09_disagreement")


# ---------------------------------------------------------------- 10. clustering
def fig_clusters():
    import stock_clusters as sc
    from features import compute_features
    stocks = pd.read_parquet(Path("data/processed/stocks.parquet"))
    vix = pd.read_parquet(Path("data/processed/vix.parquet"))
    rows, _ = compute_features(stocks, vix)
    a, b = SPLITS["train"]
    rows = rows[(rows["date"] >= a) & (rows["date"] <= b)]
    lv = rows.pivot(index="date", columns="ticker", values="lv")
    lv = lv.loc[:, lv.notna().sum() >= sc.MIN_DAYS]
    corr = sc.market_residuals(lv).corr(min_periods=sc.MIN_OVERLAP).to_numpy(copy=True)
    corr[np.isnan(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    sectors = stocks.drop_duplicates("ticker").set_index("ticker").loc[lv.columns, "sector"].to_numpy()[order]
    c = corr[np.ix_(order, order)].copy()
    np.fill_diagonal(c, np.nan)
    q = pd.read_csv(R / "cluster_quality.csv")
    ari = q[(q["labels"] == "clusters") & (q["k"] == 11)]["ari_vs_sector"].iloc[0]

    fig = plt.figure(figsize=SIZE)
    fig.text(0.04, 0.955, "Do stocks group by sector once the market is removed?", fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.895, f"Correlation of 471 stocks' volatility after removing the market (2010-19), ordered by hierarchical clustering.\n"
             f"Mostly weak (pale). A few real blocks: Utilities, banks, Energy. Agreement of 11 clusters with sectors: ARI {ari:.2f} (1 = identical).",
             fontsize=12.5, color=INK2, va="top")
    ax = fig.add_axes([0.30, 0.04, 0.42, 0.74])
    strip = fig.add_axes([0.725, 0.04, 0.02, 0.74])
    im = ax.imshow(c, cmap=DIVERGE, norm=TwoSlopeNorm(0, -0.4, 0.4), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    hl = {"Utilities": ORANGE, "Financials": BLUE, "Energy": AQUA}
    col = np.array([matplotlib.colors.to_rgb(hl.get(s, LIGHT)) for s in sectors])[:, None, :]
    strip.imshow(col, aspect="auto", interpolation="nearest"); strip.axis("off")
    strip.set_title("sector", fontsize=10, color=INK2, loc="center")
    cax = fig.add_axes([0.80, 0.30, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.set_label("residual correlation", color=INK2)
    fig.legend(handles=[Patch(color=v, label=k) for k, v in hl.items()] + [Patch(color=LIGHT, label="other 8 sectors")],
               loc="lower left", bbox_to_anchor=(0.78, 0.66), fontsize=10.5)
    s = pd.read_csv(R / "cluster_summary.csv").sort_values("mean_corr_within", ascending=False).head(5)
    lines = ["Strongest clusters (k = 11)", ""] + [f"{r.top_sector} ({r.size} stocks)  r = {r.mean_corr_within:.2f}" for r in s.itertuples()]
    fig.text(0.04, 0.70, "\n".join(lines), fontsize=11, color=INK2, va="top", linespacing=1.6)
    fig.text(0.04, 0.36, "Sensible pairs across sectors:\n\nDevon / ConocoPhillips + Freeport (copper)\nPulte / Lennar + Masco (building)\nAmazon + Meta",
             fontsize=11, color=INK2, va="top", linespacing=1.6)
    save(fig, "fig10_clustering")


# ---------------------------------------------------------------- 11. April 2025
def fig_april():
    def load(n):
        p = pd.read_parquet(PREDICTIONS_DIR / f"{n}_with_test.parquet")
        p["f"] = p["yhat"] + 0.5 * np.log(p["smear"])
        return p
    own, peer = load("har"), load("har_sector")
    w = (own["y_date"] >= "2025-03-03") & (own["y_date"] <= "2025-05-30")
    o, pe = own[w], peer[(peer["y_date"] >= "2025-03-03") & (peer["y_date"] <= "2025-05-30")]
    actual = ann_vol(o.groupby("y_date")["y"].median())
    f_own = ann_vol(o.groupby("y_date")["f"].median())
    f_peer = ann_vol(pe.groupby("y_date")["f"].median())
    fig, ax = figure("Case: the April 2025 tariff sell-off",
                     "Median stock: actual next-day volatility vs the two forecasts. The peer model rises faster in the crash,\nbut stays too high for days after the 9 April rebound.")
    ax.plot(actual.index, actual.values, color=INK, lw=2.2, label="Actual", marker="o", ms=4)
    ax.plot(f_own.index, f_own.values, color=BLUE, lw=2, label="HAR (own history)")
    ax.plot(f_peer.index, f_peer.values, color=ORANGE, lw=2, label="HAR + sector peers")
    for d, txt, ha in [("2025-04-02", "2 Apr: tariffs announced ", "right"), ("2025-04-09", " 9 Apr: 90-day pause", "left")]:
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=1)
        ax.text(pd.Timestamp(d), actual.max() * 1.08, txt, fontsize=10.5, color=INK2, va="top", ha=ha)
    ax.set_ylabel("Annualized volatility (%)"); ax.set_ylim(0, actual.max() * 1.12)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    save(fig, "fig11_april_2025")


def main():
    for f in [fig_timeline, fig_graphs, fig_main, fig_placebo, fig_seeds, fig_curves, fig_stress,
              fig_sectors, fig_disagreement, fig_clusters, fig_april]:
        f()


if __name__ == "__main__":
    main()
