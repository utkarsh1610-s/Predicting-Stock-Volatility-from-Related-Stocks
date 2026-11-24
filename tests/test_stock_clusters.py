import numpy as np
import pandas as pd

import stock_clusters as sc


def test_ari_known_values():
    assert sc.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert sc.adjusted_rand_index([0, 0, 1, 1], ["b", "b", "a", "a"]) == 1.0      # label names don't matter
    assert np.isclose(sc.adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 2]), 0.5714285714)
    assert np.isclose(sc.adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]), 0.0)


def test_silhouette_hand_computed():
    pts = np.array([0.0, 1.0, 10.0, 12.0])
    dist = np.abs(pts[:, None] - pts[None, :])
    # a = within-cluster mean distance, b = mean distance to the other cluster
    expected = np.mean([(11 - 1) / 11, (10 - 1) / 10, (9.5 - 2) / 9.5, (11.5 - 2) / 11.5])
    assert np.isclose(sc.silhouette(dist, [0, 0, 1, 1]), expected)


def panel(n_groups=3, per_group=6, n_days=800, group_strength=1.0, seed=0):
    """Log-vol = strong market factor + group factor + noise. Group = 'sector'."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    market = np.cumsum(rng.normal(0, 0.1, n_days))
    cols, sectors = {}, {}
    for g in range(n_groups):
        factor = rng.normal(0, 1, n_days)
        for j in range(per_group):
            t = f"G{g}S{j}"
            cols[t] = -4 + 3 * market + group_strength * factor + rng.normal(0, 1, n_days)
            sectors[t] = f"sector{g}"
    lv = pd.DataFrame(cols, index=dates)
    meta = pd.DataFrame({"company": list(sectors), "sector": list(sectors.values())}, index=list(sectors))
    return lv, meta


def test_recovers_planted_groups_despite_a_dominant_market_factor(monkeypatch):
    monkeypatch.setattr(sc, "K_MAIN", 3)
    monkeypatch.setattr(sc, "K_RANGE", range(2, 6))
    lv, meta = panel()
    members, quality, summary, pairs = sc.analyse(lv, meta)
    assert sc.adjusted_rand_index(members["cluster"], members["sector"]) == 1.0
    assert (summary["top_sector_share"] == 1.0).all()
    assert quality.loc[quality["labels"] == "gics_sector", "silhouette"].iloc[0] > 0.2


def test_no_groups_means_no_agreement_with_sectors(monkeypatch):
    monkeypatch.setattr(sc, "K_MAIN", 3)
    monkeypatch.setattr(sc, "K_RANGE", range(2, 6))
    lv, meta = panel(group_strength=0.0, seed=3)
    members, quality, _, _ = sc.analyse(lv, meta)
    assert abs(quality.loc[quality["labels"] == "gics_sector", "silhouette"].iloc[0]) < 0.05


def test_market_residuals_remove_the_market():
    lv, _ = panel(group_strength=0.0, seed=1)
    raw = lv.corr().to_numpy()[np.triu_indices(lv.shape[1], 1)].mean()
    res = sc.market_residuals(lv).corr().to_numpy()[np.triu_indices(lv.shape[1], 1)].mean()
    assert raw > 0.5 and abs(res) < 0.1


def test_short_histories_are_left_out(monkeypatch):
    monkeypatch.setattr(sc, "K_MAIN", 3)
    monkeypatch.setattr(sc, "K_RANGE", range(2, 4))
    lv, meta = panel()
    lv.iloc[: len(lv) - sc.MIN_DAYS + 1, 0] = np.nan        # one stock with too few days
    members, *_ = sc.analyse(lv, meta)
    assert lv.columns[0] not in set(members["ticker"])
