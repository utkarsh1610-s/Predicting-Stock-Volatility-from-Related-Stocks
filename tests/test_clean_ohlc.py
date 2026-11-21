import pandas as pd

from clean import clean_ohlc


def make(rows):
    """rows: (open, high, low, close, volume) tuples."""
    idx = pd.date_range("2020-01-02", periods=len(rows), freq="B")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


GOOD = (100.0, 102.0, 99.0, 101.0, 1000)


def test_valid_row_kept():
    out, counts = clean_ohlc(make([GOOD]))
    assert len(out) == 1 and counts["dropped_total"] == 0


def test_zero_volume_dropped():
    out, counts = clean_ohlc(make([GOOD, (100, 102, 99, 101, 0)]))
    assert len(out) == 1 and counts["drop_zero_volume"] == 1


def test_nonpositive_price_dropped():
    out, counts = clean_ohlc(make([GOOD, (100, 102, 0.0, 101, 500)]))
    assert len(out) == 1 and counts["drop_nonpositive"] == 1


def test_nan_dropped():
    out, counts = clean_ohlc(make([GOOD, (100, float("nan"), 99, 101, 500)]))
    assert len(out) == 1 and counts["drop_nan"] == 1


def test_zero_range_dropped():
    out, counts = clean_ohlc(make([GOOD, (100, 100, 100, 100, 500)]))
    assert len(out) == 1 and counts["drop_zero_range"] == 1


def test_bad_tick_dropped():
    # High (101) is well below Close (105): more than float noise
    out, counts = clean_ohlc(make([GOOD, (100, 101, 99, 105, 500)]))
    assert len(out) == 1 and counts["drop_bad_tick"] == 1


def test_float_noise_is_snapped_not_dropped():
    # Close exceeds High by ~1e-7 relative: snapped, row kept
    out, counts = clean_ohlc(make([(100, 102, 99, 102.00001, 500)]))
    assert len(out) == 1 and counts["dropped_total"] == 0
    assert out["High"].iloc[0] == 102.00001


def test_input_not_mutated():
    df = make([(100, 102, 99, 102.00001, 500)])
    clean_ohlc(df)
    assert df["High"].iloc[0] == 102
