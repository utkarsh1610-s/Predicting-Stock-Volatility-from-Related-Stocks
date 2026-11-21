"""Check that the raw data snapshot matches data/MANIFEST.sha256.

  python scripts/verify_data.py           # verify (run after unzipping the snapshot)
  python scripts/verify_data.py --write   # regenerate the manifest (only when the snapshot changes)

Covers the raw inputs only; everything in data/processed is rebuilt by clean.py.
"""
import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST = Path("data/MANIFEST.sha256")
FILES = ["data/sp500_constituents.csv", "data/run_info.txt", "data/failed_tickers.txt"]
GLOBS = ["data/raw_full/*.csv"]


def snapshot_files():
    paths = [Path(f) for f in FILES]
    for g in GLOBS:
        paths += sorted(Path().glob(g))
    return paths


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write():
    lines = [f"{sha256(p)}  {p.as_posix()}" for p in snapshot_files()]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} entries to {MANIFEST}")


def verify():
    if not MANIFEST.exists():
        sys.exit(f"{MANIFEST} not found; unzip the data snapshot into the project root first.")
    expected = dict(reversed(line.split("  ", 1)) for line in MANIFEST.read_text().splitlines())
    missing = [p for p in expected if not Path(p).exists()]
    changed = [p for p, h in expected.items() if Path(p).exists() and sha256(Path(p)) != h]
    extra = [p.as_posix() for p in snapshot_files() if p.as_posix() not in expected]
    print(f"{len(expected) - len(missing) - len(changed)}/{len(expected)} files match")
    for label, items in [("Missing", missing), ("Changed", changed), ("Not in manifest", extra)]:
        if items:
            print(f"{label} ({len(items)}): {items[:10]}")
    if missing or changed:
        sys.exit(1)
    print("Snapshot OK. Run scripts/clean.py to rebuild data/processed/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    write() if ap.parse_args().write else verify()
