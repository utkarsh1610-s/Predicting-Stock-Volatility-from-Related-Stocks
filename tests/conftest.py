import sys
from pathlib import Path

# scripts/ is not a package; make `import config` / `import clean` work
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
