"""Run the full ECG classification pipeline end to end."""
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecgnet.pipeline import run  # noqa: E402

if __name__ == "__main__":
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    metrics = run(cfg, outdir=str(ROOT / "results"))
    print(json.dumps(metrics, indent=2))
