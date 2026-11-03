"""Verification: run all four methods, check against known truth.

Prints real metrics from the actual run. Do not copy metrics
into the README until this script has printed them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import evaluate  # noqa: E402

m = evaluate()
print(json.dumps(m, indent=2))
assert m["mmm"]["r2"] > 0.7, "MMM R2 unexpectedly low"
assert abs(m["incrementality_geo_experiment"]["estimated_lift"] - 0.15) < 0.05, \
    f"incrementality estimate off: {m['incrementality_geo_experiment']['estimated_lift']}"
assert abs(sum(m["attribution"]["position_based_share"].values()) - 1.0) < 0.01, "attribution shares do not sum to 1"
assert sum(s["size"] for s in m["segmentation"]["segments"]) == 2000, "segmentation sizes wrong"
print("verify: PASS")
