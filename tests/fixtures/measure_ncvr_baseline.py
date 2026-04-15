"""One-off: measure baseline F1 for auto-config on NCVR synthetic dupes.
Run once to populate tests/parity/autoconfig-f1-floors.json's ncvr_synth.
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.fixtures.ncvr_synth_dupes import build_ncvr_synth_df
from goldenmatch._api import dedupe_df


def main():
    df, gt_pairs = build_ncvr_synth_df()
    result = dedupe_df(df)
    predicted = set()
    for cluster in result.clusters.values():
        m = sorted(cluster["members"])
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                predicted.add((m[i], m[j]))
    tp = len(predicted & gt_pairs)
    fp = len(predicted - gt_pairs)
    fn = len(gt_pairs - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(f"NCVR synth: F1={f1:.4f}  P={p:.4f}  R={r:.4f}  TP={tp} FP={fp} FN={fn}")
    return f1


if __name__ == "__main__":
    main()
