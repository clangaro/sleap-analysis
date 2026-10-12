"""
Compare predicted keypoints to hand-labelled ground truth on the same frames.

Use this to measure out-of-distribution accuracy of a SLEAP model:

    1. Hand-label some frames in a video the model has never seen (BLIND
       to the predictions, otherwise the measurement is biased).
    2. Run sleap-track on the same video to produce a .predictions.slp.
    3. Run this script. For every frame that has BOTH a hand label and a
       prediction, it computes the Euclidean distance per keypoint.

Outputs to stdout and writes a per-frame CSV to --out.

Usage:
    python compare_to_truth.py --truth GROUND_TRUTH.slp --pred PRED.predictions.slp
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from slp_io import load_slp


def index_by_frame(labels):
    """Return {frame_idx: {node_name: (x, y, visible)}}."""
    out = {}
    for lf in labels.labeled_frames:
        if not lf.instances:
            continue
        inst = lf.instances[0]
        node_names = [nd.name for nd in inst.skeleton.nodes]
        per_node = {}
        for name, pt in zip(node_names, inst.points):
            x = float(pt.x) if pt.visible else float('nan')
            y = float(pt.y) if pt.visible else float('nan')
            per_node[name] = (x, y, bool(pt.visible))
        out[lf.frame_idx] = per_node
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--truth', required=True, help='Path to hand-labelled .slp')
    p.add_argument('--pred', required=True, help='Path to .predictions.slp from sleap-track')
    p.add_argument('--out', default='comparison.csv', help='Per-frame comparison CSV output')
    args = p.parse_args()

    truth = load_slp(args.truth)
    pred = load_slp(args.pred)

    truth_by_frame = index_by_frame(truth)
    pred_by_frame = index_by_frame(pred)

    if not truth_by_frame:
        print('Ground truth file has no labelled frames.', file=sys.stderr)
        sys.exit(1)

    # Get node names from the ground truth skeleton; assume single skeleton
    node_names = [nd.name for nd in truth.skeletons[0].nodes]

    common_frames = sorted(set(truth_by_frame) & set(pred_by_frame))
    only_truth = sorted(set(truth_by_frame) - set(pred_by_frame))
    print(f'Ground truth frames:        {len(truth_by_frame)}')
    print(f'Prediction frames:          {len(pred_by_frame)}')
    print(f'Frames in both:             {len(common_frames)}')
    print(f'Labeled but no prediction:  {len(only_truth)}  (model missed these)')
    print()

    # Per-node accumulators
    dists = {n: [] for n in node_names}

    out_rows = []
    for fi in sorted(truth_by_frame):
        t = truth_by_frame[fi]
        p_frame = pred_by_frame.get(fi, {})
        row = {'frame': fi}
        for name in node_names:
            tx, ty, tv = t.get(name, (float('nan'), float('nan'), False))
            px, py, pv = p_frame.get(name, (float('nan'), float('nan'), False))
            if tv and pv:
                d = math.hypot(tx - px, ty - py)
                dists[name].append(d)
            else:
                d = float('nan')
            row[f'{name}_truth_x'] = tx
            row[f'{name}_truth_y'] = ty
            row[f'{name}_pred_x'] = px
            row[f'{name}_pred_y'] = py
            row[f'{name}_dist_px'] = d
            row[f'{name}_pred_present'] = int(pv)
        out_rows.append(row)

    # Write per-frame CSV
    fields = ['frame']
    for n in node_names:
        fields += [
            f'{n}_truth_x',
            f'{n}_truth_y',
            f'{n}_pred_x',
            f'{n}_pred_y',
            f'{n}_dist_px',
            f'{n}_pred_present',
        ]
    out_path = Path(args.out)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(
                {k: ('' if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()}
            )
    print(f'Per-frame comparison written: {out_path}')
    print()

    # Stats per node
    print(
        f'{"keypoint":<10} {"n_compared":>10} {"mean px":>9} {"median":>9} '
        f'{"p95 px":>9} {"p99 px":>9} {"within 5":>10} {"within 10":>10}'
    )
    print('=' * 80)
    for name in node_names:
        arr = np.array(dists[name], dtype=float)
        if len(arr) == 0:
            print(f'{name:<10}  no overlapping frames')
            continue
        within_5 = 100.0 * float(np.mean(arr <= 5))
        within_10 = 100.0 * float(np.mean(arr <= 10))
        print(
            f'{name:<10} {len(arr):>10d} '
            f'{float(np.mean(arr)):>9.2f} '
            f'{float(np.median(arr)):>9.2f} '
            f'{float(np.percentile(arr, 95)):>9.2f} '
            f'{float(np.percentile(arr, 99)):>9.2f} '
            f'{within_5:>9.1f}% '
            f'{within_10:>9.1f}%'
        )


if __name__ == '__main__':
    main()
