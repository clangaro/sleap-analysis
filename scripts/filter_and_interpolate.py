"""
Filter and interpolate SLEAP nose/tail predictions.

For each .predictions.slp file under --pred-root:
  1. Load the predicted nose and tail trajectories.
  2. Drop predictions where nose and tail collapsed to the same point
     (Euclidean distance below --collapse-thresh pixels).
  3. Linearly interpolate gaps no longer than --max-gap frames.
  4. Leave longer gaps as NaN (mouse genuinely off-arena / off-camera).
  5. Write one CSV per video to --out-dir with columns
     frame, nose_x, nose_y, nose_score, tail_x, tail_y, tail_score,
     nose_status, tail_status
     where each status is one of: ok / interpolated / missing.

A summary CSV with per-video counts is written to <out-dir>/summary.csv.

Usage:
    python filter_and_interpolate.py --pred-root PATH [--out-dir PATH]
                                      [--max-gap N] [--collapse-thresh PX]
                                      [--require-any barnes,nor]

Pass --require-any to restrict processing to videos whose ancestor folder
names contain at least one of the given case-insensitive substrings. Videos
in unrecognized folders keep their .predictions.slp but no CSV is written.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from slp_io import load_slp

DEFAULT_OUT_DIR = str(Path(__file__).resolve().parent.parent / 'results')


def extract_nose_tail(slp_path):
    """Return (n_frames, nose, tail) where nose and tail are float arrays of shape (n, 3).

    Columns: x, y, score. Rows are indexed by frame_idx (0..max_frame_idx).
    Missing predictions are NaN.
    """
    labels = load_slp(str(slp_path))
    if not labels.labeled_frames:
        return 0, np.empty((0, 3)), np.empty((0, 3))

    max_idx = max(lf.frame_idx for lf in labels.labeled_frames)
    n = max_idx + 1
    nose = np.full((n, 3), np.nan, dtype=float)
    tail = np.full((n, 3), np.nan, dtype=float)

    for lf in labels.labeled_frames:
        if not lf.instances:
            continue
        inst = lf.instances[0]
        node_names = [nd.name for nd in inst.skeleton.nodes]
        for name, pt in zip(node_names, inst.points):
            if not pt.visible:
                continue
            arr = nose if name == 'nose' else (tail if name == 'tail' else None)
            if arr is None:
                continue
            arr[lf.frame_idx, 0] = pt.x
            arr[lf.frame_idx, 1] = pt.y
            arr[lf.frame_idx, 2] = pt.score if pt.score is not None else np.nan
    return n, nose, tail


def filter_collapsed(nose, tail, collapse_thresh):
    """Set nose and tail to NaN on frames where they are closer than collapse_thresh px.

    Returns count of frames marked collapsed.
    """
    both_present = ~np.isnan(nose[:, 0]) & ~np.isnan(tail[:, 0])
    dx = nose[:, 0] - tail[:, 0]
    dy = nose[:, 1] - tail[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    collapsed = both_present & (dist < collapse_thresh)
    nose[collapsed, :] = np.nan
    tail[collapsed, :] = np.nan
    return int(collapsed.sum())


def interpolate_with_max_gap(values, max_gap):
    """Linearly interpolate NaN gaps in `values` (1D float array) up to length max_gap.

    Gaps longer than max_gap remain NaN. Leading and trailing NaNs are left as-is.
    Returns (filled_array, n_interpolated_frames).
    """
    n = len(values)
    if n == 0:
        return values, 0
    known = np.where(~np.isnan(values))[0]
    if len(known) < 2:
        return values.copy(), 0

    out = values.copy()
    interpolated_count = 0
    for a, b in zip(known[:-1], known[1:]):
        gap_len = b - a - 1
        if gap_len <= 0 or gap_len > max_gap:
            continue
        # linear interpolate frames a+1 .. b-1
        for i in range(a + 1, b):
            weight = (i - a) / (b - a)
            out[i] = (1 - weight) * values[a] + weight * values[b]
            interpolated_count += 1
    return out, interpolated_count


def process_slp(slp_path, out_csv_path, max_gap, collapse_thresh):
    """Process one .predictions.slp file. Returns dict of per-video stats."""
    n, nose, tail = extract_nose_tail(slp_path)
    if n == 0:
        return {
            'video': slp_path.stem.replace('.predictions', ''),
            'n_frames': 0,
            'collapsed': 0,
            'nose_interpolated': 0,
            'tail_interpolated': 0,
            'nose_missing_after': 0,
            'tail_missing_after': 0,
        }

    collapsed_count = filter_collapsed(nose, tail, collapse_thresh)

    nose_x_filled, n_nose_x_interp = interpolate_with_max_gap(nose[:, 0].copy(), max_gap)
    nose_y_filled, _ = interpolate_with_max_gap(nose[:, 1].copy(), max_gap)
    tail_x_filled, n_tail_x_interp = interpolate_with_max_gap(tail[:, 0].copy(), max_gap)
    tail_y_filled, _ = interpolate_with_max_gap(tail[:, 1].copy(), max_gap)

    raw_nose_missing = np.isnan(nose[:, 0])
    raw_tail_missing = np.isnan(tail[:, 0])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'frame',
                'nose_x',
                'nose_y',
                'nose_score',
                'tail_x',
                'tail_y',
                'tail_score',
                'nose_status',
                'tail_status',
            ]
        )
        for i in range(n):
            nose_status = (
                'missing'
                if np.isnan(nose_x_filled[i])
                else ('interpolated' if raw_nose_missing[i] else 'ok')
            )
            tail_status = (
                'missing'
                if np.isnan(tail_x_filled[i])
                else ('interpolated' if raw_tail_missing[i] else 'ok')
            )
            w.writerow(
                [
                    i,
                    _fmt(nose_x_filled[i]),
                    _fmt(nose_y_filled[i]),
                    _fmt(nose[i, 2]),
                    _fmt(tail_x_filled[i]),
                    _fmt(tail_y_filled[i]),
                    _fmt(tail[i, 2]),
                    nose_status,
                    tail_status,
                ]
            )

    return {
        'video': slp_path.stem.replace('.predictions', ''),
        'n_frames': n,
        'collapsed': collapsed_count,
        'nose_interpolated': n_nose_x_interp,
        'tail_interpolated': n_tail_x_interp,
        'nose_missing_after': int(np.isnan(nose_x_filled).sum()),
        'tail_missing_after': int(np.isnan(tail_x_filled).sum()),
    }


def _fmt(v):
    return '' if (isinstance(v, float) and math.isnan(v)) else f'{v:.3f}'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pred-root',
        required=True,
        help='Root folder containing .predictions.slp files (searched recursively).',
    )
    parser.add_argument(
        '--out-dir',
        default=DEFAULT_OUT_DIR,
        help='Output folder for trajectory CSVs and summary.csv.',
    )
    parser.add_argument(
        '--max-gap',
        type=int,
        default=30,
        help='Maximum gap length (frames) to interpolate. Longer gaps stay NaN.',
    )
    parser.add_argument(
        '--collapse-thresh',
        type=float,
        default=15.0,
        help='Pixel distance below which nose/tail are considered collapsed and dropped.',
    )
    parser.add_argument(
        '--require-any',
        default='',
        help='Comma-separated list of case-insensitive substrings. If set, '
        'only process .predictions.slp files where at least one ancestor '
        'folder name (relative to --pred-root) contains one of these. '
        'Videos in unrecognized folders are skipped without writing a CSV.',
    )
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)

    slp_files = sorted(pred_root.rglob('*.predictions.slp'))
    if not slp_files:
        print(f'No .predictions.slp files found under {pred_root}', file=sys.stderr)
        sys.exit(1)

    required = [s.strip().lower() for s in args.require_any.split(',') if s.strip()]
    skipped_unrecognized = []
    if required:
        kept = []
        for slp in slp_files:
            dir_parts = [p.lower() for p in slp.relative_to(pred_root).parts[:-1]]
            if any(sub in part for part in dir_parts for sub in required):
                kept.append(slp)
            else:
                skipped_unrecognized.append(slp)
        slp_files = kept

    print(f'Found {len(slp_files) + len(skipped_unrecognized)} prediction files under {pred_root}')
    if skipped_unrecognized:
        print(
            f'Skipping {len(skipped_unrecognized)} in unrecognized folders '
            f'(no match for: {", ".join(required)}):'
        )
        for slp in skipped_unrecognized:
            print(f'  - {slp.relative_to(pred_root)}')
    if not slp_files:
        print('\nNothing to process after filtering. No CSVs written.')
        return
    print(f'Writing trajectory CSVs to {out_dir}')
    print(f'Params: max_gap={args.max_gap} frames, collapse_thresh={args.collapse_thresh} px')
    print()

    summary_rows = []
    for i, slp in enumerate(slp_files, 1):
        rel = slp.relative_to(pred_root).with_suffix('')  # strips .slp
        rel = rel.with_suffix('')  # strips .predictions
        out_csv = out_dir / f'{rel}.csv'
        stats = process_slp(slp, out_csv, args.max_gap, args.collapse_thresh)
        summary_rows.append(stats)
        print(
            f'[{i:>2}/{len(slp_files)}] {rel}: '
            f'frames={stats["n_frames"]:>6} '
            f'collapsed={stats["collapsed"]:>5} '
            f'nose_interp={stats["nose_interpolated"]:>5} '
            f'tail_interp={stats["tail_interpolated"]:>5} '
            f'nose_missing={stats["nose_missing_after"]:>5} '
            f'tail_missing={stats["tail_missing_after"]:>5}'
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nSummary written to {out_dir}/summary.csv')


if __name__ == '__main__':
    main()
