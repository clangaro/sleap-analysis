"""
Sanity-check filter_and_interpolate trajectories with biological plausibility metrics.

For each trajectory CSV produced by filter_and_interpolate.py under --results-root:
  1. Compute frame-to-frame Euclidean distance (pixel speed) for nose and tail.
  2. Flag frames where speed exceeds --max-speed (default 30 px/frame).
     At 30 fps a rat moving 30 px/frame is roughly 900 px/sec — well above
     normal locomotion. Sustained higher values are almost always model errors.
  3. Compute per-video LONGEST consecutive run of status=="missing" (post-
     interpolation, so a "missing" frame here means a gap that was too long
     for filter_and_interpolate.py to fill). Convert to seconds via --fps
     and set high_missingness=1 when BOTH nose and tail longest gaps exceed
     --max-gap-seconds. See process() for the rationale behind AND vs OR
     and behind gap-length vs global-fraction.
  4. Write:
       - <video>_validation.csv (per-frame speed flags)
       - validation_summary.csv (one row per video, incl. gap columns)

Nothing here modifies the trajectory CSVs from filter_and_interpolate.py —
this script is diagnostic-only. Downstream metrics scripts read the raw
trajectory CSVs; flagged videos are not auto-excluded.

Usage:
    python validate_trajectories.py [--results-root PATH] [--max-speed N]
                                    [--max-gap-seconds S] [--fps F]
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from trajectory import parse_coord

DEFAULT_RESULTS_ROOT = str(Path(__file__).resolve().parent.parent / 'results')


def load_trajectory_csv(path):
    """Read a CSV produced by filter_and_interpolate.py.

    Returns dict with keys: frame, nose_x, nose_y, tail_x, tail_y, nose_status, tail_status
    (all as numpy arrays). Missing positions are NaN.
    """
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n = len(rows)
    out = {
        'frame': np.zeros(n, dtype=int),
        'nose_x': np.full(n, np.nan),
        'nose_y': np.full(n, np.nan),
        'tail_x': np.full(n, np.nan),
        'tail_y': np.full(n, np.nan),
        'nose_status': np.empty(n, dtype=object),
        'tail_status': np.empty(n, dtype=object),
    }
    for i, r in enumerate(rows):
        out['frame'][i] = int(r['frame'])
        for k in ('nose_x', 'nose_y', 'tail_x', 'tail_y'):
            v = r[k]
            out[k][i] = parse_coord(v)
        out['nose_status'][i] = r['nose_status']
        out['tail_status'][i] = r['tail_status']
    return out


def per_frame_speed(x, y):
    """Frame-to-frame Euclidean distance.

    Returns array of length len(x); speeds[0] is NaN. NaN at either endpoint of a pair → NaN speed.
    """
    n = len(x)
    speeds = np.full(n, np.nan)
    if n < 2:
        return speeds
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    speeds[1:] = np.sqrt(dx * dx + dy * dy)
    return speeds


def percentile_ignore_nan(arr, q):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float('nan')
    return float(np.percentile(valid, q))


def longest_run_of_true(mask):
    """Length of the longest consecutive run of True values in a 1D boolean array.

    Returns 0 for an empty or all-False array. Pads the input with False on
    both sides so runs starting at index 0 or ending at index n-1 are still
    detected as complete runs by the diff-based transition scan below.
    """
    if len(mask) == 0:
        return 0
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]  # False -> True transitions
    ends = np.where(diffs == -1)[0]  # True -> False transitions
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def process(csv_path, out_csv_path, max_speed, max_gap_frames, fps):
    traj = load_trajectory_csv(csv_path)
    nose_speed = per_frame_speed(traj['nose_x'], traj['nose_y'])
    tail_speed = per_frame_speed(traj['tail_x'], traj['tail_y'])

    nose_suspicious = nose_speed > max_speed
    tail_suspicious = tail_speed > max_speed

    # Longest CONSECUTIVE run of status=="missing" per keypoint. "missing"
    # here means a frame that stayed NaN after linear interpolation in
    # filter_and_interpolate.py (gap longer than --max-gap, or a leading/
    # trailing NaN the interpolator doesn't extrapolate into).
    #
    # Why longest-gap and not global fraction: a single 5-second blackout
    # during the exact window that matters for latency / discrimination
    # index will silently distort those metrics, but is easily hidden inside
    # a global average dominated by minor scattered dropouts. Consecutive
    # gap length reflects the real behavioural risk. A video peppered with
    # single-frame dropouts totalling 15% missing is fine (the interpolator
    # handles those); a video with 3% missing but a single 5s blackout is
    # not.
    nose_longest_gap_frames = longest_run_of_true(traj['nose_status'] == 'missing')
    tail_longest_gap_frames = longest_run_of_true(traj['tail_status'] == 'missing')
    nose_longest_gap_sec = nose_longest_gap_frames / fps if fps > 0 else float('nan')
    tail_longest_gap_sec = tail_longest_gap_frames / fps if fps > 0 else float('nan')

    # AND, not OR — flag fires ONLY when BOTH keypoints have a run longer
    # than --max-gap-seconds. Reader beware: this is LESS conservative than
    # OR, which is the opposite of the usual intuition. Rationale: one
    # struggling keypoint is usually local occlusion (e.g. tail out of frame
    # during a rear while nose tracks fine) and the metrics that depend on
    # nose are still valid; a globally unreliable video shows up in BOTH
    # keypoints simultaneously. Flip to `or` here if you'd rather over-flag
    # than under-flag.
    high_missingness = (nose_longest_gap_frames > max_gap_frames) and (
        tail_longest_gap_frames > max_gap_frames
    )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'frame',
                'nose_speed_px',
                'nose_suspicious',
                'tail_speed_px',
                'tail_suspicious',
            ]
        )
        for i in range(len(traj['frame'])):
            w.writerow(
                [
                    int(traj['frame'][i]),
                    _fmt(nose_speed[i]),
                    int(bool(nose_suspicious[i])) if not np.isnan(nose_speed[i]) else '',
                    _fmt(tail_speed[i]),
                    int(bool(tail_suspicious[i])) if not np.isnan(tail_speed[i]) else '',
                ]
            )

    def _stats(speed, suspicious):
        n_valid = int(np.sum(~np.isnan(speed)))
        if n_valid == 0:
            return {
                'n_speed_frames': 0,
                'mean_speed': float('nan'),
                'p95_speed': float('nan'),
                'p99_speed': float('nan'),
                'max_speed': float('nan'),
                'n_suspicious': 0,
                'pct_suspicious': float('nan'),
                'worst_jump_frame': -1,
            }
        worst_idx = int(np.nanargmax(speed))
        return {
            'n_speed_frames': n_valid,
            'mean_speed': float(np.nanmean(speed)),
            'p95_speed': percentile_ignore_nan(speed, 95),
            'p99_speed': percentile_ignore_nan(speed, 99),
            'max_speed': float(np.nanmax(speed)),
            'n_suspicious': int(np.nansum(suspicious)),
            'pct_suspicious': 100.0 * float(np.nansum(suspicious)) / n_valid,
            'worst_jump_frame': worst_idx,
        }

    ns = _stats(nose_speed, nose_suspicious)
    ts = _stats(tail_speed, tail_suspicious)

    return {
        'video': csv_path.stem,
        'n_frames': len(traj['frame']),
        'nose_speed_frames': ns['n_speed_frames'],
        'nose_mean_px': _fmt(ns['mean_speed']),
        'nose_p95_px': _fmt(ns['p95_speed']),
        'nose_p99_px': _fmt(ns['p99_speed']),
        'nose_max_px': _fmt(ns['max_speed']),
        'nose_n_suspicious': ns['n_suspicious'],
        'nose_pct_suspicious': _fmt(ns['pct_suspicious']),
        'nose_worst_jump_frame': ns['worst_jump_frame'],
        'tail_speed_frames': ts['n_speed_frames'],
        'tail_mean_px': _fmt(ts['mean_speed']),
        'tail_p95_px': _fmt(ts['p95_speed']),
        'tail_p99_px': _fmt(ts['p99_speed']),
        'tail_max_px': _fmt(ts['max_speed']),
        'tail_n_suspicious': ts['n_suspicious'],
        'tail_pct_suspicious': _fmt(ts['pct_suspicious']),
        'tail_worst_jump_frame': ts['worst_jump_frame'],
        'nose_longest_gap_frames': nose_longest_gap_frames,
        'nose_longest_gap_sec': _fmt(nose_longest_gap_sec),
        'tail_longest_gap_frames': tail_longest_gap_frames,
        'tail_longest_gap_sec': _fmt(tail_longest_gap_sec),
        'high_missingness': int(high_missingness),
    }


def _fmt(v):
    if isinstance(v, float) and math.isnan(v):
        return ''
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--results-root',
        default=DEFAULT_RESULTS_ROOT,
        help='Folder containing trajectory CSVs from filter_and_interpolate.py.',
    )
    parser.add_argument(
        '--max-speed',
        type=float,
        default=30.0,
        help='Pixel speed above which a frame is flagged as suspicious (px / frame).',
    )
    parser.add_argument(
        '--max-gap-seconds',
        type=float,
        default=5.0,
        help='A video is flagged high_missingness=1 when BOTH nose and tail have '
        "a longest CONSECUTIVE run of status=='missing' exceeding this many "
        'seconds. AND (not OR) — flag fires only when both keypoints have a '
        'long gap, on the assumption that one struggling keypoint is usually '
        'local occlusion rather than a globally unreliable video.',
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frame rate used to convert --max-gap-seconds into a frame count. '
        'Should match the fps passed to the metrics scripts.',
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f'results root does not exist: {root}', file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(
        p
        for p in root.rglob('*.csv')
        if p.name not in ('summary.csv', 'validation_summary.csv') and '_validation' not in p.stem
    )
    if not csv_files:
        print(f'No trajectory CSVs found under {root}', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(csv_files)} trajectory CSVs under {root}')
    print(f'Flagging speed > {args.max_speed} px/frame')
    print()

    summary_rows = []
    for i, csv_path in enumerate(csv_files, 1):
        rel = csv_path.relative_to(root)
        out_csv = root / rel.with_name(rel.stem + '_validation.csv')
        max_gap_frames = int(round(args.max_gap_seconds * args.fps))
        stats = process(csv_path, out_csv, args.max_speed, max_gap_frames, args.fps)
        summary_rows.append(stats)
        print(
            f'[{i:>2}/{len(csv_files)}] {rel.with_suffix("")}: '
            f'nose mean={stats["nose_mean_px"]} px, '
            f'p99={stats["nose_p99_px"]} px, '
            f'max={stats["nose_max_px"]} px, '
            f'suspicious={stats["nose_n_suspicious"]} ({stats["nose_pct_suspicious"]}%)'
        )

    summary_path = root / 'validation_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nSummary: {summary_path}')


if __name__ == '__main__':
    main()
