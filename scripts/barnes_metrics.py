"""
Barnes maze behavioural metrics from a SLEAP trajectory CSV + cohort geometry.

Reads a trajectory CSV (produced by filter_and_interpolate.py) and the
cohort geometry JSON (produced by annotate_barnes_geometry.py). Computes:

  - first_target_interaction_frame  : frame index of first nose-at-target event
  - latency_to_target_sec           : same, converted to seconds (using --fps)
  - n_target_interactions           : number of distinct target-hole visits
  - frames_at_target                : total frames nose was within target zone
  - time_at_target_sec              : same, in seconds
  - target_quadrant_frames          : frames where nose was in the target quadrant
  - target_quadrant_pct             : fraction of valid frames in target quadrant
  - path_length_px                  : sum of nose pixel displacement across valid frames
  - mean_speed_px_per_frame         : path length / number of speed-valid frames

A "target-zone visit" is a maximal run of contiguous frames within the
target radius. The target zone radius defaults to 12% of the centre-to-target
distance. Geometry files in the older schema (radius_px + n_holes) default to
1.5 x half the hole spacing instead.

Quadrants: 4 equal angular sectors centred on the target (so target is
in the middle of the target quadrant, +/- 45 deg).

Usage:
    python barnes_metrics.py --trajectory PATH.csv --geometry PATH.json \
                             [--fps 30] [--target-radius PX]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from trajectory import angle_clockwise_from_vertical, nose_path_length, parse_coord


def load_trajectory(path):
    """Returns list of dicts: {frame, nose_x, nose_y} with NaN where missing."""
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append(
                    {
                        'frame': int(row['frame']),
                        'nose_x': parse_coord(row['nose_x']),
                        'nose_y': parse_coord(row['nose_y']),
                    }
                )
            except KeyError:
                sys.stderr.write('Trajectory CSV must have columns: frame, nose_x, nose_y\n')
                sys.exit(1)
    return rows


def compute_metrics(rows, geom, target_radius_px, fps):
    # Accept the British spelling ('centre') as canonical; fall back to legacy 'center'.
    centre = geom.get('centre') or geom['center']
    cx = centre['x']
    cy = centre['y']
    target = geom['target_hole']
    target_angle = float(target['angle_deg'])

    # Determine per-frame state
    at_target = []
    in_target_quadrant = []
    for r in rows:
        x, y = r['nose_x'], r['nose_y']
        if x != x or y != y:  # NaN
            at_target.append(False)
            in_target_quadrant.append(False)
            continue

        d_to_target = math.hypot(x - target['x'], y - target['y'])
        at_target.append(d_to_target <= target_radius_px)

        a = angle_clockwise_from_vertical(x - cx, y - cy)
        # angular distance to target, smallest delta in either direction
        diff = abs(((a - target_angle) + 180.0) % 360.0 - 180.0)
        in_target_quadrant.append(diff <= 45.0)

    # First interaction + count
    first_interaction = next((r['frame'] for r, t in zip(rows, at_target) if t), None)
    visits = 0
    prev = False
    for t in at_target:
        if t and not prev:
            visits += 1
        prev = t

    frames_at_target = sum(at_target)
    valid_frames = sum(1 for r in rows if r['nose_x'] == r['nose_x'])
    quad_frames = sum(in_target_quadrant)

    path_px, n_speed = nose_path_length(rows)

    return {
        'n_frames_total': len(rows),
        'n_frames_valid_nose': valid_frames,
        'first_target_interaction_frame': first_interaction,
        'latency_to_target_sec': (first_interaction / fps)
        if first_interaction is not None
        else None,
        'n_target_interactions': visits,
        'frames_at_target': frames_at_target,
        'time_at_target_sec': frames_at_target / fps,
        'target_quadrant_frames': quad_frames,
        'target_quadrant_pct': (100.0 * quad_frames / valid_frames) if valid_frames else None,
        'path_length_px': path_px,
        'mean_speed_px_per_frame': (path_px / n_speed) if n_speed else None,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--trajectory',
        required=True,
        help='Per-video trajectory CSV from filter_and_interpolate.py.',
    )
    p.add_argument(
        '--geometry', required=True, help='Cohort geometry JSON from annotate_barnes_geometry.py.'
    )
    p.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Video frame rate, for converting frames to seconds.',
    )
    p.add_argument(
        '--target-radius',
        type=float,
        default=None,
        help="Radius (px) around target hole counted as 'at target'. "
        'Default = 12%% of the centre-to-target distance '
        '(approximately one hole diameter on our setup).',
    )
    p.add_argument(
        '--out', default=None, help='Output JSON (default: <trajectory>.barnes_metrics.json).'
    )
    args = p.parse_args()

    rows = load_trajectory(args.trajectory)
    with open(args.geometry) as f:
        geom = json.load(f)

    # Default target zone radius: 12% of the centre-to-target distance.
    # On our setup the maze radius is ~the centre-to-target distance, and a
    # hole is roughly 7-12% of that radius, so 12% gives "the hole plus a
    # small body-length margin". Override with --target-radius if needed.
    if args.target_radius is None:
        if 'centre_to_target_px' in geom:
            target_radius_px = 0.12 * float(geom['centre_to_target_px'])
        elif 'radius_px' in geom and 'n_holes' in geom:  # legacy schema
            circumference = 2 * math.pi * geom['radius_px']
            hole_spacing = circumference / geom['n_holes']
            target_radius_px = 1.5 * (hole_spacing / 2)
        else:
            sys.stderr.write(
                'Cannot infer a default target radius from this geometry JSON. '
                'Pass --target-radius explicitly.\n'
            )
            sys.exit(1)
    else:
        target_radius_px = args.target_radius

    metrics = compute_metrics(rows, geom, target_radius_px, args.fps)
    metrics['target_radius_px_used'] = target_radius_px
    metrics['fps_used'] = args.fps
    metrics['video'] = Path(args.trajectory).stem
    metrics['cohort'] = geom.get('cohort')
    metrics['target_hole_index'] = geom.get('target_hole_index')

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.trajectory).with_suffix('').with_suffix('.barnes_metrics.json')
    )
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(
        f'=== {metrics["video"]} (cohort {metrics["cohort"]}, '
        f'target hole {metrics["target_hole_index"]}) ==='
    )
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
