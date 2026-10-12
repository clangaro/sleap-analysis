"""
Novel Object Recognition (NOR) metrics from a SLEAP trajectory CSV + scene info.

NOR has two phases. Both share the same two object positions, set per cohort:

  - "familiar" phase (videos with '_familiar' in the name): two identical
    familiar objects. We report time/exploration at left vs right (to confirm
    no side bias) plus locomotor activity.

  - "test" phase (videos with '_novel' in the name): one familiar + one novel.
    Sides vary per rat, so a per-video sidecar tells us which side is novel.
    We report time at novel, time at familiar, discrimination index, etc.

Inputs:
  --cohort-geometry  : configs/cohort_<N>_nor_geometry.json (from
                       annotate_nor_geometry.py or the UI). Contains the
                       'object_left' and 'object_right' positions, constant
                       across the cohort.
  --scene (optional) : per-video sidecar (test phase only). Contains
                       'novel_side' = 'left' | 'right'. Omit for familiar phase.

"Investigating" = nose within --object-radius pixels of the object centre
                  AND the head-vector (tail->nose) is pointing towards the object
                  (cosine similarity >= --orientation-threshold, default 0.0 = within 90 deg).

Usage:
    # Familiar phase
    python nor_metrics.py --trajectory PATH.csv \\
                          --cohort-geometry configs/cohort_11_nor_geometry.json

    # Test phase
    python nor_metrics.py --trajectory PATH.csv \\
                          --cohort-geometry configs/cohort_11_nor_geometry.json \\
                          --scene 110_novel.scene.json
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from trajectory import nose_path_length, parse_coord


def load_trajectory(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    'frame': int(row['frame']),
                    'nose_x': parse_coord(row['nose_x']),
                    'nose_y': parse_coord(row['nose_y']),
                    'tail_x': parse_coord(row['tail_x']),
                    'tail_y': parse_coord(row['tail_y']),
                }
            )
    return rows


def investigating(rows, obj_x, obj_y, obj_radius_px, orient_thresh):
    """Per-frame: True iff nose is near the object AND the head points at it."""
    out = []
    for r in rows:
        nx, ny = r['nose_x'], r['nose_y']
        tx, ty = r['tail_x'], r['tail_y']
        if nx != nx or ny != ny or tx != tx or ty != ty:
            out.append(False)
            continue
        d = math.hypot(nx - obj_x, ny - obj_y)
        if d > obj_radius_px:
            out.append(False)
            continue
        hx, hy = nx - tx, ny - ty  # head direction (tail->nose)
        ox, oy = obj_x - nx, obj_y - ny  # vector from nose to object
        h_mag = math.hypot(hx, hy)
        o_mag = math.hypot(ox, oy)
        if h_mag < 1e-6 or o_mag < 1e-6:
            out.append(False)
            continue
        cos = (hx * ox + hy * oy) / (h_mag * o_mag)
        out.append(cos >= orient_thresh)
    return out


def first_true_frame(rows, flags):
    for r, f in zip(rows, flags):
        if f:
            return r['frame']
    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--trajectory', required=True, help='Trajectory CSV from filter_and_interpolate.py.'
    )
    p.add_argument(
        '--cohort-geometry',
        required=True,
        help='Per-cohort NOR geometry JSON (object_left + object_right).',
    )
    p.add_argument(
        '--scene',
        default=None,
        help="Per-video sidecar with 'novel_side'. Required for test-phase videos, "
        'omitted for familiar-phase ones.',
    )
    p.add_argument('--fps', type=float, default=30.0, help='Video frame rate.')
    p.add_argument(
        '--object-radius',
        type=float,
        default=80.0,
        help="Pixel radius around object counted as 'near object' (default 80).",
    )
    p.add_argument(
        '--orientation-threshold',
        type=float,
        default=0.0,
        help='Cosine similarity threshold for head-pointing-at-object. '
        '0.0 = within 90 deg, 0.5 = within 60 deg, 1.0 = exactly aligned. '
        'Default 0.0.',
    )
    p.add_argument(
        '--out', default=None, help='Output JSON (default: <trajectory>.nor_metrics.json).'
    )
    args = p.parse_args()

    rows = load_trajectory(args.trajectory)
    with open(args.cohort_geometry) as f:
        cohort = json.load(f)
    lx, ly = cohort['object_left']['x'], cohort['object_left']['y']
    rx, ry = cohort['object_right']['x'], cohort['object_right']['y']

    left_flags = investigating(rows, lx, ly, args.object_radius, args.orientation_threshold)
    right_flags = investigating(rows, rx, ry, args.object_radius, args.orientation_threshold)

    t_left = sum(left_flags) / args.fps
    t_right = sum(right_flags) / args.fps
    plen, n_speed = nose_path_length(rows)

    common = {
        'video': Path(args.trajectory).stem,
        'frames_total': len(rows),
        'path_length_px': plen,
        'mean_speed_px_per_frame': (plen / n_speed) if n_speed else None,
        'object_radius_px_used': args.object_radius,
        'orientation_threshold_used': args.orientation_threshold,
        'fps_used': args.fps,
    }

    if args.scene:
        # test phase
        with open(args.scene) as f:
            scene = json.load(f)
        novel_side = scene.get('novel_side')
        if novel_side not in ('left', 'right'):
            sys.stderr.write(f"Scene {args.scene} missing valid 'novel_side'.\n")
            sys.exit(1)
        if novel_side == 'left':
            novel_flags, familiar_flags = left_flags, right_flags
        else:
            novel_flags, familiar_flags = right_flags, left_flags

        t_nov = sum(novel_flags) / args.fps
        t_fam = sum(familiar_flags) / args.fps
        total = t_nov + t_fam
        di = (t_nov - t_fam) / total if total > 0 else None

        first_fam = first_true_frame(rows, familiar_flags)
        first_nov = first_true_frame(rows, novel_flags)

        metrics = {
            'phase': 'test',
            'novel_side': novel_side,
            'familiar_side': 'right' if novel_side == 'left' else 'left',
            'frames_investigating_familiar': sum(familiar_flags),
            'frames_investigating_novel': sum(novel_flags),
            'time_familiar_sec': t_fam,
            'time_novel_sec': t_nov,
            'total_exploration_sec': total,
            'discrimination_index': di,
            'first_approach_familiar_frame': first_fam,
            'first_approach_novel_frame': first_nov,
            'first_approach_familiar_sec': (first_fam / args.fps)
            if first_fam is not None
            else None,
            'first_approach_novel_sec': (first_nov / args.fps) if first_nov is not None else None,
        }
    else:
        # familiar phase
        total = t_left + t_right
        bias = (t_left - t_right) / total if total > 0 else None

        first_left = first_true_frame(rows, left_flags)
        first_right = first_true_frame(rows, right_flags)

        metrics = {
            'phase': 'familiar',
            'frames_at_left': sum(left_flags),
            'frames_at_right': sum(right_flags),
            'time_left_sec': t_left,
            'time_right_sec': t_right,
            'total_exploration_sec': total,
            'left_right_bias': bias,
            'first_approach_left_frame': first_left,
            'first_approach_right_frame': first_right,
            'first_approach_left_sec': (first_left / args.fps) if first_left is not None else None,
            'first_approach_right_sec': (first_right / args.fps)
            if first_right is not None
            else None,
        }

    metrics.update(common)

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.trajectory).with_suffix('').with_suffix('.nor_metrics.json')
    )
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'=== {metrics["video"]} ({metrics["phase"]}) ===')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
