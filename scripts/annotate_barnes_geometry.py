"""
Interactive Barnes maze geometry annotation.

Run ONCE per cohort. Click the table centre, then click the target hole on a
representative video frame. The script saves only those two points (plus the
derived target angle and centre→target distance) — it does NOT try to fit the
other 19 holes. The cameras in our setup are tilted enough that the table
appears noticeably elliptical, so any single-radius circle would misplace
most holes.

For the metrics that actually use this geometry (`barnes_metrics.py`), the
centre and the target hole position are sufficient: latency, time at target,
number of target interactions, and target-quadrant occupancy all derive from
those two points alone.

Assumptions:
- Camera is fixed across the cohort (same maze framing in every video).
- Target hole is the SAME hole for every mouse in the cohort.

Usage:
    python annotate_barnes_geometry.py --video PATH --cohort N
    python annotate_barnes_geometry.py --video PATH --cohort N --frame 51

Output:
    configs/cohort_<N>_barnes_geometry.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from trajectory import angle_clockwise_from_vertical


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--video', required=True, help='Path to a representative Barnes video for this cohort.'
    )
    p.add_argument('--cohort', required=True, help="Cohort identifier, e.g. '11'.")
    p.add_argument(
        '--frame',
        type=int,
        default=51,
        help='Frame number to display for calibration (default 51).',
    )
    p.add_argument(
        '--out',
        default=None,
        help='Output JSON path (default configs/cohort_<N>_barnes_geometry.json).',
    )
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.stderr.write(f'Could not read frame {args.frame} from {args.video}\n')
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f'Barnes cohort {args.cohort} — calibration\n'
        'Click 1: TABLE CENTRE, then  Click 2: TARGET HOLE'
    )
    ax.axis('off')
    print('Click 1: table centre, then Click 2: target hole.')
    clicks = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(clicks) != 2:
        sys.stderr.write('Two clicks required.\n')
        sys.exit(1)

    (cx, cy), (tx, ty) = clicks
    dx, dy = tx - cx, ty - cy
    centre_to_target_px = math.hypot(dx, dy)
    target_angle = angle_clockwise_from_vertical(dx, dy)

    out = {
        'cohort': args.cohort,
        'calibration_video': args.video,
        'calibration_frame': args.frame,
        'image_width': frame.shape[1],
        'image_height': frame.shape[0],
        'centre': {'x': cx, 'y': cy},
        'target_hole': {'x': tx, 'y': ty, 'angle_deg': target_angle},
        'target_angle_deg': target_angle,
        'centre_to_target_px': centre_to_target_px,
        'convention_note': 'angle measured clockwise from image-vertical-up.',
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).parent.parent / 'configs' / f'cohort_{args.cohort}_barnes_geometry.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # Verification overlay.
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.plot(cx, cy, 'b+', markersize=18, markeredgewidth=3)
    ax.plot(
        tx, ty, 'o', markersize=18, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2
    )
    ax.plot([cx, tx], [cy, ty], color='grey', linewidth=1, linestyle='--')
    ax.set_title(
        f'VERIFY — centre (blue +), target hole (red O)\n'
        f'target angle from vertical (clockwise): {target_angle:.1f}°\n'
        f'centre→target distance: {centre_to_target_px:.0f} px'
    )
    ax.axis('off')
    plt.show()

    print(f'\nSaved geometry to: {out_path}')
    print(f'Target angle:           {target_angle:.1f}° from vertical (clockwise)')
    print(f'Centre→target distance: {centre_to_target_px:.0f} pixels')


if __name__ == '__main__':
    main()
