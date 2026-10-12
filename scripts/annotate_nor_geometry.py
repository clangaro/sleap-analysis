"""
NOR cohort calibration — click left + right object positions.

In the familiar phase, the same two positions hold two identical familiar
objects across every NOR video in a cohort. Run this script ONCE per cohort
on any *_familiar video. The result is reused by every NOR metrics run.

Usage:
    python annotate_nor_geometry.py --video PATH --cohort N
    python annotate_nor_geometry.py --video PATH --cohort N --frame 30

Output:
    configs/cohort_<N>_nor_geometry.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--video', required=True, help='A representative familiar-phase video.')
    p.add_argument('--cohort', required=True, help="Cohort identifier, e.g. '11'.")
    p.add_argument('--frame', type=int, default=0, help='Frame to display for calibration.')
    p.add_argument(
        '--out', default=None, help='Output JSON (default configs/cohort_<N>_nor_geometry.json).'
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
        f'NOR cohort {args.cohort} — calibration\nClick 1: LEFT object, then  Click 2: RIGHT object'
    )
    ax.axis('off')
    print('Click 1: left object, then Click 2: right object.')
    clicks = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(clicks) != 2:
        sys.stderr.write('Two clicks required.\n')
        sys.exit(1)

    (lx, ly), (rx, ry) = clicks
    out = {
        'cohort': args.cohort,
        'calibration_video': args.video,
        'calibration_frame': args.frame,
        'image_width': frame.shape[1],
        'image_height': frame.shape[0],
        'object_left': {'x': lx, 'y': ly},
        'object_right': {'x': rx, 'y': ry},
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).parent.parent / 'configs' / f'cohort_{args.cohort}_nor_geometry.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.plot(lx, ly, 'go', markersize=18, markerfacecolor='none', markeredgewidth=3, label='left')
    ax.plot(rx, ry, 'ro', markersize=18, markerfacecolor='none', markeredgewidth=3, label='right')
    ax.legend()
    ax.set_title('VERIFY — green=left, red=right. Close the window when correct.')
    ax.axis('off')
    plt.show()

    print(f'\nSaved cohort calibration: {out_path}')
    print(f'  Left:  ({lx:.0f}, {ly:.0f})')
    print(f'  Right: ({rx:.0f}, {ry:.0f})')


if __name__ == '__main__':
    main()
