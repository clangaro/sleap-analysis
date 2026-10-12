"""
NOR test-phase per-video annotation — which side has the novel object?

In the test phase, the same two positions hold a familiar and a novel object,
but which side is which varies per mouse. This script saves that side flag
to a small per-video sidecar.

Two ways to run:

  Interactive (shows the frame with both positions marked, asks via prompt):
      python annotate_nor_scene.py --video PATH --cohort N

  Non-interactive (for scripted use):
      python annotate_nor_scene.py --video PATH --cohort N --novel-side left

Output:
    <video>.scene.json next to the source video, e.g.
        {"video": ..., "phase": "test", "novel_side": "left", "familiar_side": "right"}

Requires that annotate_nor_geometry.py has already been run for this cohort.
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
    p.add_argument('--video', required=True, help='A *_novel.mp4 video.')
    p.add_argument('--cohort', required=True, help="Cohort identifier, e.g. '11'.")
    p.add_argument(
        '--novel-side',
        choices=['left', 'right'],
        default=None,
        help='Non-interactive choice. If omitted, opens the frame and prompts.',
    )
    p.add_argument(
        '--frame', type=int, default=0, help='Frame to display when interactive (default 0).'
    )
    args = p.parse_args()

    cohort_geom = (
        Path(__file__).parent.parent / 'configs' / f'cohort_{args.cohort}_nor_geometry.json'
    )
    if not cohort_geom.exists():
        sys.stderr.write(
            f'Cohort calibration not found at {cohort_geom}.\nRun annotate_nor_geometry.py first.\n'
        )
        sys.exit(1)
    geom = json.loads(cohort_geom.read_text())
    lx, ly = geom['object_left']['x'], geom['object_left']['y']
    rx, ry = geom['object_right']['x'], geom['object_right']['y']

    side = args.novel_side
    if side is None:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.stderr.write(f'Could not read frame {args.frame} from {args.video}\n')
            sys.exit(1)

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.plot(lx, ly, 'yo', markersize=18, markerfacecolor='none', markeredgewidth=3)
        ax.plot(rx, ry, 'yo', markersize=18, markerfacecolor='none', markeredgewidth=3)
        ax.text(lx + 18, ly, 'LEFT', color='yellow', fontsize=12)
        ax.text(rx + 18, ry, 'RIGHT', color='yellow', fontsize=12)
        ax.set_title(
            f'{Path(args.video).name}\n'
            'Note which side has the NOVEL object, then close this window.'
        )
        ax.axis('off')
        plt.show()

        while side not in ('left', 'right'):
            side = input('Which side has the novel object? [left/right]: ').strip().lower()

    out = {
        'video': args.video,
        'phase': 'test',
        'novel_side': side,
        'familiar_side': 'right' if side == 'left' else 'left',
    }
    sidecar = Path(args.video).parent / f'{Path(args.video).stem}.scene.json'
    with open(sidecar, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {sidecar}')
    print(f'  novel_side = {side}')


if __name__ == '__main__':
    main()
