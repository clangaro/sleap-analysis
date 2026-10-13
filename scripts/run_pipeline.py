"""
Run the full SLEAP analysis pipeline end-to-end for one dataset.

Steps:
  0. (optional) run_inference.py on any video under --videos-root that
     doesn't already have a .predictions.slp file
  1. filter_and_interpolate.py on all .predictions.slp under --pred-root
  2. validate_trajectories.py on the resulting CSVs
  3. For each Barnes trajectory:  barnes_metrics.py with the cohort geometry
  4. For each NOR trajectory:     nor_metrics.py with that video's scene JSON

Videos are classified as Barnes or NOR by scanning their ancestor folder
names under the results root for a case-insensitive substring match
("barnes" or "nor"); the first match wins.

NOR scene JSON lookup tries, in order:
  --nor-scene-dir/<video>.scene.json
  --nor-scene-dir/<video_subdir>/<video>.scene.json

Frames are converted to seconds with each video's own frame rate, read
with ffprobe in step 1 and stored in summary.csv. Pass --fps to use one
frame rate for every video instead. The video path stored in each .slp is
used when it exists on this machine; otherwise the video is looked up by
file name under --videos-root.

--barnes-exclusions names Barnes videos the cohort geometry does not fit,
as a JSON object mapping each video's path relative to the cohort folder to
a reason, for example {"BARNES_DAY6/112.mp4": "different frame size"}. Their
metrics JSON is written with excluded=true and the reason, and no metrics.

Videos that have no scene JSON (NOR), no geometry passed (Barnes) or no
frame rate are skipped from the metrics step and listed at the end. The trajectory CSVs
and validation flags are still produced for them.

Usage:
    python run_pipeline.py \\
        --videos-root data/cohort_12 \\
        --cohort cohort_12 \\
        --barnes-geometry configs/cohort_12_barnes_geometry.json \\
        --nor-geometry    configs/cohort_12_nor_geometry.json \\
        --nor-scene-dir   data/cohort_12/NOR_cohort_12

--pred-root defaults to <videos-root>/predictions/. Pass --pred-root
explicitly if predictions live elsewhere. Pass --force-reinference to
overwrite existing prediction files.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Work around the well-known SLEAP/TensorFlow protobuf >= 3.21 incompatibility
# without requiring the user to pin protobuf manually. This env var tells
# protobuf to use the pure-Python parser, which is slower but always compatible.
# Subprocesses we spawn inherit this.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

# Ancestor-folder substrings that mark a video as a supported experiment type.
# Used twice: filter_and_interpolate skips CSV creation for videos whose
# folders match none of these, and the metrics step routes each surviving CSV
# to barnes_metrics or nor_metrics based on the first substring that matches.
# Keep both branches in sync when adding a new experiment type.
RECOGNIZED_SUBSTRINGS = ('barnes', 'nor')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--videos-root',
        default=None,
        help='Cohort folder containing raw videos. When set, inference '
        'runs first for any video that lacks a .predictions.slp file.',
    )
    p.add_argument(
        '--pred-root',
        default=None,
        help='Root containing .predictions.slp files (recursive). '
        'Default: <videos-root>/predictions/ when --videos-root is set.',
    )
    p.add_argument(
        '--force-reinference',
        action='store_true',
        help='Re-run inference even for videos that already have predictions. '
        'Requires --videos-root.',
    )
    p.add_argument(
        '--results-root',
        default=None,
        help='Output root for trajectory + metrics files. Default: <repo>/results.',
    )
    p.add_argument(
        '--cohort',
        default=None,
        help='Cohort label. When set, results are written under '
        "<results-root>/<cohort>/ so multiple cohorts don't collide.",
    )
    p.add_argument(
        '--barnes-geometry',
        default=None,
        help='Cohort Barnes geometry JSON (from annotate_barnes_geometry.py). '
        'If omitted, Barnes metrics are skipped.',
    )
    p.add_argument(
        '--barnes-exclusions',
        default=None,
        help='JSON mapping Barnes video paths (relative to the cohort folder) to '
        'the reason they are excluded from Barnes metrics.',
    )
    p.add_argument(
        '--nor-geometry',
        default=None,
        help='Cohort NOR geometry JSON (object_left + object_right). '
        'If omitted, all NOR metrics are skipped.',
    )
    p.add_argument(
        '--nor-scene-dir',
        default=None,
        help='Folder containing per-video <name>.scene.json files for '
        "test-phase NOR videos. Familiar-phase videos don't need a sidecar.",
    )
    p.add_argument(
        '--fps',
        type=float,
        default=None,
        help="Frame rate to use for every video. Default: read each video's own "
        'frame rate (ffprobe avg_frame_rate).',
    )
    p.add_argument(
        '--max-gap', type=int, default=30, help='Pass-through to filter_and_interpolate.'
    )
    p.add_argument(
        '--collapse-thresh',
        type=float,
        default=15.0,
        help='Pass-through to filter_and_interpolate.',
    )
    p.add_argument(
        '--max-speed', type=float, default=30.0, help='Pass-through to validate_trajectories.'
    )
    p.add_argument(
        '--max-gap-seconds',
        type=float,
        default=5.0,
        help='Pass-through to validate_trajectories: flag videos where BOTH nose '
        'and tail have a consecutive missing-frame run longer than this '
        "(in seconds, using each video's frame rate for the conversion).",
    )
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    scripts = repo / 'scripts'
    results_base = (
        Path(args.results_root).resolve() if args.results_root else (repo / 'results').resolve()
    )
    results_root = (results_base / args.cohort).resolve() if args.cohort else results_base

    py = sys.executable

    videos_root = Path(args.videos_root).resolve() if args.videos_root else None
    if args.pred_root:
        pred_root = Path(args.pred_root).resolve()
    elif videos_root is not None:
        pred_root = videos_root / 'predictions'
    else:
        sys.exit('Must pass at least one of --videos-root or --pred-root.')

    if args.force_reinference and videos_root is None:
        sys.exit('--force-reinference requires --videos-root.')

    # Marker file the UI checks to warn about the soft-fail case in Step 0
    # (inference had failures but we continued with existing predictions).
    # Cleared at pipeline start so it never carries over from a previous run.
    inference_fail_marker = results_root / '.last_inference_had_failures'
    results_root.mkdir(parents=True, exist_ok=True)
    if inference_fail_marker.exists():
        inference_fail_marker.unlink()

    # Step 0 (optional): inference on any video missing a prediction
    if videos_root is not None:
        print('=' * 70)
        print('Step 0/3: inference (skipping videos that already have predictions)')
        print('=' * 70)
        cmd = [
            py,
            str(scripts / 'run_inference.py'),
            '--videos-root',
            str(videos_root),
            '--pred-root',
            str(pred_root),
            '--require-any',
            ','.join(RECOGNIZED_SUBSTRINGS),
        ]
        if args.force_reinference:
            cmd.append('--force')
        r = subprocess.run(cmd)
        if r.returncode != 0:
            # Soft-fail: keep going with whatever predictions exist on disk
            # (from a prior successful run or from a manual sleap-track drop).
            # Leave a marker so the UI can surface a warning banner even
            # though the pipeline exits 0 — stderr warnings alone get missed.
            sys.stderr.write(
                'inference reported failures; continuing with whatever predictions exist.\n'
            )
            inference_fail_marker.write_text(
                'run_inference.py returned a non-zero exit code on the last pipeline run.\n'
                'Some videos may have been skipped or produced no predictions.\n'
                'Expand the pipeline log for details.\n'
            )

    # Step 1
    print('=' * 70)
    print('Step 1/3: filter_and_interpolate')
    print('=' * 70)
    cmd = [
        py,
        str(scripts / 'filter_and_interpolate.py'),
        '--pred-root',
        str(pred_root),
        '--out-dir',
        str(results_root),
        '--max-gap',
        str(args.max_gap),
        '--collapse-thresh',
        str(args.collapse_thresh),
        '--require-any',
        ','.join(RECOGNIZED_SUBSTRINGS),
    ]
    if videos_root is not None:
        cmd += ['--videos-root', str(videos_root)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.stderr.write('filter_and_interpolate failed; aborting.\n')
        sys.exit(1)

    # Step 2
    print('\n' + '=' * 70)
    print('Step 2/3: validate_trajectories')
    print('=' * 70)
    cmd = [
        py,
        str(scripts / 'validate_trajectories.py'),
        '--results-root',
        str(results_root),
        '--max-speed',
        str(args.max_speed),
        '--max-gap-seconds',
        str(args.max_gap_seconds),
    ]
    if args.fps is not None:
        cmd += ['--fps', str(args.fps)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.stderr.write('validate_trajectories failed; aborting.\n')
        sys.exit(1)

    # Step 3: per-video behavioural metrics
    print('\n' + '=' * 70)
    print('Step 3/3: behavioural metrics')
    print('=' * 70)

    nor_dir = Path(args.nor_scene_dir).resolve() if args.nor_scene_dir else None
    geometry_path = Path(args.barnes_geometry).resolve() if args.barnes_geometry else None
    nor_geometry = Path(args.nor_geometry).resolve() if args.nor_geometry else None

    # Video path per trajectory, recorded by filter_and_interpolate.py. Each
    # metrics script reads the frame rate from the video itself.
    video_by_trajectory = {}
    with open(results_root / 'summary.csv') as f:
        for row in csv.DictReader(f):
            if row.get('trajectory') and row.get('fps'):
                video_by_trajectory[row['trajectory']] = row['video_path']

    def fps_args(rel):
        if args.fps is not None:
            return ['--fps', str(args.fps)]
        video = video_by_trajectory.get(rel)
        return ['--video', video] if video else None

    barnes_exclusions = {}
    if args.barnes_exclusions:
        with open(args.barnes_exclusions) as f:
            barnes_exclusions = json.load(f)

    def exclusion_reason(rel):
        video = Path(video_by_trajectory.get(rel) or '').as_posix()
        for key, reason in barnes_exclusions.items():
            if video.endswith('/' + key.lstrip('/')):
                return reason
        return None

    skipped = []
    excluded_barnes = []
    processed_barnes = 0
    processed_nor = 0

    for csv_path in sorted(results_root.rglob('*.csv')):
        name = csv_path.name
        if name in ('summary.csv', 'validation_summary.csv'):
            continue
        if '_validation' in csv_path.stem:
            continue

        rel_parts = csv_path.relative_to(results_root).parts
        dir_parts = [p.lower() for p in rel_parts[:-1]]
        category = ''
        for part in dir_parts:
            for sub in RECOGNIZED_SUBSTRINGS:
                if sub in part:
                    category = sub
                    break
            if category:
                break
        video_name = csv_path.stem
        rel = csv_path.relative_to(results_root).as_posix()
        frame_rate = fps_args(rel)
        if category in RECOGNIZED_SUBSTRINGS and frame_rate is None:
            skipped.append(
                (category, rel, 'no frame rate (video not found; pass --videos-root or --fps)')
            )
            continue

        if category == 'barnes':
            if not geometry_path:
                skipped.append(
                    ('barnes', str(csv_path.relative_to(results_root)), 'no --barnes-geometry')
                )
                continue
            cmd = [
                py,
                str(scripts / 'barnes_metrics.py'),
                '--trajectory',
                str(csv_path),
                '--geometry',
                str(geometry_path),
            ] + frame_rate
            reason = exclusion_reason(rel)
            if reason:
                cmd += ['--exclude-reason', reason]
            r = subprocess.run(cmd)
            if r.returncode == 0 and reason:
                excluded_barnes.append((rel, reason))
            elif r.returncode == 0:
                processed_barnes += 1
            else:
                skipped.append(
                    ('barnes', str(csv_path.relative_to(results_root)), 'metrics failed')
                )
        elif category == 'nor':
            if nor_geometry is None:
                skipped.append(
                    ('nor', str(csv_path.relative_to(results_root)), 'no --nor-geometry')
                )
                continue
            # Classify phase by filename
            is_test = '_novel' in video_name.lower()
            cmd = [
                py,
                str(scripts / 'nor_metrics.py'),
                '--trajectory',
                str(csv_path),
                '--cohort-geometry',
                str(nor_geometry),
            ] + frame_rate
            if is_test:
                if nor_dir is None:
                    skipped.append(
                        (
                            'nor-test',
                            str(csv_path.relative_to(results_root)),
                            'test-phase video but no --nor-scene-dir',
                        )
                    )
                    continue
                scene_candidates = [
                    nor_dir / f'{video_name}.scene.json',
                    nor_dir / video_name / f'{video_name}.scene.json',
                ]
                scene_path = next((c for c in scene_candidates if c.exists()), None)
                if scene_path is None:
                    skipped.append(
                        (
                            'nor-test',
                            str(csv_path.relative_to(results_root)),
                            f'missing {video_name}.scene.json (novel_side)',
                        )
                    )
                    continue
                cmd += ['--scene', str(scene_path)]
            r = subprocess.run(cmd)
            if r.returncode == 0:
                processed_nor += 1
            else:
                skipped.append(('nor', str(csv_path.relative_to(results_root)), 'metrics failed'))
        else:
            skipped.append(('?', str(csv_path.relative_to(results_root)), 'unknown category'))

    # "Pipeline finished" is deliberately neutral: the exit code below decides
    # success or failure, so a run that produces no metrics never exits 0.
    print('\n' + '=' * 70)
    print('Pipeline finished')
    print('=' * 70)
    print(f'Barnes metrics produced: {processed_barnes}')
    print(f'NOR    metrics produced: {processed_nor}')
    if excluded_barnes:
        print(f'\nExcluded {len(excluded_barnes)} Barnes video(s) (marked in their metrics JSON):')
        for path, reason in excluded_barnes:
            print(f'  {path}: {reason}')
    if skipped:
        print(f'\nSkipped {len(skipped)} videos (trajectory CSV still exists):')
        for cat, path, reason in skipped:
            print(f'  [{cat:6s}] {path}: {reason}')

    # Partition skips by reason so we can distinguish "expected config gaps"
    # (missing geometry, missing annotation) from "real problems" (crashed
    # metrics run, uncategorisable video). Only the latter drive non-zero
    # exit codes; expected gaps just mean the researcher hasn't finished
    # annotating yet.
    metrics_failed = [s for s in skipped if s[2] == 'metrics failed']
    unknown_category = [s for s in skipped if s[2] == 'unknown category']
    trajectories_seen = processed_barnes + processed_nor + len(skipped) + len(excluded_barnes)
    produced = processed_barnes + processed_nor

    # Unknown-category videos mean a folder needs renaming. This gets its own
    # exit code (3) so the UI can show a distinct failure state.
    if unknown_category:
        print('\n' + '!' * 70)
        print(f'!! UNKNOWN CATEGORY: {len(unknown_category)} trajectory CSV(s) lived in folders')
        print("!! whose ancestor names contain neither 'barnes' nor 'nor'. The classifier")
        print('!! could not decide which metrics to compute, so they were skipped.')
        print('!! Fix by renaming the containing folder or moving the videos.')
        print('!' * 70)
        for _, path, _ in unknown_category:
            print(f'  - {path}')
        sys.exit(3)

    # Any per-video metrics run that crashed: exit 2.
    if metrics_failed:
        print(
            f'\nERROR: {len(metrics_failed)} per-video metrics run(s) CRASHED. '
            f"See '[barnes] ... metrics failed' entries above. Exiting non-zero."
        )
        sys.exit(2)

    # Trajectories existed but no metrics were produced (e.g. no
    # --barnes-geometry passed, or no NOR test videos annotated yet): exit 2.
    if trajectories_seen > 0 and produced == 0:
        print(
            f'\nERROR: {trajectories_seen} trajectory CSV(s) were found but ZERO metrics were '
            f'produced. Check the skip reasons above — the most common cause is a missing '
            f'--barnes-geometry / --nor-geometry, or an un-annotated NOR test phase. '
            f'Exiting non-zero.'
        )
        sys.exit(2)


if __name__ == '__main__':
    main()
