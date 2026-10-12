"""
Run SLEAP inference on every video under --videos-root that doesn't already
have a matching .predictions.slp file. Outputs mirror the input folder shape.

Uses the pretrained model bundled at model/pretrained_unified_v2/ (relative to
the repo root).

This calls the predictor in Python rather than running sleap-track. The
model's training_config.json lists 'nose_skeleton' twice in
data.labels.skeletons, and with that config sleap-track drops every
detection: the model still finds high-confidence peaks, but the ambiguous
skeleton lookup means no instances are written to the output .slp.

Instead, the predictor's inference model is called directly and
PredictedInstance objects are built against a single clean skeleton. The
resulting .slp files are read normally by filter_and_interpolate.py.

Two backends. SLEAP 1.5+ replaced its TensorFlow inference with the
PyTorch-based sleap-nn package (`uv tool install "sleap[nn]"`); SLEAP 1.4
(the Mac conda env) is TensorFlow only. The backend is detected by trying
`import sleap_nn`. Both paths build instances the same way: the duplicate
skeleton problem was seen with TensorFlow, and sleap-nn has not yet been
checked on real cohort data.

Usage:

    # Infer only missing predictions for a cohort
    python scripts/run_inference.py --videos-root data/cohort_12

    # Force re-inference for every video (overwrites existing predictions)
    python scripts/run_inference.py --videos-root data/cohort_12 --force

    # Custom output location (default is <videos-root>/predictions/)
    python scripts/run_inference.py --videos-root data/cohort_12 --pred-root /elsewhere/preds

Videos under folders named 'predictions' or 'results' are skipped (those are
outputs, not inputs).
"""

import argparse
import sys
from pathlib import Path

# Windows PowerShell's console defaults to cp1252 and crashes with
# UnicodeEncodeError when a filename contains characters outside that codepage
# (e.g. U+F025 and other U+F000-U+F8FF private-use glyphs that Finder/OneDrive
# occasionally injects during cross-platform renames). UTF-8 with error
# replacement keeps the run reportable even if a filename isn't representable.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# .mov is intentionally excluded: sleap_io / imageio-ffmpeg reads some .mov
# containers (ProRes / HEVC / non-yuv420p) unreliably — silent seek failures,
# wrong frame counts, garbage frames — which surface as 0/N detections that
# look like model failures. Convert .mov -> .mp4 upstream with
# scripts/convert_mov_to_mp4.py (bundled ffmpeg on Mac; system ffmpeg on
# Windows), then re-run. Orphan .mov files are logged at start-of-run.
VIDEO_EXTS = {'.mp4', '.avi', '.mkv'}
SKIP_DIR_NAMES = {'predictions', 'results'}

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / 'model' / 'pretrained_unified_v2'

# The skeleton the pretrained model was trained with. Must match the node
# names the model outputs — 'nose' and 'tail'. If you retrain against a new
# skeleton, update BOTH names here.
SKELETON_NAME = 'nose_skeleton'
SKELETON_NODES = ['nose', 'tail']

# Frames per model call. 8 is safe for M-series Metal with 32GB unified RAM
# and typical Linux+CUDA setups; raise if you have headroom for speed.
BATCH_SIZE = 8


def find_videos(videos_root: Path):
    """Recursively find video files, skipping the predictions/ and results/ subtrees.

    .mov is intentionally excluded from VIDEO_EXTS — see the note there. Use
    _warn_orphan_movs to surface .mov files the user forgot to convert.
    """
    out = []
    for p in sorted(videos_root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if any(part in SKIP_DIR_NAMES for part in p.relative_to(videos_root).parts):
            continue
        out.append(p)
    return out


def _warn_orphan_movs(videos_root: Path):
    """Print a helpful notice for any .mov files under videos_root that don't
    have a .mp4 sibling. These would be silently skipped by find_videos.
    """
    orphans = []
    for p in videos_root.rglob('*'):
        if not p.is_file() or p.suffix.lower() != '.mov':
            continue
        if any(part in SKIP_DIR_NAMES for part in p.relative_to(videos_root).parts):
            continue
        if not p.with_suffix('.mp4').exists():
            orphans.append(p)
    if not orphans:
        return
    print(f'\nNOTICE: {len(orphans)} .mov file(s) with no .mp4 sibling — these will be skipped.')
    print('Convert them first:')
    print(f'    python scripts/convert_mov_to_mp4.py --videos-root {videos_root}')
    for p in sorted(orphans)[:10]:
        print(f'  - {p.relative_to(videos_root)}')
    if len(orphans) > 10:
        print(f'  ... and {len(orphans) - 10} more')
    print()


def prediction_path_for(video: Path, videos_root: Path, pred_root: Path) -> Path:
    """Mirror the video's rel path under pred_root, with .predictions.slp suffix."""
    rel = video.relative_to(videos_root)
    return pred_root / rel.with_suffix('.predictions.slp')


def build_skeleton_legacy_tf():
    """Construct the clean skeleton the legacy TF outputs will use (sleap.Skeleton)."""
    from sleap.skeleton import Skeleton

    sk = Skeleton(name=SKELETON_NAME)
    for name in SKELETON_NODES:
        sk.add_node(name)
    for a, b in zip(SKELETON_NODES[:-1], SKELETON_NODES[1:]):
        sk.add_edge(a, b)
    return sk


def _infer_one_video_legacy_tf(video_path: Path, predictor, skeleton, out_path: Path):
    """Legacy TensorFlow (SLEAP <=1.4) inference path.

    Returns (n_frames_total, n_frames_with_instances).
    """
    import numpy as np
    import sleap
    import tensorflow as tf
    from sleap import LabeledFrame, Labels, Video
    from sleap.instance import PredictedInstance, PredictedPoint

    v = sleap.load_video(str(video_path))
    video_ref = Video.from_filename(str(video_path))
    inference_model = predictor.inference_model
    n_frames = v.num_frames

    labeled_frames = []
    for start in range(0, n_frames, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_frames)
        batch = tf.cast(v[start:end], tf.float32)
        out = inference_model(batch)
        # SingleInstancePredictor: peaks shape (B, 1, n_nodes, 2), vals (B, 1, n_nodes).
        peaks = out['instance_peaks'].numpy()
        scores = out['instance_peak_vals'].numpy()

        for local_i, frame_idx in enumerate(range(start, end)):
            xy = peaks[local_i, 0]
            conf = scores[local_i, 0]
            points = {}
            for node_i, node in enumerate(skeleton.nodes):
                x, y = float(xy[node_i, 0]), float(xy[node_i, 1])
                c = float(conf[node_i])
                if not (np.isnan(x) or np.isnan(y)):
                    points[node] = PredictedPoint(x=x, y=y, score=c, visible=True)
            if points:
                inst = PredictedInstance(
                    skeleton=skeleton,
                    points=points,
                    score=float(conf.mean()),
                )
                labeled_frames.append(
                    LabeledFrame(video=video_ref, frame_idx=frame_idx, instances=[inst])
                )

    labels = Labels(labeled_frames=labeled_frames, videos=[video_ref], skeletons=[skeleton])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.save(str(out_path))
    return n_frames, len(labeled_frames)


def _infer_one_video_sleap_nn(video_path: Path, predictor, out_path: Path):
    """Modern PyTorch (sleap-nn 0.3+) inference path.

    Bypasses sleap-nn's builtin label constructor — extracts raw peaks per
    batch and builds sio.PredictedInstance objects ourselves against a clean
    single-skeleton sio.Skeleton (same conservative strategy as the legacy
    TF path).

    Returns (n_frames_processed, n_frames_with_instances).
    """
    import numpy as np
    import sleap_io as sio

    outputs_list = predictor.predict(str(video_path), make_labels=False)

    skeleton = sio.Skeleton(
        nodes=SKELETON_NODES,
        edges=[(SKELETON_NODES[0], SKELETON_NODES[1])],
        name=SKELETON_NAME,
    )
    video_ref = sio.Video.from_filename(str(video_path))

    labeled_frames = []
    n_frames = 0
    for outputs in outputs_list:
        arr = outputs.numpy()
        frame_indices = arr['frame_indices']  # (B,)
        pred_keypoints = arr['pred_keypoints']  # (B, I=1, N, 2), image coords
        pred_peak_values = arr.get('pred_peak_values')  # (B, I=1, N) or None
        for local_i, frame_idx in enumerate(frame_indices):
            n_frames += 1
            pts = pred_keypoints[local_i, 0]  # (N, 2)
            if pred_peak_values is not None:
                vals = pred_peak_values[local_i, 0]  # (N,)
            else:
                vals = np.full(pts.shape[0], np.nan, dtype=np.float32)
            if np.isnan(pts).all():
                continue
            inst = sio.PredictedInstance.from_numpy(
                points_data=pts,
                skeleton=skeleton,
                score=float(np.nansum(vals)),
                point_scores=vals,
            )
            labeled_frames.append(
                sio.LabeledFrame(video=video_ref, frame_idx=int(frame_idx), instances=[inst])
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.Labels(
        videos=[video_ref],
        skeletons=[skeleton],
        labeled_frames=labeled_frames,
    ).save(str(out_path))
    return n_frames, len(labeled_frames)


def _resolve_torch_device() -> str:
    """Pick the best available PyTorch device. sleap-nn accepts a plain string."""
    import torch

    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def infer_missing(
    videos_root: Path, pred_root: Path, force: bool, peak_threshold: float, required_substrings=None
):
    """Run inference for every video whose .predictions.slp is missing.

    If required_substrings is non-empty, videos whose ancestor folder names
    contain none of them are skipped (no inference, no prediction file).

    Returns (ran, skipped, failed).
    """
    if not MODEL_DIR.exists():
        sys.exit(f'Model folder missing: {MODEL_DIR}')

    # Flag any .mov files up front — they're intentionally excluded from
    # inference (see VIDEO_EXTS) and would otherwise be silently skipped.
    _warn_orphan_movs(videos_root)

    videos = find_videos(videos_root)
    if not videos:
        return 0, 0, 0

    required = [s.lower() for s in (required_substrings or []) if s]
    skipped_unrecognized = []
    if required:
        kept = []
        for v in videos:
            dir_parts = [p.lower() for p in v.relative_to(videos_root).parts[:-1]]
            if any(sub in part for part in dir_parts for sub in required):
                kept.append(v)
            else:
                skipped_unrecognized.append(v)
        videos = kept
    if skipped_unrecognized:
        print(
            f'Skipping {len(skipped_unrecognized)} video(s) in unrecognized folders '
            f'(no match for: {", ".join(required)}):'
        )
        for v in skipped_unrecognized:
            print(f'  - {v.relative_to(videos_root)}')

    # Filter first so we only pay for model loading when there's real work to do.
    to_run = []
    skipped = 0
    for v in videos:
        out_path = prediction_path_for(v, videos_root, pred_root)
        if out_path.exists() and not force:
            skipped += 1
        else:
            to_run.append((v, out_path))
    if not to_run:
        return 0, skipped, 0

    # Detect which backend is available. SLEAP 1.5+ ships the PyTorch-based
    # sleap-nn package; SLEAP <=1.4 has only the TensorFlow-based sleap.
    try:
        import sleap_nn  # noqa: F401

        _use_sleap_nn = True
    except ImportError:
        _use_sleap_nn = False

    skeleton = None
    if _use_sleap_nn:
        from sleap_nn.inference import Predictor

        device = _resolve_torch_device()
        print(f'Loading pretrained model (sleap-nn / PyTorch, device={device})...')
        predictor = Predictor.from_model_paths(
            [str(MODEL_DIR)],
            device=device,
            batch_size=BATCH_SIZE,
            peak_threshold=peak_threshold,
        )
    else:
        print('Loading pretrained model (legacy sleap / TensorFlow)...')
        import sleap

        predictor = sleap.load_model(str(MODEL_DIR), peak_threshold=peak_threshold)
        skeleton = build_skeleton_legacy_tf()

    ran, failed = 0, 0
    for v, out_path in to_run:
        print(f'\n>>> {v.name}')
        try:
            if _use_sleap_nn:
                n_frames, n_inst = _infer_one_video_sleap_nn(v, predictor, out_path)
            else:
                n_frames, n_inst = _infer_one_video_legacy_tf(v, predictor, skeleton, out_path)
            print(f'    -> {out_path}: {n_inst}/{n_frames} frames with detections')
            if n_inst == 0:
                # not an error, but zero detections usually means a config
                # or skeleton mismatch
                print(
                    f'    WARNING: 0/{n_frames} detections. Model config or skeleton '
                    'may be wrong for this video.'
                )
            ran += 1
        except Exception as e:
            print(f'!!! inference failed on {v.name}: {type(e).__name__}: {e}')
            failed += 1
    return ran, skipped, failed


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--videos-root', required=True, help='Folder to scan recursively for videos.')
    p.add_argument(
        '--pred-root',
        default=None,
        help='Output root for .predictions.slp files. Default: <videos-root>/predictions/',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='Re-run inference even for videos that already have a prediction.',
    )
    p.add_argument(
        '--peak-threshold',
        type=float,
        default=0.2,
        help='Minimum confidence for a peak to be kept. Passed to the loaded predictor.',
    )
    p.add_argument(
        '--require-any',
        default='',
        help='Comma-separated case-insensitive substrings. If set, only run '
        'inference on videos where at least one ancestor folder name '
        '(relative to --videos-root) contains one of these. Videos in '
        'unrecognized folders are skipped without producing a .predictions.slp.',
    )
    args = p.parse_args()

    videos_root = Path(args.videos_root).resolve()
    if not videos_root.is_dir():
        sys.exit(f'Not a folder: {videos_root}')
    pred_root = Path(args.pred_root).resolve() if args.pred_root else (videos_root / 'predictions')

    required = [s.strip() for s in args.require_any.split(',') if s.strip()]
    ran, skipped, failed = infer_missing(
        videos_root,
        pred_root,
        args.force,
        args.peak_threshold,
        required_substrings=required,
    )

    print('\n' + '=' * 70)
    print(f'Inference done: {ran} ran, {skipped} skipped (already existed), {failed} failed')
    print('=' * 70)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
