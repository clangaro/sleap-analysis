"""
Convert every .mov video under --videos-root to H.264 .mp4 so the rest of the
pipeline reads them reliably.

Why this exists: the lab camera writes .mov. sleap/imageio-ffmpeg accepts .mov
in principle, but ProRes / HEVC / non-yuv420p .mov files often produce wrong
frame counts, silent seek failures, or garbage frames — which shows up
downstream as missing detections that look like model failures. Normalizing to
H.264 yuv420p + faststart .mp4 makes the rest of the pipeline deterministic.

Encoder settings:
  -c:v libx264 -preset medium -crf 18   near-visually-lossless, small enough
  -pix_fmt yuv420p                      broadest downstream compatibility
  -movflags +faststart                  moov atom up front (fast seek)
  -an                                   drop audio (irrelevant for pose)

Usage:
    # Convert everything under a cohort folder in-place (writes .mp4 next to .mov)
    python scripts/convert_mov_to_mp4.py --videos-root data/cohort_12

    # Write .mp4s to a mirrored folder tree instead
    python scripts/convert_mov_to_mp4.py --videos-root data/cohort_12 \\
        --output-root data/cohort_12_mp4

    # Re-encode even if the .mp4 already exists
    python scripts/convert_mov_to_mp4.py --videos-root data/cohort_12 --force

    # Delete the source .mov after a successful conversion (off by default)
    python scripts/convert_mov_to_mp4.py --videos-root data/cohort_12 --delete-source

Folders named 'predictions' or 'results' are skipped (they hold outputs, not
inputs — matches run_inference.py).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MOV_EXTS = {'.mov'}
SKIP_DIR_NAMES = {'predictions', 'results'}


def resolve_ffmpeg() -> str:
    """Prefer imageio-ffmpeg's bundled binary (ships with sleap); fall back to PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    sys.exit(
        'ERROR: could not find an ffmpeg binary.\n'
        '  Either activate the sleap conda env (which bundles imageio-ffmpeg),\n'
        '  or install ffmpeg system-wide: `brew install ffmpeg`.'
    )


def find_movs(videos_root: Path):
    out = []
    for p in sorted(videos_root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in MOV_EXTS:
            continue
        if any(part in SKIP_DIR_NAMES for part in p.relative_to(videos_root).parts):
            continue
        out.append(p)
    return out


def output_path_for(mov: Path, videos_root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        return mov.with_suffix('.mp4')
    rel = mov.relative_to(videos_root)
    return output_root / rel.with_suffix('.mp4')


def convert_one(ffmpeg: str, src: Path, dst: Path, crf: int, preset: str) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Write to a .part file first, rename on success — protects against
    # half-written .mp4s if the process is killed mid-encode.
    tmp = dst.with_suffix(dst.suffix + '.part')
    cmd = [
        ffmpeg,
        '-y',
        '-i',
        str(src),
        '-c:v',
        'libx264',
        '-preset',
        preset,
        '-crf',
        str(crf),
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-an',
        str(tmp),
    ]
    print(f'  ffmpeg -> {dst.name}', flush=True)
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f'  FAILED ({res.returncode}):', file=sys.stderr)
        sys.stderr.write(res.stderr.decode('utf-8', 'replace')[-2000:])
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dst)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--videos-root', type=Path, required=True, help='Folder to scan recursively for .mov files.'
    )
    ap.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Where to write .mp4s (mirrors input tree). Default: alongside each .mov.',
    )
    ap.add_argument(
        '--force', action='store_true', help='Re-encode even if the .mp4 already exists.'
    )
    ap.add_argument(
        '--delete-source',
        action='store_true',
        help='Delete the .mov after a successful conversion. Off by default.',
    )
    ap.add_argument(
        '--crf',
        type=int,
        default=18,
        help='x264 CRF quality (lower = better, larger). Default 18 = visually lossless.',
    )
    ap.add_argument(
        '--preset', default='medium', help='x264 preset (ultrafast..veryslow). Default medium.'
    )
    args = ap.parse_args()

    if not args.videos_root.is_dir():
        sys.exit(f'ERROR: --videos-root {args.videos_root} is not a directory')

    ffmpeg = resolve_ffmpeg()
    movs = find_movs(args.videos_root)
    if not movs:
        print(f'No .mov files under {args.videos_root}')
        return

    print(f'Found {len(movs)} .mov file(s) under {args.videos_root}')
    print(f'Using ffmpeg: {ffmpeg}')

    n_ok = n_skip = n_fail = 0
    for src in movs:
        dst = output_path_for(src, args.videos_root, args.output_root)
        print(f'\n{src.relative_to(args.videos_root)}')
        if dst.exists() and not args.force:
            print(f'  SKIP (exists): {dst}')
            n_skip += 1
            continue
        if convert_one(ffmpeg, src, dst, args.crf, args.preset):
            n_ok += 1
            if args.delete_source:
                src.unlink()
                print(f'  deleted source: {src.name}')
        else:
            n_fail += 1

    print(f'\nDone. converted={n_ok}  skipped={n_skip}  failed={n_fail}')
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
