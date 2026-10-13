"""Frame rate and size of a video, read with ffprobe (installed with ffmpeg in the sleap env)."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


def _ffprobe(video_path, entries):
    ffprobe = shutil.which('ffprobe')
    if ffprobe is None:
        raise RuntimeError('ffprobe not found on PATH. Install ffmpeg, or pass --fps.')
    r = subprocess.run(
        [
            ffprobe,
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            f'stream={entries}',
            '-of',
            'csv=p=0',
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f'ffprobe could not read {video_path}: {r.stderr.strip()}')
    return r.stdout.strip().splitlines()[0].split(',')


def probe_fps(video_path):
    """Average frame rate of the first video stream (ffprobe avg_frame_rate).

    This is the number of frames divided by the stream duration, so for
    variable-frame-rate recordings it gives the mean rate over the video.
    """
    (rate,) = _ffprobe(video_path, 'avg_frame_rate')
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if not fps > 0:
        raise RuntimeError(f'ffprobe gave no usable frame rate for {video_path} ({rate!r}).')
    return fps


def probe_size(video_path):
    """(width, height) in pixels of the first video stream."""
    width, height = _ffprobe(video_path, 'width,height')
    return int(width), int(height)


def find_video(stored_path, videos_root=None):
    """Locate the video a .slp file was made from.

    Uses the path stored in the .slp if it exists on this machine. Otherwise,
    when videos_root is given, looks for a single file with the same name
    under it. Returns a Path, or None when the video cannot be found.
    """
    if not stored_path:
        return None
    p = Path(stored_path)
    if p.exists():
        return p
    if videos_root is None:
        return None
    name = Path(str(stored_path).replace('\\', '/')).name
    matches = sorted(Path(videos_root).rglob(name))
    return matches[0] if len(matches) == 1 else None


def fps_from_args(fps, video):
    """Frame rate for a metrics script and where it came from.

    --fps wins when given. Otherwise the frame rate is read from --video.
    Exits with a message when neither is available.
    """
    if fps is not None:
        return fps, '--fps'
    if video is None:
        raise SystemExit('Pass --video so its frame rate can be read, or set --fps.')
    try:
        return probe_fps(video), 'video'
    except RuntimeError as e:
        raise SystemExit(f'{e} Pass --fps to set the frame rate by hand.')
