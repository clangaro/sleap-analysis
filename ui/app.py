"""
Streamlit UI for the SLEAP analysis pipeline.

Run from the repo root:
    streamlit run ui/app.py

Requires (install once):
    pip install -r ui/requirements.txt
"""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

matplotlib.use('Agg')  # non-interactive backend for headless Streamlit
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# pink theme palette (matches .streamlit/config.toml)
PINK_BG = '#fff0f7'
PINK_PANEL = '#ffe0ef'
PINK_PRIMARY = '#ff4fb8'
PINK_DARK = '#d6007f'
PINK_LIGHT = '#ffb3d9'
PINK_TEXT = '#3a0a24'
PINK_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'pinks', ['#ffb3d9', '#ff4fb8', '#d6007f', '#3a0a24']
)


def pink_fig(figsize=(10, 4)):
    """Return (fig, ax) styled to match the Streamlit pink theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PINK_BG)
    ax.set_facecolor(PINK_PANEL)
    for spine in ax.spines.values():
        spine.set_color(PINK_DARK)
        spine.set_linewidth(1)
    ax.tick_params(colors=PINK_TEXT)
    ax.xaxis.label.set_color(PINK_TEXT)
    ax.yaxis.label.set_color(PINK_TEXT)
    ax.title.set_color(PINK_DARK)
    ax.grid(True, color=PINK_LIGHT, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def pink_bar(df, x_col, y_col, title, xlabel=None, ylabel=None, rotate=45):
    sub = df.dropna(subset=[y_col])
    fig, ax = pink_fig()
    if sub.empty:
        ax.set_title(title + '  (no data)')
        return fig
    ax.bar(sub[x_col].astype(str), sub[y_col], color=PINK_PRIMARY, edgecolor=PINK_DARK, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    plt.xticks(rotation=rotate, ha='right' if rotate else 'center')
    plt.tight_layout()
    return fig


def csv_download(df, filename, label='Download CSV'):
    st.download_button(
        label,
        df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv',
        key=f'dl_{filename}',
    )


REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / 'scripts'
CONFIGS = REPO / 'configs'
RESULTS = REPO / 'results'

st.set_page_config(page_title='SLEAP pipeline', layout='wide')


# helpers


def get_frame(video_path, frame_idx=0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def list_videos(folder, exclude_reencoded=True):
    p = Path(folder)
    if not p.exists():
        return []
    out = sorted(p.glob('*.mp4'))
    if exclude_reencoded:
        out = [v for v in out if '_reencoded' not in v.stem]
    return out


def list_barnes_videos(project_root):
    """Return .mp4 files from any subdir whose name starts with BARNES_DAY (case-insensitive)."""
    out = []
    p = Path(project_root)
    if not p.exists():
        return out
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and sub.name.upper().startswith('BARNES_DAY'):
            out.extend(list_videos(sub))
    return out


def list_nor_videos(project_root):
    """Return .mp4 files from the first subdir whose name starts with NOR (case-insensitive)."""
    p = Path(project_root)
    if not p.exists():
        return []
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and sub.name.upper().startswith('NOR'):
            return list_videos(sub)
    return []


def list_cohorts():
    """Find cohort directories under data/ (each is a folder containing videos + predictions)."""
    data_dir = REPO / 'data'
    if not data_dir.exists():
        return []
    return sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])


def derive_cohort_id(cohort_dir):
    """Pull a short identifier from the folder name, e.g. 'cohort_11' -> '11'."""
    name = cohort_dir.name
    digits = ''
    for c in reversed(name):
        if c.isdigit():
            digits = c + digits
        else:
            break
    return digits or name


def compute_geometry(cx, cy, tx, ty):
    """From a centre click and a target-hole click, return the Barnes geometry dict.

    We do NOT auto-place the other holes — the camera is tilted enough that the
    table is visibly elliptical in the image, so a single-radius circle would
    misplace most holes. Only the actual clicked target position is stored.
    """
    dx, dy = tx - cx, ty - cy
    radius_to_target = math.hypot(dx, dy)
    target_angle = math.degrees(math.atan2(dx, -dy)) % 360.0
    return {
        'centre': {'x': cx, 'y': cy},
        'target_hole': {'x': tx, 'y': ty, 'angle_deg': target_angle},
        'target_angle_deg': target_angle,
        'centre_to_target_px': radius_to_target,
    }


def draw_centre(img, x, y):
    cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.putText(img, 'centre', (int(x) + 15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_target_only(img, x, y):
    cv2.circle(img, (int(x), int(y)), 16, (255, 0, 0), 3)
    cv2.putText(img, 'target', (int(x) + 18, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


def draw_target_zone(img, geom):
    """Draw centre marker, target hole circle, and a translucent target-zone ring
    showing the radius used by barnes_metrics by default (12% of centre→target)."""
    cx, cy = int(geom['centre']['x']), int(geom['centre']['y'])
    tx, ty = int(geom['target_hole']['x']), int(geom['target_hole']['y'])
    zone_r = int(round(0.12 * geom['centre_to_target_px']))
    cv2.line(img, (cx, cy), (tx, ty), (180, 180, 180), 1)
    cv2.drawMarker(img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    cv2.putText(img, 'centre', (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.circle(img, (tx, ty), 16, (255, 0, 0), 3)
    cv2.circle(img, (tx, ty), zone_r, (255, 0, 0), 1)
    cv2.putText(img, 'target', (tx + 18, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)


def draw_marker(img, x, y, label, colour):
    cv2.circle(img, (int(x), int(y)), 14, colour, 3)
    cv2.putText(img, label, (int(x) + 16, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


_STEP_RE = re.compile(r'^Step (\d+)/(\d+):\s*(.+?)\s*$')
_VIDEO_RE = re.compile(r'^\[\s*(\d+)/(\d+)\]\s+(.+?):')
_INFER_RE = re.compile(r'^>>>\s+(.+?)\s*$')


def run_cmd(cmd, status):
    """Stream a subprocess into a Streamlit status container.

    The full raw output is written into the status body via status.write, which
    the user sees only when they expand the container. In parallel, known
    progress markers printed by our own sub-scripts are parsed to keep the
    always-visible status LABEL updated in plain English:
      - "Step X/3: <name>" (from run_pipeline.py) sets the current phase
      - "[ i/N] name: ..." (from filter_and_interpolate.py, validate_trajectories.py)
        updates a per-video counter within the current phase
      - ">>> video.mp4" (from run_inference.py) shows the video currently
        being inferred

    The phase state is stored across lines so that the individual-step
    buttons (which don't print "Step X/3" headers) still get sensible
    per-video labels via the video-counter regex.
    """
    phase_verb = 'Processing'  # what we're doing to each item
    phase_headline = None  # top-level "we're currently…" label

    # PYTHONUNBUFFERED=1 + bufsize=1 together defeat Python's default full
    # buffering on piped stdout, which otherwise holds several KB of output
    # until the child exits — making the UI look frozen while the pipeline
    # is actually making progress. Env vars propagate to grandchildren, so
    # this covers run_pipeline.py and every run_inference.py / filter / etc
    # subprocess it spawns.
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        status.write(line)

        m = _STEP_RE.match(line)
        if m:
            name = m.group(3).lower()
            if 'inference' in name:
                phase_verb = 'Running inference on'
                phase_headline = 'Running inference — first pass is 1–3 min per video…'
            elif 'filter' in name:
                phase_verb = 'Filtering trajectory'
                phase_headline = 'Filtering and interpolating trajectories…'
            elif 'validate' in name:
                phase_verb = 'Validating trajectory'
                phase_headline = 'Validating trajectories…'
            elif 'metric' in name or 'behavioural' in name:
                phase_verb = 'Computing metrics for'
                phase_headline = 'Computing behavioural metrics…'
            if phase_headline:
                status.update(label=phase_headline)
            continue

        m = _VIDEO_RE.match(line)
        if m:
            i, n, video = m.groups()
            status.update(label=f'{phase_verb} {i}/{n}: {video}')
            continue

        m = _INFER_RE.match(line)
        if m:
            status.update(label=f'Running inference on: {m.group(1)}')
            continue

    proc.wait()
    return proc.returncode


def run_with_status(
    cmd,
    initial_label='Starting…',
    success_label='Pipeline complete',
    failure_label='Pipeline failed — expand for the log',
):
    """Run a subprocess inside a collapsed st.status container.

    Log is hidden by default (expanded=False). On success the container stays
    collapsed with a green tick + success label. On failure it auto-expands
    so the researcher sees the log immediately without having to click.
    """
    with st.status(initial_label, expanded=False) as status:
        rc = run_cmd(cmd, status)
        if rc == 0:
            status.update(label=success_label, state='complete', expanded=False)
        else:
            status.update(label=failure_label, state='error', expanded=True)
    return rc


# sidebar config

st.sidebar.title('Configuration')

cohorts = list_cohorts()
if not cohorts:
    st.sidebar.error(
        'No cohorts found under `data/`.\n\n'
        'Add a cohort by creating a folder under `data/` (e.g. `data/cohort_12/`) '
        'containing your `BARNES_DAY*` and `NOR_*` subfolders, plus a `predictions/` '
        'subfolder for the SLEAP inference files. See `data/README.md` for details.'
    )
    st.stop()

cohort_dir = st.sidebar.selectbox('Cohort', cohorts, format_func=lambda p: p.name)
project_root = str(cohort_dir)
predictions_root = str(cohort_dir / 'predictions')
cohort_id = derive_cohort_id(cohort_dir)
cohort_results = RESULTS / cohort_dir.name
st.sidebar.caption(f'Cohort ID: `{cohort_id}`')
st.sidebar.caption(f'Predictions: `{predictions_root}`')
if not Path(predictions_root).exists():
    st.sidebar.warning(
        f'`predictions/` not found inside this cohort. Drop the SLEAP '
        f'`.predictions.slp` files into `data/{cohort_dir.name}/predictions/` '
        f'and reload.'
    )

st.sidebar.markdown('---')
st.sidebar.markdown('### Pipeline parameters')
fps_override = st.sidebar.checkbox(
    'Set the frame rate by hand',
    value=False,
    help="By default each video's own frame rate is read from the video file. "
    'Tick to use the value below for every video instead.',
)
fps = st.sidebar.number_input('FPS', value=30.0, step=1.0, disabled=not fps_override)
fps_args = ['--fps', str(fps)] if fps_override else []
max_gap = st.sidebar.number_input('Max gap (frames) for interpolation', value=30, step=5)
collapse_thresh = st.sidebar.number_input('Collapse threshold (px)', value=15.0, step=1.0)
max_speed = st.sidebar.number_input(
    'Max speed (px/frame) for suspicious flag', value=30.0, step=5.0
)
max_gap_seconds = st.sidebar.number_input(
    'Max continuous missing-gap (seconds) before flagging',
    value=5.0,
    step=1.0,
    min_value=0.0,
    help='A video is flagged when BOTH nose and tail have a consecutive run of '
    'missing (post-interpolation) frames longer than this. AND — not OR — '
    'so short single-keypoint occlusions (e.g. tail out of frame during a '
    "rear) don't trigger the flag; only videos where both keypoints have a "
    'long blackout do. Flagged videos still contribute to metrics; the flag '
    'is a review prompt, not an auto-exclusion.',
)

geom_path = CONFIGS / f'cohort_{cohort_id}_barnes_geometry.json'
# Optional: Barnes videos the cohort geometry does not fit, mapped to a reason.
exclusions_path = CONFIGS / f'cohort_{cohort_id}_barnes_exclusions.json'
nor_geom_sidebar_path = CONFIGS / f'cohort_{cohort_id}_nor_geometry.json'
nor_videos = list_nor_videos(project_root)
nor_test_videos_sidebar = [v for v in nor_videos if '_novel' in v.stem.lower()]
nor_test_done = [v for v in nor_test_videos_sidebar if (v.parent / f'{v.stem}.scene.json').exists()]

st.sidebar.markdown('### Annotation status')
st.sidebar.markdown(f'Barnes geometry: {"set" if geom_path.exists() else "not set"}')
st.sidebar.markdown(
    f'NOR cohort calibration: {"set" if nor_geom_sidebar_path.exists() else "not set"}'
)
st.sidebar.markdown(
    f'NOR test-phase sides annotated: {len(nor_test_done)} / {len(nor_test_videos_sidebar)}'
)


# main tabs

tab_protocol, tab1, tab2, tab3, tab4 = st.tabs(
    [
        'Protocol / How to use',
        '1. Barnes geometry',
        '2. NOR scenes',
        '3. Run pipeline',
        '4. Results',
    ]
)


# Tab 1: Barnes geometry

with tab1:
    st.header('Barnes maze geometry calibration')
    st.caption(
        'Pick the centre and the target hole one at a time. '
        "Click a 'Pick …' button to enter pick-mode, then click the corresponding "
        'spot on the image. You can re-pick either point at any time.'
    )

    # Initialise state
    for key in ('barnes_centre', 'barnes_target'):
        if key not in st.session_state:
            st.session_state[key] = None
    if 'barnes_mode' not in st.session_state:
        st.session_state.barnes_mode = 'idle'

    videos = list_barnes_videos(project_root)
    if not videos:
        st.warning('No Barnes videos found under the configured Cohort folder.')
    else:
        # Compact control row on top
        ctrl_left, ctrl_right = st.columns([3, 2])

        with ctrl_left:
            video = st.selectbox(
                'Calibration video',
                videos,
                format_func=lambda p: p.parent.name + '/' + p.name,
                key='barnes_video_select',
            )
            frame_idx = st.number_input(
                'Frame to display', value=51, step=1, key='barnes_frame_idx'
            )

            b1, b2, b3 = st.columns(3)
            centre_done = st.session_state.barnes_centre is not None
            target_done = st.session_state.barnes_target is not None
            centre_active = st.session_state.barnes_mode == 'picking_centre'
            target_active = st.session_state.barnes_mode == 'picking_target'

            centre_label = 'Pick centre' + (' (set)' if centre_done else '')
            target_label = 'Pick target hole' + (' (set)' if target_done else '')

            if b1.button(
                centre_label,
                key='barnes_pick_centre',
                type='primary' if centre_active else 'secondary',
            ):
                st.session_state.barnes_mode = 'idle' if centre_active else 'picking_centre'
                st.rerun()
            if b2.button(
                target_label,
                key='barnes_pick_target',
                type='primary' if target_active else 'secondary',
            ):
                st.session_state.barnes_mode = 'idle' if target_active else 'picking_target'
                st.rerun()
            if b3.button('Reset all', key='barnes_reset_all'):
                st.session_state.barnes_centre = None
                st.session_state.barnes_target = None
                st.session_state.barnes_mode = 'idle'
                st.rerun()

            if centre_active:
                st.info('Click the table **centre** on the image below.')
            elif target_active:
                st.info('Click the **target hole** on the image below.')
            else:
                st.caption("Pick-mode: idle. Click a 'Pick …' button to start.")

        with ctrl_right:
            st.markdown('### Current selections')
            if centre_done:
                cx_disp, cy_disp = st.session_state.barnes_centre
                st.write(f'**Centre**: ({cx_disp:.0f}, {cy_disp:.0f})')
            else:
                st.write('**Centre**: _not set_')
            if target_done:
                tx_disp, ty_disp = st.session_state.barnes_target
                st.write(f'**Target hole**: ({tx_disp:.0f}, {ty_disp:.0f})')
            else:
                st.write('**Target hole**: _not set_')

            if centre_done and target_done:
                cx_disp, cy_disp = st.session_state.barnes_centre
                tx_disp, ty_disp = st.session_state.barnes_target
                preview_disp = compute_geometry(cx_disp, cy_disp, tx_disp, ty_disp)
                st.markdown(
                    f'Derived target angle: **{preview_disp["target_angle_deg"]:.1f}°** '
                    f'clockwise from vertical.  '
                    f'Centre→target distance: **{preview_disp["centre_to_target_px"]:.0f} px**.'
                )
                st.caption(
                    'Only the centre and the target hole are stored — the camera tilt '
                    "makes the table elliptical, so we don't try to fit the other 19 holes."
                )
                st.success('Both points set — verify on the image, then save.')
                if st.button('Save geometry', type='primary', key='barnes_save'):
                    out = {
                        'cohort': cohort_id,
                        'calibration_video': str(video),
                        'calibration_frame': int(frame_idx),
                        'image_width': int(get_frame(video, frame_idx).shape[1]),
                        'image_height': int(get_frame(video, frame_idx).shape[0]),
                        'convention_note': 'angle measured clockwise from image-vertical-up.',
                    }
                    out.update(preview_disp)
                    CONFIGS.mkdir(parents=True, exist_ok=True)
                    with open(geom_path, 'w') as f:
                        json.dump(out, f, indent=2)
                    st.success(f'Saved to {geom_path}')

        # Full-width image (outside the control columns so it spans the page).
        frame = get_frame(video, frame_idx)
        if frame is None:
            st.error(f'Could not read frame {frame_idx}.')
        else:
            img = frame.copy()
            if centre_done and target_done:
                cx, cy = st.session_state.barnes_centre
                tx, ty = st.session_state.barnes_target
                draw_target_zone(img, compute_geometry(cx, cy, tx, ty))
            else:
                if centre_done:
                    cx, cy = st.session_state.barnes_centre
                    draw_centre(img, cx, cy)
                if target_done:
                    tx, ty = st.session_state.barnes_target
                    draw_target_only(img, tx, ty)

            value = streamlit_image_coordinates(img, key='barnes_clicker')
            if value:
                if centre_active:
                    st.session_state.barnes_centre = (value['x'], value['y'])
                    st.session_state.barnes_mode = 'idle'
                    st.rerun()
                elif target_active:
                    st.session_state.barnes_target = (value['x'], value['y'])
                    st.session_state.barnes_mode = 'idle'
                    st.rerun()


# Tab 2: NOR

nor_geom_path = CONFIGS / f'cohort_{cohort_id}_nor_geometry.json'
nor_test_videos = [v for v in nor_videos if '_novel' in v.stem.lower()]
nor_familiar_videos = [v for v in nor_videos if '_familiar' in v.stem.lower()]


def _scene_path_for(v):
    return v.parent / f'{v.stem}.scene.json'


with tab2:
    st.header('NOR scene annotation')
    st.caption(
        'Two-step NOR setup. Step 1 (cohort calibration) is done once: '
        'click the two object positions on a *familiar* video, which are the same '
        'across the cohort. Step 2 (test-phase) is done per *novel video*: pick '
        'which side carried the novel object on that trial.'
    )

    # Step 1: cohort calibration
    st.subheader('Step 1. Cohort calibration (once)')

    if 'nor_left' not in st.session_state:
        st.session_state.nor_left = None
    if 'nor_right' not in st.session_state:
        st.session_state.nor_right = None
    if 'nor_mode' not in st.session_state:
        st.session_state.nor_mode = 'idle'

    # If config exists and we haven't loaded it into the editor yet, preload.
    if nor_geom_path.exists() and st.session_state.get('nor_geom_loaded') != str(nor_geom_path):
        try:
            g = json.loads(nor_geom_path.read_text())
            st.session_state.nor_left = (g['object_left']['x'], g['object_left']['y'])
            st.session_state.nor_right = (g['object_right']['x'], g['object_right']['y'])
            st.session_state.nor_geom_loaded = str(nor_geom_path)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if not nor_familiar_videos:
        st.warning('No *_familiar.mp4 videos found — needed to calibrate the cohort.')
    else:
        cohort_collapsed = nor_geom_path.exists() and not st.session_state.get('nor_recal', False)
        with st.expander(
            'Calibrate cohort'
            + (' (calibration found — expand to re-do)' if nor_geom_path.exists() else ''),
            expanded=not cohort_collapsed,
        ):
            col_l, col_r = st.columns([3, 2])
            with col_l:
                cal_video = st.selectbox(
                    'Calibration video (a familiar one)',
                    nor_familiar_videos,
                    format_func=lambda p: p.name,
                    key='nor_cal_video',
                )
                cal_frame = st.number_input(
                    'Frame to display', value=0, step=1, key='nor_cal_frame'
                )

                left_done = st.session_state.nor_left is not None
                right_done = st.session_state.nor_right is not None
                left_active = st.session_state.nor_mode == 'picking_left'
                right_active = st.session_state.nor_mode == 'picking_right'

                b1, b2, b3 = st.columns(3)
                left_label = 'Pick left object' + (' (set)' if left_done else '')
                right_label = 'Pick right object' + (' (set)' if right_done else '')
                if b1.button(
                    left_label, key='nor_pick_left', type='primary' if left_active else 'secondary'
                ):
                    st.session_state.nor_mode = 'idle' if left_active else 'picking_left'
                    st.rerun()
                if b2.button(
                    right_label,
                    key='nor_pick_right',
                    type='primary' if right_active else 'secondary',
                ):
                    st.session_state.nor_mode = 'idle' if right_active else 'picking_right'
                    st.rerun()
                if b3.button('Reset', key='nor_reset_cohort'):
                    st.session_state.nor_left = None
                    st.session_state.nor_right = None
                    st.session_state.nor_mode = 'idle'
                    st.rerun()

                if left_active:
                    st.info('Click the **left object** on the image below.')
                elif right_active:
                    st.info('Click the **right object** on the image below.')
                else:
                    st.caption("Pick-mode: idle. Click a 'Pick …' button to start.")

            with col_r:
                st.markdown('### Current selections')
                if left_done:
                    lx_disp, ly_disp = st.session_state.nor_left
                    st.write(f'**Left object**: ({lx_disp:.0f}, {ly_disp:.0f})')
                else:
                    st.write('**Left object**: _not set_')
                if right_done:
                    rx_disp, ry_disp = st.session_state.nor_right
                    st.write(f'**Right object**: ({rx_disp:.0f}, {ry_disp:.0f})')
                else:
                    st.write('**Right object**: _not set_')

                if left_done and right_done:
                    st.success('Both positions set — verify on the image, then save.')
                    if st.button('Save cohort calibration', type='primary', key='nor_save_cohort'):
                        lx, ly = st.session_state.nor_left
                        rx, ry = st.session_state.nor_right
                        ref_frame = get_frame(cal_video, cal_frame)
                        out = {
                            'cohort': cohort_id,
                            'calibration_video': str(cal_video),
                            'calibration_frame': int(cal_frame),
                            'image_width': int(ref_frame.shape[1]),
                            'image_height': int(ref_frame.shape[0]),
                            'object_left': {'x': lx, 'y': ly},
                            'object_right': {'x': rx, 'y': ry},
                        }
                        CONFIGS.mkdir(parents=True, exist_ok=True)
                        with open(nor_geom_path, 'w') as f:
                            json.dump(out, f, indent=2)
                        st.success(f'Saved to {nor_geom_path}')
                        st.session_state.nor_recal = False

        # Full-width image for cohort calibration (outside the control columns).
        frame_cal = get_frame(cal_video, cal_frame)
        if frame_cal is None:
            st.error(f'Could not read frame {cal_frame}.')
        else:
            img_cal = frame_cal.copy()
            if left_done:
                lx, ly = st.session_state.nor_left
                draw_marker(img_cal, lx, ly, 'left', (0, 200, 0))
            if right_done:
                rx, ry = st.session_state.nor_right
                draw_marker(img_cal, rx, ry, 'right', (255, 0, 0))
            value = streamlit_image_coordinates(img_cal, key='nor_cohort_clicker')
            if value:
                if left_active:
                    st.session_state.nor_left = (value['x'], value['y'])
                    st.session_state.nor_mode = 'idle'
                    st.rerun()
                elif right_active:
                    st.session_state.nor_right = (value['x'], value['y'])
                    st.session_state.nor_mode = 'idle'
                    st.rerun()

    # Step 2: test-phase annotation
    st.subheader('Step 2. Test-phase video annotation')

    if not nor_geom_path.exists():
        st.warning('Calibrate the cohort first (Step 1 above).')
    elif not nor_test_videos:
        st.info('No *_novel.mp4 videos found under the configured Cohort folder.')
    else:
        cohort_geom = json.loads(nor_geom_path.read_text())
        left_pos = (cohort_geom['object_left']['x'], cohort_geom['object_left']['y'])
        right_pos = (cohort_geom['object_right']['x'], cohort_geom['object_right']['y'])

        status_rows = []
        for v in nor_test_videos:
            sp = _scene_path_for(v)
            if sp.exists():
                d = json.loads(sp.read_text())
                status_rows.append(
                    {'video': v.name, 'novel_side': d.get('novel_side', '?'), 'annotated': 'yes'}
                )
            else:
                status_rows.append({'video': v.name, 'novel_side': '—', 'annotated': 'no'})
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            test_video = st.selectbox(
                'Test-phase video',
                nor_test_videos,
                format_func=lambda p: p.name,
                key='nor_test_video',
            )
            test_frame = st.number_input('Frame to display', value=0, step=1, key='nor_test_frame')

        with col_r:
            sp = _scene_path_for(test_video)
            existing_side = None
            if sp.exists():
                try:
                    existing_side = json.loads(sp.read_text()).get('novel_side')
                except (OSError, ValueError, AttributeError):
                    pass

            choice = st.radio(
                'Which side has the **novel** object on this trial?',
                options=['left', 'right'],
                horizontal=True,
                index=(0 if existing_side == 'left' else 1 if existing_side == 'right' else 0),
                key=f'nor_side_{test_video.stem}',
            )
            st.caption(
                f'Familiar will be the other side ({"right" if choice == "left" else "left"}).'
            )
            if st.button('Save side', type='primary', key='nor_save_side'):
                out = {
                    'video': str(test_video),
                    'phase': 'test',
                    'novel_side': choice,
                    'familiar_side': 'right' if choice == 'left' else 'left',
                }
                with open(sp, 'w') as f:
                    json.dump(out, f, indent=2)
                st.success(f'Saved to {sp}')

        # Full-width image showing both cohort positions labelled.
        frame_test = get_frame(test_video, test_frame)
        if frame_test is None:
            st.error(f'Could not read frame {test_frame}.')
        else:
            img_test = frame_test.copy()
            draw_marker(img_test, *left_pos, 'LEFT', (255, 220, 0))
            draw_marker(img_test, *right_pos, 'RIGHT', (255, 220, 0))
            st.image(img_test, channels='RGB', use_container_width=True)


# Tab 3: Run pipeline

with tab3:
    st.header('Run pipeline')
    st.markdown(
        'Streams subprocess output below. Re-running just adds metrics for newly '
        'annotated videos; it does not redo finished trajectory work.'
    )

    run_inference = st.checkbox(
        'Run inference on videos missing predictions',
        value=True,
        help='Runs SLEAP inference (bundled pretrained model) on any .mp4 under '
        "this cohort folder that doesn't already have a matching "
        '.predictions.slp file. Uncheck if you already have all your '
        'predictions and just want the analysis.',
    )
    force_reinfer = False
    if run_inference:
        force_reinfer = st.checkbox(
            'Force re-inference (overwrite existing predictions)',
            value=False,
            help='Slow. Only tick if the model changed or you know the existing '
            'predictions are wrong.',
        )

    cmd_full = [
        sys.executable,
        str(SCRIPTS / 'run_pipeline.py'),
        '--pred-root',
        predictions_root,
        '--results-root',
        str(RESULTS),
        '--cohort',
        cohort_dir.name,
        *fps_args,
        '--max-gap',
        str(max_gap),
        '--collapse-thresh',
        str(collapse_thresh),
        '--max-speed',
        str(max_speed),
        '--max-gap-seconds',
        str(max_gap_seconds),
    ]
    if run_inference:
        cmd_full += ['--videos-root', project_root]
        if force_reinfer:
            cmd_full += ['--force-reinference']
    if geom_path.exists():
        cmd_full += ['--barnes-geometry', str(geom_path)]
    if exclusions_path.exists():
        cmd_full += ['--barnes-exclusions', str(exclusions_path)]
    if nor_geom_sidebar_path.exists():
        cmd_full += ['--nor-geometry', str(nor_geom_sidebar_path)]
    if nor_videos:
        cmd_full += ['--nor-scene-dir', str(nor_videos[0].parent)]

    st.code('python ' + ' '.join(c for c in cmd_full[1:]), language='bash')

    if not geom_path.exists():
        st.warning('Barnes geometry not annotated yet — Barnes metrics will be skipped.')
    if not nor_geom_sidebar_path.exists():
        st.warning('NOR cohort calibration not done yet — all NOR metrics will be skipped.')
    elif not nor_test_done:
        st.warning(
            'No NOR test-phase sides annotated yet — test-phase metrics will be skipped '
            '(familiar-phase ones will still run).'
        )

    if st.button('Run full pipeline', type='primary', key='run_pipeline'):
        run_with_status(cmd_full, initial_label='Starting pipeline…')

    st.markdown('---')
    st.markdown('### Individual steps')
    if st.button('Filter + interpolate only'):
        run_with_status(
            [
                sys.executable,
                str(SCRIPTS / 'filter_and_interpolate.py'),
                '--pred-root',
                predictions_root,
                '--out-dir',
                str(cohort_results),
                '--max-gap',
                str(max_gap),
                '--collapse-thresh',
                str(collapse_thresh),
            ],
            initial_label='Filtering and interpolating trajectories…',
            success_label='Filter + interpolate complete',
            failure_label='Filter + interpolate failed — expand for the log',
        )
    if st.button('Validate trajectories only'):
        run_with_status(
            [
                sys.executable,
                str(SCRIPTS / 'validate_trajectories.py'),
                '--results-root',
                str(cohort_results),
                '--max-speed',
                str(max_speed),
                '--max-gap-seconds',
                str(max_gap_seconds),
                *fps_args,
            ],
            initial_label='Validating trajectories…',
            success_label='Validation complete',
            failure_label='Validation failed — expand for the log',
        )


# Tab 4: Results

with tab4:
    st.header('Results')

    summary_csv = cohort_results / 'summary.csv'
    val_csv = cohort_results / 'validation_summary.csv'

    # Pre-load the validation summary once so we can (a) render the
    # data-quality banner above every table and (b) reuse the DataFrame
    # in the Validation section below without reading the CSV twice.
    vf = pd.read_csv(val_csv) if val_csv.exists() else None

    # Inference-failure banner
    # run_pipeline.py leaves a marker file when Step 0 (inference) returned
    # non-zero but the pipeline kept going because some predictions already
    # existed on disk. The exit code is still 0 in that case (soft-fail),
    # so without this banner the researcher would trust results that may
    # have been computed from stale/partial predictions.
    inference_fail_marker = cohort_results / '.last_inference_had_failures'
    if inference_fail_marker.exists():
        st.warning(
            'Inference reported failures during the last pipeline run — some videos '
            'may have been skipped or produced no predictions. The results below were '
            'computed from whatever predictions were already on disk (which may be stale '
            'or partial). Expand the pipeline log in Tab 3 for details, or re-run with '
            '**Force re-inference** ticked to try again.'
        )

    # Data-quality banner (missingness flags)
    # High-missingness videos still contribute to behavioural metrics —
    # this is a review prompt, not an auto-exclusion. See validate_trajectories.py
    # for the AND-not-OR flagging rationale.
    if vf is not None and 'high_missingness' in vf.columns:
        flagged = vf[vf['high_missingness'] == 1]
        if not flagged.empty:
            lines = [
                f'- **{row["video"]}** — longest missing gaps: '
                f'nose {row["nose_longest_gap_sec"]}s, tail {row["tail_longest_gap_sec"]}s'
                for _, row in flagged.iterrows()
            ]
            st.warning(
                f'{len(flagged)} video(s) exceeded the missing-gap threshold '
                f'({max_gap_seconds:g}s) on BOTH nose and tail. Their behavioural '
                f'metrics may be distorted — review before trusting:\n\n'
                + '\n'.join(lines)
                + '\n\n'
                '**Recommended action.** Open each flagged video in `sleap-label`, '
                'scrub to the missing gap, and manually add or correct the nose/tail '
                'points across that stretch. Then export the corrected predictions '
                'and save the file next to the original with `_corrected` inserted '
                'before the `.predictions.slp` suffix — e.g. '
                '`110_Trial_1.predictions.slp` → `110_Trial_1_corrected.predictions.slp`, '
                'in the same predictions folder. Re-running the pipeline will pick up '
                'the corrected file automatically and produce a parallel set of '
                'metrics without overwriting the original run, so you can compare '
                'corrected vs. original side-by-side in this tab.'
            )

    # Version filter (Original vs Corrected)
    # Corrected trajectories come from RAs re-exporting a hand-fixed .slp with
    # `_corrected` inserted before `.predictions.slp` (see the missingness
    # banner above for the workflow). Filter naming is a strict "stem ends
    # with _corrected" check — no metadata files, no folder magic. If a real
    # video is genuinely named `..._corrected.mp4` it will be misclassified,
    # but that's the accepted edge case for a zero-code-change ingest.
    def _is_corrected(stem):
        return str(stem).endswith('_corrected')

    def _apply_version_filter(dframe, video_col, choice):
        """Add a Version column (first position) and filter to the chosen slice."""
        if dframe is None or dframe.empty:
            return dframe
        out = dframe.copy()
        out['Version'] = out[video_col].apply(
            lambda v: 'Corrected' if _is_corrected(v) else 'Original'
        )
        # Move Version to the front for at-a-glance identification.
        cols = ['Version'] + [c for c in out.columns if c != 'Version']
        out = out[cols]
        if choice == 'Original only':
            out = out[out['Version'] == 'Original']
        elif choice == 'Corrected only':
            out = out[out['Version'] == 'Corrected']
        return out

    version_filter = st.radio(
        'Show:',
        options=['All', 'Original only', 'Corrected only'],
        horizontal=True,
        key='version_filter',
        help='Rows tagged **Corrected** come from `<stem>_corrected.predictions.slp` '
        'files an RA has hand-corrected in sleap-label. Both versions coexist; '
        'use this filter to compare like-for-like.',
    )

    # Filter / interpolation summary
    if summary_csv.exists():
        st.subheader('Filter / interpolation summary')
        sf = pd.read_csv(summary_csv)
        sf = _apply_version_filter(sf, 'video', version_filter)
        if sf is None or sf.empty:
            st.info(f'No filter-summary rows match version filter: {version_filter}.')
        else:
            st.dataframe(sf, use_container_width=True, hide_index=True)
            csv_download(sf, 'filter_summary.csv', 'Download filter summary as CSV')
            if 'collapsed' in sf.columns:
                st.pyplot(
                    pink_bar(sf, 'video', 'collapsed', 'Collapsed nose/tail predictions per video'),
                    clear_figure=True,
                )
            if 'nose_interpolated' in sf.columns:
                st.pyplot(
                    pink_bar(
                        sf, 'video', 'nose_interpolated', 'Frames where the nose was interpolated'
                    ),
                    clear_figure=True,
                )
    else:
        st.info('No filter summary yet. Run the pipeline first.')

    # Validation summary
    if vf is not None:
        st.subheader('Validation summary (speed flags + missingness)')
        vf_view = _apply_version_filter(vf, 'video', version_filter)
        if vf_view is None or vf_view.empty:
            st.info(f'No validation-summary rows match version filter: {version_filter}.')
        else:
            st.dataframe(vf_view, use_container_width=True, hide_index=True)
            csv_download(vf_view, 'validation_summary.csv', 'Download validation summary as CSV')
            if 'nose_pct_suspicious' in vf_view.columns:
                st.pyplot(
                    pink_bar(
                        vf_view,
                        'video',
                        'nose_pct_suspicious',
                        '% nose frames flagged as biologically implausible',
                    ),
                    clear_figure=True,
                )

    # Behavioural metrics
    st.subheader('Behavioural metrics')
    metric_files = sorted(cohort_results.rglob('*.barnes_metrics.json')) + sorted(
        cohort_results.rglob('*.nor_metrics.json')
    )
    if not metric_files:
        st.info('No per-video metrics yet. Annotate the cohort and re-run the pipeline.')
    else:
        rows = []
        for mf in metric_files:
            d = json.loads(mf.read_text())
            d['_file'] = str(mf.relative_to(cohort_results))
            d['task'] = 'barnes' if mf.name.endswith('.barnes_metrics.json') else 'nor'
            # Derive the video stem from the metrics filename so the version
            # filter has a column to key off, even if the metrics JSON didn't
            # set its own "video" field. setdefault preserves any existing one.
            suffix = '.barnes_metrics.json' if d['task'] == 'barnes' else '.nor_metrics.json'
            d.setdefault('video', mf.name[: -len(suffix)])
            rows.append(d)
        df = pd.DataFrame(rows)
        df = _apply_version_filter(df, 'video', version_filter)
        if df is None or df.empty:
            st.info(f'No behavioural-metrics rows match version filter: {version_filter}.')
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv_download(df, 'behavioural_metrics.csv', 'Download all metrics as CSV')

            # Barnes plots. Excluded videos stay in the table above but have no metrics to plot.
            barnes = df[df['task'] == 'barnes']
            if 'excluded' in barnes.columns:
                barnes = barnes[~barnes['excluded'].eq(True)]
            if not barnes.empty:
                st.markdown('#### Barnes')
                if 'latency_to_target_sec' in barnes.columns:
                    st.pyplot(
                        pink_bar(
                            barnes,
                            'video',
                            'latency_to_target_sec',
                            'Barnes — latency to first target interaction (s)',
                        ),
                        clear_figure=True,
                    )
                if 'time_at_target_sec' in barnes.columns:
                    st.pyplot(
                        pink_bar(
                            barnes,
                            'video',
                            'time_at_target_sec',
                            'Barnes — total time at target zone (s)',
                        ),
                        clear_figure=True,
                    )
                if 'target_quadrant_pct' in barnes.columns:
                    st.pyplot(
                        pink_bar(
                            barnes,
                            'video',
                            'target_quadrant_pct',
                            'Barnes — % of valid frames in the target quadrant',
                        ),
                        clear_figure=True,
                    )

            # NOR plots — separate familiar and test phase
            nor = df[df['task'] == 'nor']
            if not nor.empty:
                st.markdown('#### NOR — test phase')
                nor_test = nor[nor.get('phase', '') == 'test']
                if not nor_test.empty and 'discrimination_index' in nor_test.columns:
                    st.pyplot(
                        pink_bar(
                            nor_test,
                            'video',
                            'discrimination_index',
                            'NOR — Discrimination Index (test phase)',
                        ),
                        clear_figure=True,
                    )
                if not nor_test.empty and 'time_novel_sec' in nor_test.columns:
                    st.pyplot(
                        pink_bar(
                            nor_test,
                            'video',
                            'time_novel_sec',
                            'NOR — time investigating novel object (s)',
                        ),
                        clear_figure=True,
                    )

                st.markdown('#### NOR — familiar phase')
                nor_fam = nor[nor.get('phase', '') == 'familiar']
                if not nor_fam.empty and 'left_right_bias' in nor_fam.columns:
                    st.pyplot(
                        pink_bar(
                            nor_fam,
                            'video',
                            'left_right_bias',
                            'NOR familiar — left/right side bias  (positive = left)',
                        ),
                        clear_figure=True,
                    )

    # Trajectory viewer
    st.subheader('Trajectory viewer')
    traj_files = sorted(
        [
            p
            for p in cohort_results.rglob('*.csv')
            if p.name not in ('summary.csv', 'validation_summary.csv')
            and '_validation' not in p.stem
        ]
    )
    if version_filter == 'Original only':
        traj_files = [p for p in traj_files if not _is_corrected(p.stem)]
    elif version_filter == 'Corrected only':
        traj_files = [p for p in traj_files if _is_corrected(p.stem)]
    if not traj_files:
        st.info(f'No trajectory CSVs match version filter: {version_filter}.')
    else:
        sel = st.selectbox(
            'Pick a video to plot the nose path',
            traj_files,
            format_func=lambda p: str(p.relative_to(cohort_results)),
            key='traj_pick',
        )
        traj = pd.read_csv(sel)
        valid = traj.dropna(subset=['nose_x', 'nose_y'])
        if valid.empty:
            st.info('That video has no usable nose positions to plot.')
        else:
            fig, ax = pink_fig(figsize=(8, 6))
            ax.plot(valid['nose_x'], valid['nose_y'], color=PINK_LIGHT, linewidth=0.5, alpha=0.6)
            sc = ax.scatter(valid['nose_x'], valid['nose_y'], c=valid['frame'], s=3, cmap=PINK_CMAP)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label('frame number', color=PINK_TEXT)
            cb.outline.set_edgecolor(PINK_DARK)
            cb.ax.yaxis.set_tick_params(color=PINK_TEXT)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=PINK_TEXT)
            ax.set_xlabel('x (px)')
            ax.set_ylabel('y (px)')
            ax.set_title(f'{sel.stem} — nose trajectory')
            ax.set_aspect('equal')
            ax.invert_yaxis()  # video coords have y growing downwards
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)


# Protocol / How to use tab
# Renders PROTOCOL.md from the repo root directly, so the doc stays a
# single source of truth. Do NOT duplicate the protocol text into app.py.

with tab_protocol:
    protocol_path = REPO / 'PROTOCOL.md'
    if protocol_path.exists():
        st.markdown(protocol_path.read_text())
    else:
        st.error(
            f'Could not find `PROTOCOL.md` at {protocol_path}. '
            "The RA-facing protocol should live at the repo root — if it's been "
            "moved or renamed, update this tab's path or restore the file."
        )
