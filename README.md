# sleap-analysis

Behavioural metrics for mice in the Barnes maze and the novel object recognition (NOR) test, computed from SLEAP pose estimates of the nose and the tail.

SLEAP gives a nose and a tail position for every video frame. This repository turns those predictions into per-video behavioural measures: latency to the target hole, time at the target and target-quadrant occupancy for the Barnes maze, and exploration times and a discrimination index for NOR. It also cleans the trajectories first and flags videos whose tracking is unreliable. A Streamlit interface lets lab members without coding experience run the whole pipeline.

## Data

**Animals and set-up.** Data come from 5 PER2::LUC mice, a circadian reporter line, recorded in the Kriegsfeld Lab at UC Berkeley. Behaviour was filmed from above in a Barnes maze and a novel object recognition arena at 30 frames per second, and body parts were tracked with SLEAP using a model trained on 302 labelled frames.

Videos, predictions and the trained model are not included. The model weights are available on request.

## Method

1. **Inference** (`scripts/run_inference.py`). A single-animal SLEAP model with two keypoints (nose and tail) predicts both positions in every frame. The predictor is called from Python rather than through `sleap-track`, because the model's training config lists the skeleton twice and `sleap-track` then writes no instances. Both SLEAP 1.4 (TensorFlow) and SLEAP 1.5+ (sleap-nn, PyTorch) are supported.
2. **Cleaning** (`scripts/filter_and_interpolate.py`). Frames where the nose and tail are closer than 15 px are treated as a tracking failure and set to missing. Gaps of up to 30 frames (about 1 s) are filled by linear interpolation; longer gaps stay missing.
3. **Validation** (`scripts/validate_trajectories.py`). Frames where a keypoint moves more than 30 px between consecutive frames are flagged as implausible. A video is flagged for high missingness when both the nose and the tail have a run of missing frames longer than 5 s. Flags are reported, not used to exclude videos.
4. **Barnes metrics** (`scripts/barnes_metrics.py`). The maze centre and target hole are clicked once per cohort. The nose is "at the target" when it is within 12% of the centre-to-target distance of the target hole. The target quadrant is the 90° sector centred on the target, measured around the maze centre.
5. **NOR metrics** (`scripts/nor_metrics.py`). The two object positions are clicked once per cohort, and the side of the novel object is recorded for each test video. A frame counts as investigation when the nose is within 80 px of an object and the tail-to-nose vector points within 90° of it. The discrimination index is (T_novel − T_familiar) / (T_novel + T_familiar).

`scripts/run_pipeline.py` runs steps 1-5 in order, deciding whether each video is Barnes or NOR from its folder name. All thresholds are command-line options. The defaults were chosen for this set-up: a mouse is roughly 50-100 px long in these videos, so nose and tail closer than 15 px means the model has lost them; 30 px per frame (about 900 px/s) is well above normal mouse locomotion; and the Barnes target radius is roughly one hole diameter plus a small margin.

`scripts/compare_to_truth.py` measures model accuracy on frames that were hand-labelled blind to the predictions, so that accuracy is measured on videos the model has not seen.

## Results

[CAROLINA: results and figures, if any are to be shown]

The pipeline was run end to end on 16 prediction files from one cohort (6 Barnes, 10 NOR), and produced metrics for all 16.

## Limitations

- All distances and speeds are in pixels. Nothing is converted to centimetres, so pixel thresholds need checking for each camera set-up.
- The Barnes target zone and the NOR object positions come from manual clicks on one frame per cohort.
- Investigation in NOR is defined from two keypoints only, so the head direction is approximated by the tail-to-nose vector.
- Interpolation is linear, so it cannot recover movement during a gap.

## How to run

The scripts run inside a SLEAP environment (Python 3.9 with SLEAP 1.4.1 was used here). `env/sleap_windows_cpu.yml` creates a CPU-only conda environment on Windows.

```bash
conda activate sleap
pip install -r ui/requirements.txt     # only needed for the interface
streamlit run ui/app.py                # graphical interface
```

Place the model weights in `model/pretrained_unified_v2/` before running inference. Put each cohort in `data/<cohort>/`, with one folder per experiment whose name contains `barnes` or `nor`. `data/README.md` describes the layout.

From the command line:

```bash
# once per cohort: click the maze centre and target hole, then the two NOR objects
python scripts/annotate_barnes_geometry.py --video <barnes_video.mp4> --cohort 12
python scripts/annotate_nor_geometry.py --video <nor_familiar_video.mp4> --cohort 12
# once per NOR test video: record which side holds the novel object
python scripts/annotate_nor_scene.py --video <nor_novel_video.mp4> --cohort 12

python scripts/run_pipeline.py \
    --videos-root data/cohort_12 \
    --cohort cohort_12 \
    --barnes-geometry configs/cohort_12_barnes_geometry.json \
    --nor-geometry configs/cohort_12_nor_geometry.json \
    --nor-scene-dir data/cohort_12/NOR_cohort_12
```

Use `--pred-root` instead of `--videos-root` to skip inference when predictions already exist. Every script takes `--help`. Outputs go to `results/<cohort>/`: one cleaned trajectory CSV per video, `summary.csv` (cleaning counts), `validation_summary.csv` (speed and missingness flags), and one `.barnes_metrics.json` or `.nor_metrics.json` per video. Each JSON file records the parameters it was computed with.

`scripts/convert_mov_to_mp4.py` converts `.mov` recordings to H.264 `.mp4`, which SLEAP reads more reliably. On Windows, `scripts/run_inference.bat` runs inference inside the conda environment, and `scripts/reencode_video.bat` re-encodes a video with dense keyframes when SLEAP cannot seek in it. `PROTOCOL.md` is the step-by-step guide for running the interface on the lab's Windows machine.

## Licence

MIT
