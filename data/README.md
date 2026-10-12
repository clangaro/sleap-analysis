# `data/` — drop your cohort folders here

This folder is **gitignored**. The repo never tracks your raw videos or predictions; this directory just gives them a fixed location so the UI can discover them automatically.

## Expected layout

For each cohort you want to analyse, create a subfolder of `data/` containing just the video folders:

```
data/
└── cohort_12/                     ← any name; the UI auto-derives an ID from it
    ├── BARNES_DAY2_COHORT_12/     ← Barnes day 2 trial videos (.mp4)
    ├── BARNES_DAY3_COHORT_12/     ← day 3
    ├── BARNES_DAY6/               ← day 6 (probe)
    └── NOR_cohort_12/             ← NOR videos (both *_familiar.mp4 and *_novel.mp4)
```

You **don't** need to provide a `predictions/` folder. The pipeline creates one under `data/cohort_12/predictions/` on the first run, mirroring your video-folder layout, and skips any video that already has a prediction on later runs.

## Naming rules (only these matter)

- Video folders must **contain** `barnes` or `nor` (case-insensitive) somewhere in their name so the classifier knows which metrics to compute.
- The cohort folder itself can be named anything, but that name is what shows up in `results/<cohort>/…`.

Everything else is free.

## Quickstart for a new cohort

1. Copy the cohort directory into `data/` (e.g. `data/cohort_12`).
2. Launch the UI: `streamlit run ui/app.py`.
3. Pick the cohort from the sidebar dropdown.
4. Annotate geometry in Tabs 1 and 2, then click **Run full pipeline** in Tab 3. Inference + analysis both happen in one shot.
