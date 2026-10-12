# SLEAP analysis pipeline (Windows)

This protocol cover the whole pipeline for sleap analysis. If you are already within the UI, you can jump to section 3, which will tell you how the analysis works.

 Setup is already done on this machine. Follow the sections below to add a cohort and run the pipeline. Anything in a grey box is a command — copy/paste it into PowerShell.

> **Note.** This pipeline consumes `.predictions.slp` files that inference produces automatically: when you click **Run full pipeline** with **Run inference on videos missing predictions** ticked (the default), `scripts/run_pipeline.py` first calls `scripts/run_inference.py` for any video that does not yet have predictions. That script runs the bundled model directly in Python; it does not use `sleap-track`. You do **not** need SLEAP's GUI (`sleap-label`) to use this pipeline — the GUI is only required if you want to label training data or train a new model.

---

## 1. Add a cohort to analyse

The UI auto-discovers cohorts in the repo's `data\` folder.

### 1.1 Create the cohort folder

Inside `~\Desktop\sleap-analysis\data\`, create a folder named for your cohort — e.g. `cohort_12`. This is the top-level folder the UI will list in its sidebar.

### 1.2 Add one subfolder per behavioural experiment

Inside the cohort folder, add a subfolder for **each behavioural experiment**, and drop the raw `.mp4` videos for that experiment inside it. The expected layout is:

```
data\cohort_12\
├── BARNES_DAY2_COHORT_12\    .mp4 videos only
├── BARNES_DAY3_COHORT_12\
├── BARNES_DAY6\
└── NOR_cohort_12\            both *_familiar.mp4 and *_novel.mp4
```

That's it — **just the videos**, nothing else. The pipeline runs SLEAP inference on any video it doesn't already have a prediction for, then does the downstream analysis. A `predictions\` folder will appear next to your video folders automatically on the first run.

**Only `.mp4` files are processed.** The pipeline intentionally skips `.mov` files (they read unreliably through the video decoder — silent seek failures / wrong frame counts / garbage frames, which show up downstream as 0 detections that look like a model failure). If your camera writes `.mov`, convert to `.mp4` first — the pipeline will print a NOTICE listing any orphan `.mov` files it finds:

```powershell
python scripts\convert_mov_to_mp4.py --videos-root data\cohort_12
```

This writes an `.mp4` next to each `.mov`; the pipeline picks up the `.mp4` on the next run. The `.mov` originals are kept in place unless you pass `--delete-source`.

**Naming rules:**
- The cohort folder (e.g. `cohort_12`) can be named anything — the results folder will use its exact name.
- Each experiment subfolder name must **contain** `barnes` or `nor` (case-insensitive), so the pipeline knows which metrics to compute for it.

If your data is already somewhere else on disk and you don't want to copy it, you can use a junction (Windows' equivalent of a symlink) instead:

```powershell
cd ~\Desktop\sleap-analysis
New-Item -ItemType Junction -Path "data\cohort_12" -Target "C:\path\to\your\cohort_folder"
```

`data\` is in `.gitignore`, so videos and predictions never get pushed to GitHub.

---

## 2. Launch the UI

### 2.1 In PowerShell, cd into the repo

```powershell
cd ~\Desktop\sleap-analysis
```

### 2.2 Start Streamlit

Run Streamlit using the SLEAP environment's Python (not the plain `streamlit` command — that won't be on PATH):

```powershell
& $env:APPDATA\uv\tools\sleap\Scripts\python.exe -m streamlit run ui\app.py
```

The app opens in your default browser at **http://localhost:8501**.

### 2.3 Stop the UI

Press `Ctrl+C` in the PowerShell window where Streamlit is running. You can also just close the browser tab if you'd rather leave the server alive in the background.

---

## 3. Run the pipeline (in the UI)

The page has four tabs. Pick the cohort from the sidebar dropdown, then work through the tabs in order.

### Tab 1 Barnes geometry  (once per cohort)

1. Pick a calibration video from the dropdown.
2. Click **Pick centre**, then click the table centre on the image.
3. Click **Pick target hole**, then click the target hole.
4. Click **Save geometry**.

### Tab 2 NOR scenes

**Step 1 — cohort calibration (once per cohort)**

1. Pick a *familiar* video.
2. Click **Pick left object** → click it on the image. Repeat for **Pick right object**.
3. Click **Save cohort calibration**.

**Step 2 — test-phase per video (once per *_novel.mp4)**

For each *_novel video:
1. Select it from the dropdown.
2. Choose **left** or **right** for the side that had the novel object.
3. Click **Save side**.

### Tab 3 Run pipeline (inference)

Leave **Run inference on videos missing predictions** checked (default). Click **Run full pipeline**. Output streams below the button.

First run for a cohort is slow — inference runs on every video, typically 1–3 min per video on CPU. Subsequent runs skip already-processed videos and finish in seconds.

### Tab 4 Results

Three tables (filter summary, validation summary, behavioural metrics), each with a **Download as CSV** button. Below the tables: pink bar charts for the key metrics, and a per-video nose-trajectory plot at the bottom.

---

## 4. Where the outputs live

Inside the repo, under `results\<cohort>\` — each cohort you analyse gets its own folder so runs don't collide:

- `results\<cohort>\<category>\<video>.csv` — clean nose+tail trajectory per video
- `results\<cohort>\<category>\<video>.barnes_metrics.json` or `.nor_metrics.json` — the behavioural numbers
- `results\<cohort>\summary.csv`, `results\<cohort>\validation_summary.csv` — per-cohort summaries
- A consolidated `behavioural_metrics.csv` available via the **Download all metrics as CSV** button in Tab 4 — this is the file to take to R / GraphPad / pandas for stats.

The `<cohort>` folder name matches the folder you created under `data\` in section 1.1 (e.g. `data\cohort_12\` → `results\cohort_12\`). The Results tab in the UI is always scoped to whichever cohort is picked in the sidebar.

If you'd rather skip the UI, you can run the pipeline directly from PowerShell (inference runs automatically for any video missing a prediction):

```powershell
cd ~\Desktop\sleap-analysis
& $env:APPDATA\uv\tools\sleap\Scripts\python.exe scripts\run_pipeline.py `
    --videos-root data\cohort_12 `
    --cohort cohort_12 `
    --barnes-geometry configs\cohort_12_barnes_geometry.json `
    --nor-geometry    configs\cohort_12_nor_geometry.json `
    --nor-scene-dir   data\cohort_12\NOR_cohort_12
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| "No cohorts found under data/" in the sidebar | Section 1.1–1.2 — create a cohort folder with experiment subfolders inside `data\`. |
| `NOTICE: N .mov file(s) with no .mp4 sibling` in the run output | The pipeline only processes `.mp4`. Convert the listed `.mov` files first: `python scripts\convert_mov_to_mp4.py --videos-root data\<cohort>`, then re-run. |
| `streamlit` / `'streamlit' is not recognized` | Don't run `streamlit` directly — use `& $env:APPDATA\uv\tools\sleap\Scripts\python.exe -m streamlit run ui\app.py` (section 2.2). This ensures Streamlit runs inside the same environment SLEAP is installed in. |
| `TypeError: Descriptors cannot be created directly` on Run pipeline | Ask a labmate to check the protobuf pin on this machine — restart Streamlit after. |
| Port 8501 already in use | Close the other PowerShell window running Streamlit, or find and stop the process:<br>`Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess \| Stop-Process` |
| Browser doesn't open automatically | Open it manually and go to http://localhost:8501. |
| `sleap` command not found | Ask a labmate to check the SLEAP install on this machine — it should already be set up. |

---

## 6. Updating the pipeline

When new fixes or features land in the repo, pull them with:

```powershell
cd ~\Desktop\sleap-analysis
git pull
```

Then restart Streamlit (`Ctrl+C` in its window, then re-run the command in section 2.2) to load the new code.
