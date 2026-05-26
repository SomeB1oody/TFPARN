# Plotting method (training-efficiency curves)

This file explains how `csv_analyzer.py` and `csv_compare.py` (which share `csv_common.py`) draw the time-vs-metric curves.

## 1. Goal

Compare the training efficiency of different models or ablation settings. The x-axis is cumulative time in minutes on a log scale, not the epoch index. The y-axis is one of four metrics: EER, Min DCF, Cllr, actDCF. Lower is better for all of them.

## 2. Input

- Each model gives three seed CSVs (seeds 42, 63, 2026), set in the script as `{seed: path}`.
- For a multi-model script, the total number of inputs is the number of models times 3.
- Path rules:
  - Empty string `""` means that seed has no data. It is recorded as missing, no error is raised, and only the remaining seeds are used.
  - If all three seeds of a model are empty, the model is skipped, no line is drawn, and it is recorded as skipped in the summary.
  - A non-empty path that does not exist or cannot be parsed raises an error (`FileNotFoundError` or a parse error). It is never silently ignored.

### CSV format

Columns: `run_num, type, time, max_memory, if_best, eer, min_dcf, cllr, act_dcf`
- `type`: 1 is Train, 2 is Dev, 3 is Eval (a one-off final evaluation after training).
- An empty metric cell is treated as NaN (missing), not as 0, so it is not mistaken for a best value.

## 3. Time axis (cumulative time)

- Cumulative time adds up the `time` of Train and Dev rows in file order (each epoch is Train first, then Dev).
- Eval rows (type=3) are left out of the time axis, so the single large eval time does not distort the curve.
- When a metric is missing (NaN), that point is not drawn, but time still keeps adding up.
- The curve runs all the way to the early-stopping point. It is not cut off at the best epoch.

## 4. Sampling and interpolation (core)

How a model's curve is built depends on how many seeds it has. This is done separately for each model, each split (Train/Dev), and each metric.

One seed: the raw per-epoch points are plotted directly, with no resampling. The std band is zero.

Two or more seeds: the seeds are resampled onto a shared grid and then averaged.

1. Take each seed's `(cumulative_time_min, metric_value)` sequence (sorted by time, NaN dropped).
2. Build a log-spaced grid over the time interval that all seeds share, so no seed is ever extrapolated:
   ```
   common_start = max(first time of each seed)
   common_end   = min(last time of each seed)
   time_grid    = geomspace(common_start, common_end, SAMPLES_PER_MODEL)
   ```
   `SAMPLES_PER_MODEL` is the number of grid points, default 30. If the interval touches zero it falls back to a linear `linspace`.
3. Each seed is interpolated onto this grid with linear interpolation (`np.interp`). Since the grid stays inside each seed's `[first, last]` range, it never extrapolates.
4. At each grid point, take the mean and std across seeds. The std is the sample standard deviation (`ddof=1`); with one seed it is 0.

The grid is log-spaced because the x-axis is log time. This puts more points in the early, fast-changing part of training and lets models with very different run lengths share one figure.

To change the number of points, edit `SAMPLES_PER_MODEL` at the top of the script you run (not in `csv_common.py`; the script passes its own value, which overrides the library default). This only affects models with two or more seeds. Single-seed models always use their raw points.

## 5. How each plot is drawn

- Mean line: the mean curve with `o` markers. On dense curves the markers are thinned so the line stays readable.
- Std band: `fill_between(mean-std, mean+std)`, drawn only when there is more than one seed.
- Best star marker: see the next section, drawn only on Dev plots.
- Legend: shows the model name only.
- Colors: one color per model; the line, band, and star all use that color.
- X-axis: log-scaled cumulative time, so short and long runs fit on one figure.

### Best points (stars on Dev plots)

- Taken from the raw CSV, not from the interpolated mean curve.
- Each metric uses its own best point: for each metric plot, each seed takes the point with the lowest value of that metric and marks it with a star. So the stars on the EER, Min DCF, Cllr, and actDCF plots can come from different epochs.
- Marked only on Dev plots, since model selection is done on Dev. Train plots have no stars.

## 6. Output

Each script writes the following under its `OUTPUT_DIR_PATH`:

- 8 PDFs: `{EER,Min_DCF,Cllr,actDCF}_per_{Train,Dev}.pdf`
- `summary.md`, whose content is also printed to the console.

### Statistics in summary.md (one section per model)

- Header: sampling note, seeds, generation date, and the loaded / missing seeds.
- Efficiency (per seed, plus mean and std):
  - Memory Usage (GB): mean of `max_memory` over Dev rows
  - Time per Training Epoch (s): mean of `time` over Train rows
  - Total Train+Val Time (min): sum of all Train and Dev times divided by 60
  - Time to Best Model (min): cumulative time to the best Dev epoch by min_dcf
  - Latency per Utterance (ms): mean Dev-row `time` times 1000, divided by `DEV_UTTERANCE_COUNT`
- Best Dev checkpoint (selected by min_dcf, from the raw CSV): its epoch, time to best, and all metric values.
- Per-metric best Dev value (from the raw CSV, the stars on the plots): each metric's best value per seed, with the time it was reached.
- Missing seeds are listed. A model with all seeds empty is recorded as skipped.

## 7. Scripts at a glance

| Script | Purpose | Models |
|---|---|---|
| `csv_analyzer.py` | Single-model report | 1 (x 3 seeds) |
| `csv_compare.py` | Cross-model comparison | several (each x 3 seeds) |

Both share the same loading, interpolation, plotting, and statistics functions in `csv_common.py`, so every model uses the same logic.

## 8. Known caveats

- Memory units: in the data, AASIST's `max_memory` looks like MB (around 58000), while RawNet2 and TFPARN look like GB (around 1 to 5). The code averages the column as-is and labels it "(GB)", so you need to make the units consistent yourself before comparing memory.
- `DEV_UTTERANCE_COUNT` is hard-coded in each script (default 140950). The latency stat depends on it, so change it if needed.
- `SAMPLES_PER_MODEL` is set per run and applies to all models with two or more seeds in that run.
