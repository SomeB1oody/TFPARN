"""Shared helpers for multi-seed, time-vs-metric training-efficiency plots.

Models are compared by training efficiency: the x-axis is cumulative wall-clock
time (minutes) on a log scale, not the epoch index. The log axis lets models
whose runs differ by about 10x in length (say 200 min vs 1600 min) share one
figure without the fast ones getting squeezed into the left edge.

Sampling is kept separate from any single global time-step, since one step can't
fit models with very different per-epoch costs at once:
  - A model with one seed is plotted from its raw per-epoch points, no resampling.
  - A model with two or more seeds is resampled onto a per-model log-spaced grid
    of SAMPLES_PER_MODEL points so the seeds can be averaged into a mean and std
    band. Every model then gets the same marker density regardless of its speed.

Main design choices:
  - Per-model log-spaced grid (for averaging seeds), linear interpolation, no
    extrapolation. The grid covers the time interval common to all available
    seeds, so no seed is ever extrapolated:
        common_start = max(first cumulative_time_min of each seed)
        common_end   = min(last  cumulative_time_min of each seed)
        time_grid    = geomspace(common_start, common_end, SAMPLES_PER_MODEL)
  - Each seed's best point comes from the raw CSV (best = lowest metric), not from
    the interpolated mean curve.
  - Curves run all the way to the early-stopping point; they are never cut off at
    the best epoch. The one-off final Eval pass (type=3) is left out of the
    per-epoch time axis.
  - Missing seeds (empty path) are recorded in the summary, not silently dropped.
    A non-empty but invalid path is a hard error.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the "science" / "ieee" mpl styles
from matplotlib.ticker import NullFormatter, ScalarFormatter


def _apply_academic_style() -> None:
    """Use the SciencePlots IEEE/Nature look for every figure this module draws.

    The 'science' style sets text.usetex=True, so labels render through the local
    LaTeX install (Computer Modern). If LaTeX isn't available it falls back to the
    'no-latex' variant, which keeps the same layout and colors but uses
    matplotlib's own math text renderer instead of erroring out.
    """
    try:
        plt.style.use(["science"])
        # Force a LaTeX render now so a broken or missing install fails here, where
        # a fallback is still possible, instead of later inside savefig
        figure = plt.figure()
        figure.text(0.5, 0.5, "test")
        figure.canvas.draw()
        plt.close(figure)
    except (RuntimeError, FileNotFoundError):
        plt.style.use(["science", "no-latex"])


_apply_academic_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_HEADERS = [
    "run_num",
    "type",
    "time",
    "max_memory",
    "if_best",
    "eer",
    "min_dcf",
    "cllr",
    "act_dcf",
]

# The three random seeds every model is expected to be trained with
SEEDS: tuple[int, ...] = (42, 63, 2026)

# Number of log-spaced sample points per model, used only when averaging 2+ seeds
# onto a common grid. Single-seed models ignore this and plot their raw epochs.
# The scripts pass their own SAMPLES_PER_MODEL, which overrides this value, so
# change it in the script being run, not here.
SAMPLES_PER_MODEL = 30

# Split type codes used in the "type" column
TRAIN_SPLIT = 1
DEV_SPLIT = 2
EVAL_SPLIT = 3
SPLIT_NAMES = {TRAIN_SPLIT: "Train", DEV_SPLIT: "Dev", EVAL_SPLIT: "Eval"}

# (metric attribute, axis/label text, output file stem). For all metrics lower is
# better, so the best point is the minimum.
PLOT_SPECS = [
    ("eer", "EER", "EER"),
    ("min_dcf", "Min DCF", "Min_DCF"),
    ("cllr", "Cllr", "Cllr"),
    ("act_dcf", "actDCF", "actDCF"),
]
METRIC_LABELS = {key: label for key, label, _ in PLOT_SPECS}

# Model-selection criterion (matches the best_model_min_dcf_* directory naming)
SELECTION_METRIC = "min_dcf"

# Efficiency stats: (key, display label, decimal places)
EFFICIENCY_STATS = [
    ("memory_gb", "Memory Usage (GB)", 1),
    ("train_epoch_s", "Time per Training Epoch (s)", 1),
    ("total_time_min", "Total Train+Val Time (min)", 2),
    ("time_to_best_min", "Time to Best Model (min)", 2),
    ("latency_ms", "Latency per Utterance (ms)", 4),
]

# Decimal places used when printing each metric value
METRIC_PRECISION = {"eer": 4, "min_dcf": 4, "cllr": 4, "act_dcf": 4}


@dataclass(frozen=True)
class TimingRow:
    run_num: int
    split_type: int
    time_seconds: float
    max_memory_gb: float
    if_best: int
    eer: float
    min_dcf: float
    cllr: float
    act_dcf: float


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_int(value: str, field_name: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be an integer, got {value!r}.") from exc


def parse_required_float(value: str, field_name: str, row_number: int) -> float:
    """Parse a required numeric field (time or memory). Blank or non-finite becomes 0.0."""
    if value.strip() == "":
        return 0.0
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be a float, got {value!r}.") from exc
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def parse_metric(value: str, field_name: str, row_number: int) -> float:
    """Parse a metric cell. Blank or non-finite values become NaN so they count as missing.

    Treating a blank metric as 0.0 would wrongly look like the best value,
    so it stays NaN and is dropped when building each metric's series.
    """
    if value.strip() == "":
        return math.nan
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be a float, got {value!r}.") from exc
    if math.isinf(result):
        return math.nan
    return result


def load_rows(csv_path: Path) -> list[TimingRow]:
    rows: list[TimingRow] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"The CSV is empty: {csv_path}")
        if headers != EXPECTED_HEADERS:
            raise ValueError(
                f"CSV header mismatch in {csv_path}. Expected exactly: "
                + ", ".join(EXPECTED_HEADERS)
            )

        for row_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != len(EXPECTED_HEADERS):
                raise ValueError(
                    f"{csv_path} row {row_number}: expected {len(EXPECTED_HEADERS)} columns, got {len(raw_row)}."
                )

            run_num = parse_int(raw_row[0], "run_num", row_number)
            split_type = parse_int(raw_row[1], "type", row_number)
            time_seconds = parse_required_float(raw_row[2], "time", row_number)
            max_memory_gb = parse_required_float(raw_row[3], "max_memory", row_number)
            if_best = parse_int(raw_row[4], "if_best", row_number)
            eer = parse_metric(raw_row[5], "eer", row_number)
            min_dcf = parse_metric(raw_row[6], "min_dcf", row_number)
            cllr = parse_metric(raw_row[7], "cllr", row_number)
            act_dcf = parse_metric(raw_row[8], "act_dcf", row_number)

            if run_num < 1:
                raise ValueError(f"{csv_path} row {row_number}: run_num must be >= 1.")
            if split_type not in SPLIT_NAMES:
                raise ValueError(f"{csv_path} row {row_number}: type must be 1, 2, or 3.")
            if time_seconds < 0:
                raise ValueError(f"{csv_path} row {row_number}: time must be >= 0.")
            if max_memory_gb < 0:
                raise ValueError(f"{csv_path} row {row_number}: max_memory must be >= 0.")
            if if_best not in (0, 1):
                raise ValueError(f"{csv_path} row {row_number}: if_best must be 0 or 1.")

            rows.append(
                TimingRow(
                    run_num=run_num,
                    split_type=split_type,
                    time_seconds=time_seconds,
                    max_memory_gb=max_memory_gb,
                    if_best=if_best,
                    eer=eer,
                    min_dcf=min_dcf,
                    cllr=cllr,
                    act_dcf=act_dcf,
                )
            )

    if not rows:
        raise ValueError(f"The CSV has headers but no data rows: {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# Path validation & multi-seed model loading
# ---------------------------------------------------------------------------

def validate_output_dir(output_dir_path: str) -> Path:
    if not output_dir_path.strip():
        raise ValueError("Please fill in OUTPUT_DIR_PATH before running this script.")
    output_dir = Path(output_dir_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@dataclass
class SeedRun:
    """One seed of one model. `rows` is None when the path was left blank."""
    seed: int
    path: str
    rows: list[TimingRow] | None
    status: str  # "loaded" | "missing"


@dataclass
class ModelRuns:
    """A model together with its (up to three) seed runs"""
    name: str
    seed_runs: list[SeedRun]

    @property
    def loaded(self) -> list[SeedRun]:
        return [s for s in self.seed_runs if s.status == "loaded"]

    @property
    def loaded_seeds(self) -> list[int]:
        return [s.seed for s in self.seed_runs if s.status == "loaded"]

    @property
    def missing_seeds(self) -> list[int]:
        return [s.seed for s in self.seed_runs if s.status == "missing"]

    @property
    def has_data(self) -> bool:
        return len(self.loaded) > 0


def _normalize_seed_paths(seed_paths) -> list[tuple[int, str]]:
    """Accept either {seed: path} or [(seed, path), ...] and return them as ordered pairs"""
    if isinstance(seed_paths, dict):
        items = list(seed_paths.items())
    else:
        items = list(seed_paths)
    return [(int(seed), "" if path is None else str(path)) for seed, path in items]


def load_model_runs(name: str, seed_paths) -> ModelRuns:
    """Load every seed of one model.

    - A blank path is recorded as a missing seed (no input), not an error.
    - A non-empty path that doesn't exist (or fails to parse) is a hard error.
    """
    seed_runs: list[SeedRun] = []
    for seed, path in _normalize_seed_paths(seed_paths):
        if path.strip() == "":
            seed_runs.append(SeedRun(seed=seed, path="", rows=None, status="missing"))
            continue
        csv_path = Path(path).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"CSV file not found for model '{name}', seed {seed}: {csv_path}"
            )
        rows = load_rows(csv_path)
        seed_runs.append(SeedRun(seed=seed, path=path, rows=rows, status="loaded"))
    return ModelRuns(name=name, seed_runs=seed_runs)


def load_all_models(model_inputs) -> list[ModelRuns]:
    """Load several models. `model_inputs` is [(name, seed_paths), ...].

    Models whose seeds are all blank are kept (so they show up in the summary as
    skipped) but draw no line. Raises only if every model is empty.
    """
    models = [load_model_runs(name, seed_paths) for name, seed_paths in model_inputs]
    if not any(model.has_data for model in models):
        raise ValueError("All seed paths are empty for every model; nothing to plot.")
    return models


# ---------------------------------------------------------------------------
# Time / metric series extraction (per seed)
# ---------------------------------------------------------------------------

def split_series(rows: list[TimingRow], split_type: int, metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumulative_minutes, metric_value) for one split and metric.

    Cumulative time adds up over Train and Dev rows in file order (the one-off
    final Eval pass is left out of the time axis). Rows whose metric is NaN are
    dropped, but time still advances through that epoch.
    """
    total_seconds = 0.0
    times: list[float] = []
    values: list[float] = []
    for row in rows:
        if row.split_type not in (TRAIN_SPLIT, DEV_SPLIT):
            continue
        total_seconds += row.time_seconds
        if row.split_type == split_type:
            value = getattr(row, metric_name)
            if not math.isnan(value):
                times.append(total_seconds / 60.0)
                values.append(value)

    if not times:
        return np.array([]), np.array([])

    order = np.argsort(times)
    return np.array(times)[order], np.array(values)[order]


def seed_best_point(rows: list[TimingRow], split_type: int, metric_name: str) -> tuple[float, float] | None:
    """Best (lowest-metric) point for one seed, taken from the raw CSV.

    Returns (cumulative_minutes, metric_value), or None if there's no valid data.
    """
    times, values = split_series(rows, split_type, metric_name)
    if len(times) == 0:
        return None
    best_index = int(np.argmin(values))
    return float(times[best_index]), float(values[best_index])


def seed_selection_checkpoint(rows: list[TimingRow]) -> dict | None:
    """The model-selection checkpoint for one seed: the Dev row with the lowest
    SELECTION_METRIC (min_dcf), taken from the raw CSV. Returns the time and all
    metrics.
    """
    total_seconds = 0.0
    best: dict | None = None
    for row in rows:
        if row.split_type not in (TRAIN_SPLIT, DEV_SPLIT):
            continue
        total_seconds += row.time_seconds
        if row.split_type != DEV_SPLIT:
            continue
        criterion = getattr(row, SELECTION_METRIC)
        if math.isnan(criterion):
            continue
        if best is None or criterion < best[SELECTION_METRIC]:
            best = {
                "run_num": row.run_num,
                "time_min": total_seconds / 60.0,
                "eer": row.eer,
                "min_dcf": row.min_dcf,
                "cllr": row.cllr,
                "act_dcf": row.act_dcf,
            }
    return best


# ---------------------------------------------------------------------------
# Per-model log-spaced interpolation grid (shared across seeds)
# ---------------------------------------------------------------------------

def build_time_grid(seed_times: list[np.ndarray], n_points: int = SAMPLES_PER_MODEL) -> np.ndarray:
    """Per-model, log-spaced grid over the interval covered by all seeds.

        common_start = max(first time of each seed)
        common_end   = min(last  time of each seed)
        grid         = geomspace(common_start, common_end, n_points)

    Every grid point sits inside every seed's [first, last] range, so linear
    interpolation never extrapolates. The grid is log-spaced so the points spread
    evenly on the log-time x-axis (more detail in the early, fast-changing part).
    Falls back to linear spacing if the interval touches zero. Returns an empty
    array if there's no common overlap. `n_points` is the number of sample points.
    """
    if n_points < 2:
        raise ValueError(f"Number of sample points must be >= 2, got {n_points}.")
    if not seed_times:
        return np.array([])
    common_start = max(float(times[0]) for times in seed_times)
    common_end = min(float(times[-1]) for times in seed_times)
    if common_end <= common_start + 1e-9:
        return np.array([])

    if common_start > 0:
        return np.geomspace(common_start, common_end, n_points)
    return np.linspace(common_start, common_end, n_points)


def interpolate_to_grid(
    seed_series_list: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Linearly interpolate every seed onto `grid`, then return (mean, std, n_seeds).

    std is the sample standard deviation (ddof=1) when there's more than one seed,
    and zeros for a single seed.
    """
    stacked = np.vstack([np.interp(grid, times, values) for times, values in seed_series_list])
    mean = stacked.mean(axis=0)
    n_seeds = stacked.shape[0]
    std = stacked.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
    return mean, std, n_seeds


@dataclass
class ModelMetricCurve:
    """Everything needed to draw one model's mean and std curve for one metric"""
    name: str
    grid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_seeds: int
    best_points: list[tuple[int, float, float]] = field(default_factory=list)  # (seed, time, value)
    note: str = ""


def compute_model_curve(
    model: ModelRuns,
    split_type: int,
    metric_name: str,
    samples_per_model: int = SAMPLES_PER_MODEL,
) -> ModelMetricCurve:
    """Build one model's curve for one metric.

    - One seed: the raw per-epoch points (no resampling, std = 0).
    - 2+ seeds: each seed linearly interpolated onto a per-model log-spaced grid
      of `samples_per_model` points, then averaged into a mean and std.
    """
    seed_series_list: list[tuple[np.ndarray, np.ndarray]] = []
    best_points: list[tuple[int, float, float]] = []
    for seed_run in model.loaded:
        times, values = split_series(seed_run.rows, split_type, metric_name)
        if len(times) == 0:
            continue
        seed_series_list.append((times, values))
        best_index = int(np.argmin(values))
        best_points.append((seed_run.seed, float(times[best_index]), float(values[best_index])))

    empty = np.array([])
    if not seed_series_list:
        return ModelMetricCurve(model.name, empty, empty, empty, 0, [], note="no valid data for this metric")

    # Single seed: plot the real curve, no resampling
    if len(seed_series_list) == 1:
        times, values = seed_series_list[0]
        return ModelMetricCurve(model.name, times, values, np.zeros_like(values), 1, best_points)

    grid = build_time_grid([times for times, _ in seed_series_list], n_points=samples_per_model)
    if len(grid) == 0:
        return ModelMetricCurve(model.name, empty, empty, empty, len(seed_series_list), best_points,
                                note="seeds share no common time interval")

    mean, std, n_seeds = interpolate_to_grid(seed_series_list, grid)
    return ModelMetricCurve(model.name, grid, mean, std, n_seeds, best_points)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(
    curves: list[ModelMetricCurve],
    split_type: int,
    y_label: str,
    file_stem: str,
    output_dir: Path,
    mark_best: bool,
) -> Path:
    """Draw one metric for one or more models: mean line, std band, best markers"""
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, curve in enumerate(curves):
        if len(curve.grid) == 0:
            continue
        color = color_cycle[index % len(color_cycle)]
        # Thin the markers on dense raw curves so the line stays readable
        markevery = max(1, len(curve.grid) // 40)
        axis.plot(
            curve.grid, curve.mean,
            marker="o", linewidth=1.1, markersize=2.6, markevery=markevery,
            color=color, label=curve.name,
        )
        if curve.n_seeds > 1:
            axis.fill_between(
                curve.grid, curve.mean - curve.std, curve.mean + curve.std,
                color=color, alpha=0.30, linewidth=0.0,
            )
        if mark_best and curve.best_points:
            best_x = [point[1] for point in curve.best_points]
            best_y = [point[2] for point in curve.best_points]
            axis.scatter(
                best_x, best_y,
                marker="*", s=60.0, color=color, edgecolors="black", linewidths=0.5, zorder=5,
            )

    # Log time axis so runs that differ by about 10x in length can share one
    # figure without the short runs getting squeezed into the left edge
    axis.set_xscale("log")
    time_formatter = ScalarFormatter()
    time_formatter.set_scientific(False)
    axis.xaxis.set_major_formatter(time_formatter)
    axis.xaxis.set_minor_formatter(NullFormatter())

    axis.set_xlabel("Cumulative time (min, log scale)")
    axis.set_ylabel(y_label)
    # Keep only a faint major grid so the plot doesn't look cluttered
    axis.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.3)

    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize="x-small")

    output_path = output_dir / f"{file_stem}_per_{SPLIT_NAMES[split_type]}.pdf"
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_all_plots(
    models: list[ModelRuns],
    output_dir: Path,
    samples_per_model: int = SAMPLES_PER_MODEL,
) -> list[Path]:
    """Make all the Train and Dev metric plots. Best markers show up on Dev plots
    only, since model selection is done on the Dev split. `samples_per_model` sets
    the number of log-spaced grid points used only when averaging 2+ seeds."""
    plotable = [model for model in models if model.has_data]
    plot_paths: list[Path] = []
    for split_type in (TRAIN_SPLIT, DEV_SPLIT):
        mark_best = split_type == DEV_SPLIT
        for metric_name, y_label, file_stem in PLOT_SPECS:
            curves = [
                compute_model_curve(model, split_type, metric_name, samples_per_model)
                for model in plotable
            ]
            plot_paths.append(
                plot_metric(curves, split_type, y_label, file_stem, output_dir, mark_best)
            )
    return plot_paths


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def mean_std(values: list[float]) -> tuple[float, float, int]:
    """Sample mean and std (ddof=1) over the finite values. std is 0 for a single value."""
    finite = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    count = len(finite)
    if count == 0:
        return math.nan, math.nan, 0
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if count > 1 else 0.0
    return mean, std, count


def compute_seed_stats(rows: list[TimingRow], dev_utterance_count: int) -> dict[str, float]:
    """Efficiency stats for one seed run"""
    train_rows = [row for row in rows if row.split_type == TRAIN_SPLIT]
    dev_rows = [row for row in rows if row.split_type == DEV_SPLIT]
    if not train_rows:
        raise ValueError("CSV contains no Train rows (type=1).")
    if not dev_rows:
        raise ValueError("CSV contains no Dev rows (type=2).")

    total_time_min = sum(row.time_seconds for row in (train_rows + dev_rows)) / 60.0

    checkpoint = seed_selection_checkpoint(rows)
    time_to_best_min = checkpoint["time_min"] if checkpoint else math.nan

    latency_ms = (
        _mean([row.time_seconds for row in dev_rows]) * 1000.0 / dev_utterance_count
        if dev_utterance_count > 0 else math.nan
    )

    return {
        "memory_gb": _mean([row.max_memory_gb for row in dev_rows]),
        "train_epoch_s": _mean([row.time_seconds for row in train_rows]),
        "total_time_min": total_time_min,
        "time_to_best_min": time_to_best_min,
        "latency_ms": latency_ms,
    }


@dataclass
class ModelSummary:
    name: str
    loaded_seeds: list[int]
    missing_seeds: list[int]
    per_seed_stats: dict[int, dict[str, float]]            # seed to efficiency stats
    per_seed_checkpoint: dict[int, dict | None]            # seed to min_dcf checkpoint
    per_seed_metric_best: dict[int, dict[str, tuple[float, float] | None]]  # seed to metric to (time, value)


def compute_model_summary(model: ModelRuns, dev_utterance_count: int) -> ModelSummary:
    per_seed_stats: dict[int, dict[str, float]] = {}
    per_seed_checkpoint: dict[int, dict | None] = {}
    per_seed_metric_best: dict[int, dict[str, tuple[float, float] | None]] = {}

    for seed_run in model.loaded:
        per_seed_stats[seed_run.seed] = compute_seed_stats(seed_run.rows, dev_utterance_count)
        per_seed_checkpoint[seed_run.seed] = seed_selection_checkpoint(seed_run.rows)
        per_seed_metric_best[seed_run.seed] = {
            metric: seed_best_point(seed_run.rows, DEV_SPLIT, metric)
            for metric in METRIC_LABELS
        }

    return ModelSummary(
        name=model.name,
        loaded_seeds=model.loaded_seeds,
        missing_seeds=model.missing_seeds,
        per_seed_stats=per_seed_stats,
        per_seed_checkpoint=per_seed_checkpoint,
        per_seed_metric_best=per_seed_metric_best,
    )


# ---------------------------------------------------------------------------
# Summary rendering (console + summary.md)
# ---------------------------------------------------------------------------

def _fmt(value: float | None, precision: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{precision}f}"


def _fmt_mean_std(values: list[float], precision: int) -> str:
    mean, std, count = mean_std(values)
    if count == 0:
        return "—"
    if count == 1:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _render_model_section(summary: ModelSummary) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Model: {summary.name}")
    lines.append("")

    if not summary.loaded_seeds:
        lines.append("_Skipped: all seed paths are empty (no line plotted)._")
        if summary.missing_seeds:
            lines.append("")
            lines.append(f"Missing seeds (no input): {', '.join(map(str, summary.missing_seeds))}")
        lines.append("")
        return lines

    loaded = ", ".join(map(str, summary.loaded_seeds))
    missing = ", ".join(map(str, summary.missing_seeds)) if summary.missing_seeds else "none"
    lines.append(f"- Seeds loaded: {loaded}")
    lines.append(f"- Missing seeds (no input): {missing}")
    lines.append("")

    seeds = summary.loaded_seeds

    # Efficiency stats
    lines.append("### Efficiency (per seed)")
    lines.append("")
    header = "| Stat | " + " | ".join(f"seed {s}" for s in seeds) + " | mean ± std |"
    sep = "|" + "---|" * (len(seeds) + 2)
    lines.append(header)
    lines.append(sep)
    for key, label, precision in EFFICIENCY_STATS:
        per_seed_values = [summary.per_seed_stats[s][key] for s in seeds]
        cells = " | ".join(_fmt(v, precision) for v in per_seed_values)
        agg = _fmt_mean_std(per_seed_values, precision)
        lines.append(f"| {label} | {cells} | {agg} |")
    lines.append("")

    # Selected (min_dcf) checkpoint
    lines.append(f"### Best Dev checkpoint (selected by {METRIC_LABELS[SELECTION_METRIC]}, from raw CSV)")
    lines.append("")
    lines.append("| Field | " + " | ".join(f"seed {s}" for s in seeds) + " | mean ± std |")
    lines.append(sep)

    epoch_cells = " | ".join(
        str(summary.per_seed_checkpoint[s]["run_num"]) if summary.per_seed_checkpoint[s] else "—"
        for s in seeds
    )
    lines.append(f"| Best epoch (run_num) | {epoch_cells} | — |")

    time_values = [
        summary.per_seed_checkpoint[s]["time_min"] if summary.per_seed_checkpoint[s] else math.nan
        for s in seeds
    ]
    lines.append(
        f"| Time to best (min) | " + " | ".join(_fmt(v, 2) for v in time_values)
        + f" | {_fmt_mean_std(time_values, 2)} |"
    )
    for metric, label in METRIC_LABELS.items():
        precision = METRIC_PRECISION[metric]
        metric_values = [
            summary.per_seed_checkpoint[s][metric] if summary.per_seed_checkpoint[s] else math.nan
            for s in seeds
        ]
        cells = " | ".join(_fmt(v, precision) for v in metric_values)
        lines.append(f"| {label} @ best | {cells} | {_fmt_mean_std(metric_values, precision)} |")
    lines.append("")

    # Per-metric best (the points marked on the Dev plots)
    lines.append("### Per-metric best Dev value (from raw CSV, marked on plots)")
    lines.append("")
    lines.append("Cell = best value @ cumulative time (min).")
    lines.append("")
    lines.append("| Metric | " + " | ".join(f"seed {s}" for s in seeds) + " | best value mean ± std |")
    lines.append(sep)
    for metric, label in METRIC_LABELS.items():
        precision = METRIC_PRECISION[metric]
        cells = []
        best_values = []
        for s in seeds:
            point = summary.per_seed_metric_best[s][metric]
            if point is None:
                cells.append("—")
                best_values.append(math.nan)
            else:
                time_min, value = point
                cells.append(f"{value:.{precision}f} @ {time_min:.1f}")
                best_values.append(value)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {_fmt_mean_std(best_values, precision)} |")
    lines.append("")
    return lines


def write_summary(
    summaries: list[ModelSummary],
    output_dir: Path,
    title: str = "Training-efficiency summary",
    samples_per_model: int = SAMPLES_PER_MODEL,
) -> Path:
    """Print the summary to the console and save it to <output_dir>/summary.md"""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "Sampling: single-seed models plotted from raw per-epoch points; models with "
        f"2+ seeds resampled onto a per-model {samples_per_model}-point log-spaced grid "
        "(linear interpolation, no extrapolation), mean ± std across seeds. "
        "Time axis is log-scaled; curves run to early-stopping."
    )
    lines.append(f"Seeds: {', '.join(map(str, SEEDS))}.")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")
    for summary in summaries:
        lines.extend(_render_model_section(summary))

    text = "\n".join(lines).rstrip() + "\n"
    summary_path = output_dir / "summary.md"
    summary_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"summary.md saved to: {summary_path}")
    return summary_path
