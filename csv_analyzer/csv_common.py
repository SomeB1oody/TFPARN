from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

STATS_HEADERS = [
    "Memory Usage (GB)",
    "Time per Training Epoch (s)",
    "Time to Best Model (min)",
    "Latency per Utterance (ms)",
]

PLOT_SPECS = [
    ("eer", "EER", "EER"),
    ("min_dcf", "Min DCF", "Min_DCF"),
    ("cllr", "Cllr", "Cllr"),
    ("act_dcf", "actDCF", "actDCF"),
]

SPLIT_NAMES = {
    1: "Train",
    2: "Dev",
    3: "Eval",
}


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


def parse_float(value: str, field_name: str, row_number: int) -> float:
    if value.strip() == "":
        return 0.0
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be a float, got {value!r}.") from exc
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def load_rows(csv_path: Path) -> list[TimingRow]:
    rows: list[TimingRow] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
        if headers is None:
            raise ValueError("The input CSV is empty.")
        if headers != EXPECTED_HEADERS:
            raise ValueError(
                "CSV header mismatch. Expected exactly: "
                + ", ".join(EXPECTED_HEADERS)
            )

        for row_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != len(EXPECTED_HEADERS):
                raise ValueError(
                    f"Row {row_number}: expected {len(EXPECTED_HEADERS)} columns, got {len(raw_row)}."
                )

            run_num = parse_int(raw_row[0], "run_num", row_number)
            split_type = parse_int(raw_row[1], "type", row_number)
            time_seconds = parse_float(raw_row[2], "time", row_number)
            max_memory_gb = parse_float(raw_row[3], "max_memory", row_number)
            if_best = parse_int(raw_row[4], "if_best", row_number)
            eer = parse_float(raw_row[5], "eer", row_number)
            min_dcf = parse_float(raw_row[6], "min_DCF", row_number)
            cllr = parse_float(raw_row[7], "cllr", row_number)
            act_dcf = parse_float(raw_row[8], "actDCF", row_number)

            if run_num < 1:
                raise ValueError(f"Row {row_number}: run_num must be >= 1.")
            if split_type not in SPLIT_NAMES:
                raise ValueError(f"Row {row_number}: type must be 1, 2, or 3.")
            if time_seconds < 0:
                raise ValueError(f"Row {row_number}: time must be >= 0.")
            if max_memory_gb < 0:
                raise ValueError(f"Row {row_number}: max_memory must be >= 0.")
            if if_best not in (0, 1):
                raise ValueError(f"Row {row_number}: if_best must be 0 or 1.")

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
        raise ValueError("The input CSV has headers but no data rows.")

    return rows


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_csv_and_output(csv_input_path: str, output_dir_path: str) -> tuple[Path, Path]:
    if not csv_input_path.strip():
        raise ValueError("Please fill in CSV_INPUT_PATH before running this script.")
    if not output_dir_path.strip():
        raise ValueError("Please fill in OUTPUT_DIR_PATH before running this script.")

    csv_path = Path(csv_input_path).expanduser()
    output_dir = Path(output_dir_path).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, output_dir


def validate_output_dir(output_dir_path: str) -> Path:
    if not output_dir_path.strip():
        raise ValueError("Please fill in OUTPUT_DIR_PATH before running this script.")
    output_dir = Path(output_dir_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Multi-model loading (for compare scripts)
# ---------------------------------------------------------------------------

def load_all_models(model_inputs: list[tuple[str, str]]) -> list[tuple[str, list[TimingRow]]]:
    models: list[tuple[str, list[TimingRow]]] = []
    for model_name, csv_input_path in model_inputs:
        if not csv_input_path.strip():
            continue
        csv_path = Path(csv_input_path).expanduser()
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found for {model_name}: {csv_path}")
        rows = load_rows(csv_path)
        models.append((model_name, rows))

    if not models:
        raise ValueError("All CSV input paths are empty. Please provide at least one.")

    return models


# ---------------------------------------------------------------------------
# Stats computation (for analyzer scripts)
# ---------------------------------------------------------------------------

def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute an average from an empty list.")
    return sum(values) / len(values)


def compute_stats(rows: list[TimingRow], dev_utterance_count: int) -> dict[str, float]:
    train_rows = [row for row in rows if row.split_type == 1]
    dev_rows = [row for row in rows if row.split_type == 2]

    if not train_rows:
        raise ValueError("The input CSV does not contain any Train rows (type=1).")
    if not dev_rows:
        raise ValueError("The input CSV does not contain any Dev rows (type=2).")

    last_best_index = None
    for index, row in enumerate(rows):
        if row.split_type == 2 and row.if_best == 1:
            last_best_index = index

    if last_best_index is None:
        raise ValueError("The input CSV does not contain any valid Dev best-model row (type=2 and if_best=1).")

    memory_usage_gb = average([row.max_memory_gb for row in dev_rows])
    time_per_training_epoch_s = average([row.time_seconds for row in train_rows])
    time_to_best_model_min = sum(row.time_seconds for row in rows[: last_best_index + 1]) / 60.0
    latency_per_utterance_ms = average([row.time_seconds for row in dev_rows]) * 1000.0 / dev_utterance_count

    return {
        "Memory Usage (GB)": memory_usage_gb,
        "Time per Training Epoch (s)": time_per_training_epoch_s,
        "Time to Best Model (min)": time_to_best_model_min,
        "Latency per Utterance (ms)": latency_per_utterance_ms,
    }


def format_stats(stats: dict[str, float]) -> dict[str, str]:
    return {
        "Memory Usage (GB)": f"{stats['Memory Usage (GB)']:.1f}",
        "Time per Training Epoch (s)": f"{stats['Time per Training Epoch (s)']:.1f}",
        "Time to Best Model (min)": f"{stats['Time to Best Model (min)']:.2f}",
        "Latency per Utterance (ms)": f"{stats['Latency per Utterance (ms)']:.4f}",
    }


def save_stats_csv(formatted_stats: dict[str, str], output_dir: Path) -> Path:
    stats_path = output_dir / "stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_HEADERS)
        writer.writeheader()
        writer.writerow(formatted_stats)
    return stats_path


def print_stats(formatted_stats: dict[str, str]) -> None:
    print("CSV analysis summary")
    for key in STATS_HEADERS:
        print(f"{key}: {formatted_stats[key]}")


# ---------------------------------------------------------------------------
# Helpers for "last_best" variants
# ---------------------------------------------------------------------------

def get_last_best_dev_run_num(rows: list[TimingRow]) -> int:
    best_run_nums = [row.run_num for row in rows if row.split_type == 2 and row.if_best == 1]
    if not best_run_nums:
        raise ValueError("The input CSV does not contain any valid Dev best-model row (type=2 and if_best=1).")
    return max(best_run_nums)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_cumulative_minutes(rows: list[TimingRow]) -> list[float]:
    cumulative: list[float] = []
    total = 0.0
    for row in rows:
        total += row.time_seconds
        cumulative.append(total / 60.0)
    return cumulative


def plot_metric_series(
    series: list[tuple[str, list[TimingRow], list[float], int | None]],
    split_type: int,
    metric_name: str,
    y_label: str,
    file_stem: str,
    output_dir: Path,
) -> Path:
    """Plot one metric for one or more model series.

    Each element in *series* is (label, rows, cumulative_minutes, max_run_num).
    If max_run_num is None, all rows are included.
    """
    figure, axis = plt.subplots(figsize=(14.0, 6.0))
    show_legend = len(series) > 1

    for label, rows, cumulative_minutes, max_run_num in series:
        filtered = [
            (cumulative_minutes[i], row)
            for i, row in enumerate(rows)
            if row.split_type == split_type
            and (max_run_num is None or row.run_num <= max_run_num)
        ]
        filtered.sort(key=lambda pair: pair[0])

        if not filtered:
            continue

        x_values = [pair[0] for pair in filtered]
        y_values = [getattr(pair[1], metric_name) for pair in filtered]
        axis.plot(
            x_values, y_values,
            marker="o", linewidth=2.0, markersize=5.0,
            label=label if show_legend else None,
        )

    axis.set_xlabel("Time (min)")
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if show_legend:
        axis.legend()

    output_path = output_dir / f"{file_stem}_per_{SPLIT_NAMES[split_type]}.pdf"
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_all_plots(
    series: list[tuple[str, list[TimingRow], list[float], int | None]],
    output_dir: Path,
) -> list[Path]:
    plot_paths: list[Path] = []
    for split_type in (1, 2):
        for metric_name, y_label, file_stem in PLOT_SPECS:
            plot_paths.append(
                plot_metric_series(series, split_type, metric_name, y_label, file_stem, output_dir)
            )
    return plot_paths
