from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV_INPUT_PATH = ""
OUTPUT_DIR_PATH = ""
DEV_UTTERANCE_COUNT = 140950

EXPECTED_HEADERS = [
    "run_num",
    "type",
    "time",
    "max_memory",
    "if_best",
    "eer",
    "min_DCF",
    "cllr",
    "actDCF",
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


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute an average from an empty list.")
    return sum(values) / len(values)


def validate_paths() -> tuple[Path, Path]:
    if not CSV_INPUT_PATH.strip():
        raise ValueError("Please fill in CSV_INPUT_PATH before running this script.")
    if not OUTPUT_DIR_PATH.strip():
        raise ValueError("Please fill in OUTPUT_DIR_PATH before running this script.")

    csv_path = Path(CSV_INPUT_PATH).expanduser()
    output_dir = Path(OUTPUT_DIR_PATH).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return csv_path, output_dir


def parse_int(value: str, field_name: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be an integer, got {value!r}.") from exc


def parse_float(value: str, field_name: str, row_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_number}: {field_name} must be a float, got {value!r}.") from exc


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


def compute_stats(rows: list[TimingRow]) -> dict[str, float]:
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
    latency_per_utterance_ms = average([row.time_seconds for row in dev_rows]) * 1000.0 / DEV_UTTERANCE_COUNT

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


def plot_metric(rows: list[TimingRow], split_type: int, metric_name: str, y_label: str, file_stem: str, output_dir: Path) -> Path:
    filtered_rows = sorted(
        (row for row in rows if row.split_type == split_type),
        key=lambda row: row.run_num,
    )

    x_values = [row.run_num for row in filtered_rows]
    y_values = [getattr(row, metric_name) for row in filtered_rows]

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(x_values, y_values, marker="o", linewidth=2.0, markersize=5.0)
    axis.set_xlabel("Run Number")
    axis.set_ylabel(y_label)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_path = output_dir / f"{file_stem}_per_{SPLIT_NAMES[split_type]}.pdf"
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_all_plots(rows: list[TimingRow], output_dir: Path) -> list[Path]:
    plot_paths: list[Path] = []
    for split_type in (1, 2):
        for metric_name, y_label, file_stem in PLOT_SPECS:
            plot_paths.append(plot_metric(rows, split_type, metric_name, y_label, file_stem, output_dir))
    return plot_paths


def print_stats(formatted_stats: dict[str, str]) -> None:
    print("CSV analysis summary")
    print(f"Memory Usage (GB): {formatted_stats['Memory Usage (GB)']}")
    print(f"Time per Training Epoch (s): {formatted_stats['Time per Training Epoch (s)']}")
    print(f"Time to Best Model (min): {formatted_stats['Time to Best Model (min)']}")
    print(f"Latency per Utterance (ms): {formatted_stats['Latency per Utterance (ms)']}")


def main() -> dict[str, object]:
    csv_path, output_dir = validate_paths()
    rows = load_rows(csv_path)
    stats = compute_stats(rows)
    formatted_stats = format_stats(stats)
    stats_path = save_stats_csv(formatted_stats, output_dir)
    plot_paths = save_all_plots(rows, output_dir)

    print_stats(formatted_stats)
    print(f"stats.csv saved to: {stats_path}")
    print(f"Plots saved to: {output_dir}")

    return {
        "stats": formatted_stats,
        "stats_path": stats_path,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
