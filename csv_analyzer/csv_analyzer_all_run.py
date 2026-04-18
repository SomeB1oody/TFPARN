from __future__ import annotations

from csv_common import (
    compute_cumulative_minutes,
    compute_stats,
    format_stats,
    load_rows,
    print_stats,
    save_all_plots,
    save_stats_csv,
    validate_csv_and_output,
)

CSV_INPUT_PATH = "/Users/stanyin/Documents/CsvAnalyzer/AASIST_model/best_model_min_dcf_0.2896/timing_records.csv"
OUTPUT_DIR_PATH = "./AASIST_model/best_model_min_dcf_0.2896/stats_all_run"
DEV_UTTERANCE_COUNT = 140950


def main() -> dict[str, object]:
    csv_path, output_dir = validate_csv_and_output(CSV_INPUT_PATH, OUTPUT_DIR_PATH)
    rows = load_rows(csv_path)
    stats = compute_stats(rows, DEV_UTTERANCE_COUNT)
    formatted_stats = format_stats(stats)
    stats_path = save_stats_csv(formatted_stats, output_dir)

    cumulative = compute_cumulative_minutes(rows)
    series = [("", rows, cumulative, None)]
    plot_paths = save_all_plots(series, output_dir)

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
