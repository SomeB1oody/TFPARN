from __future__ import annotations

from csv_common import (
    compute_cumulative_minutes,
    get_last_best_dev_run_num,
    load_all_models,
    save_all_plots,
    validate_output_dir,
)

AASIST_CSV_INPUT_PATH = "/Users/stanyin/Documents/CsvAnalyzer/AASIST_model/best_model_min_dcf_0.2896/timing_records.csv"
RawNet2_CSV_INPUT_PATH = "/Users/stanyin/Documents/CsvAnalyzer/rawNet2/best_model_min_dcf_0.5008/timing.csv"
TFAPRN_CSV_INPUT_PATH = "/Users/stanyin/Documents/CsvAnalyzer/TFPARN_models/ce_no_pairwise/best_model_min_dcf_0.2722/timing_records.csv"
OUTPUT_DIR_PATH = "./compare_stats_last_best"
DEV_UTTERANCE_COUNT = 140950

MODEL_INPUTS: list[tuple[str, str]] = [
    ("AASIST", AASIST_CSV_INPUT_PATH),
    ("RawNet2", RawNet2_CSV_INPUT_PATH),
    ("TFPARN(CE + Mean Pooling)", TFAPRN_CSV_INPUT_PATH),
]


def main() -> dict[str, object]:
    output_dir = validate_output_dir(OUTPUT_DIR_PATH)
    models = load_all_models(MODEL_INPUTS)

    series = [
        (name, rows, compute_cumulative_minutes(rows), get_last_best_dev_run_num(rows))
        for name, rows in models
    ]
    plot_paths = save_all_plots(series, output_dir)

    print(f"Compared models: {', '.join(name for name, _ in models)}")
    print(f"Plots saved to: {output_dir}")

    return {
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
