"""Single-model training-efficiency report.

Reads one model's three seed CSVs (seeds 42, 63, 2026), draws the time-vs-metric
mean and std curves, and writes a summary.md (also printed to the console) into
OUTPUT_DIR_PATH.

Leave a seed path as "" if that seed wasn't run. It's recorded as a missing seed
and the remaining seeds are used. A non-empty but invalid path is an error.
"""

from __future__ import annotations

from csv_common import (
    compute_model_summary,
    load_model_runs,
    save_all_plots,
    validate_output_dir,
    write_summary,
)

MODEL_NAME = ""

# One CSV per seed. Use "" for a seed that wasn't run.
SEED_CSV_PATHS = {
    42: "",
    63: "",
    2026: "",
}

OUTPUT_DIR_PATH = ""
DEV_UTTERANCE_COUNT = 140950

# Number of log-spaced sample points per model (used only when averaging 2+ seeds)
SAMPLES_PER_MODEL = 30


def main() -> dict[str, object]:
    output_dir = validate_output_dir(OUTPUT_DIR_PATH)
    model = load_model_runs(MODEL_NAME, SEED_CSV_PATHS)
    if not model.has_data:
        raise ValueError(f"Model '{MODEL_NAME}' has no seed CSVs (all paths empty).")

    models = [model]
    plot_paths = save_all_plots(models, output_dir, SAMPLES_PER_MODEL)

    summaries = [compute_model_summary(model, DEV_UTTERANCE_COUNT)]
    summary_path = write_summary(
        summaries, output_dir,
        title=f"Training-efficiency summary — {MODEL_NAME}",
        samples_per_model=SAMPLES_PER_MODEL,
    )

    print(f"Plots saved to: {output_dir}")
    return {
        "summary_path": summary_path,
        "plot_paths": plot_paths,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
