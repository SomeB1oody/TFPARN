"""Cross-model training-efficiency comparison.

Each model gets its own three seed CSVs (seeds 42, 63, 2026). Every model's seeds
are resampled the same way and overlaid as mean and std curves. A summary.md
(also printed) with each model's own stats is written into OUTPUT_DIR_PATH.

Leave a seed path as "" if a seed wasn't run. If all three paths of a model are
empty, that model is skipped (no line) and recorded in the summary. A non-empty
but invalid path is an error.
"""

from __future__ import annotations

from csv_common import (
    compute_model_summary,
    load_all_models,
    save_all_plots,
    validate_output_dir,
    write_summary,
)

# Per model: a {seed: csv_path} mapping. Use "" for seeds that weren't run.
MODEL_INPUTS = [
    (
        "AASIST",
        {
            42: "",
            63: "",
            2026: "",
        },
    ),
    (
        "RawNet2",
        {
            42: "",
            63: "",
            2026: "",
        },
    ),
    (
        "TFPARN(CE + Mean Pooling)",
        {
            42: "",
            63: "",
            2026: "",
        },
    ),
]

OUTPUT_DIR_PATH = ""
DEV_UTTERANCE_COUNT = 140950

# Number of log-spaced sample points per model (used only when averaging 2+ seeds)
# Single-seed models ignore this and plot their raw per-epoch points
# Change it here, not in csv_common.py
SAMPLES_PER_MODEL = 30


def main() -> dict[str, object]:
    output_dir = validate_output_dir(OUTPUT_DIR_PATH)
    models = load_all_models(MODEL_INPUTS)

    plot_paths = save_all_plots(models, output_dir, SAMPLES_PER_MODEL)
    summaries = [compute_model_summary(model, DEV_UTTERANCE_COUNT) for model in models]
    summary_path = write_summary(
        summaries,
        output_dir,
        title="Training-efficiency comparison",
        samples_per_model=SAMPLES_PER_MODEL,
    )

    plotted = [model.name for model in models if model.has_data]
    print(f"Compared models (plotted): {', '.join(plotted)}")
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
