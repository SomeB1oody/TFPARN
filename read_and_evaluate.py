"""
Model Evaluation Script for ASVspoof5 Competition
Handles: loading trained model, evaluating on multiple datasets, computing all metrics with calibration
Supports flexible evaluation on any number of datasets (train/dev/eval)

Calibration logic:
- Dataset named "Dev" is automatically used as calibration reference
- Dataset named "Eval" automatically gets calibration applied (using Dev)
- Other datasets (e.g., "Train") do not get calibration unless explicitly enabled
"""

from dataclasses import dataclass, field
from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    print_metrics, print_classification_report_wrapper,
    load_model_weights, evaluate_model,
    apply_platt_calibration, apply_prior_correction, compute_metrics_from_scores
)


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str  # Name of the dataset (e.g., "Train", "Dev", "Eval")
    data_dir: str  # Path to audio files
    protocol_dir: str  # Path to protocol file
    apply_calibration: bool = False  # Whether to apply Platt calibration to this dataset
    use_tta: bool = False  # Whether to use Test-Time Augmentation for this dataset


@dataclass
class EvaluationConfig:
    """
    Evaluation configuration parameters
    """
    # Model path
    model_path: str = "./checkpoints/best_model.pt"

    # Dataset configurations
    datasets: List[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig(
            name="Train",
            data_dir="N:/Dataset/ASV5/flac_T/",
            protocol_dir="N:/Dataset/ASV5/ASVspoof5.train.tsv",
            use_tta=False,
        ),
        DatasetConfig(
            name="Dev",
            data_dir="N:/Dataset/ASV5/flac_D/",
            protocol_dir="N:/Dataset/ASV5/ASVspoof5.dev.track_1.tsv",
            use_tta=True,
        ),
        DatasetConfig(
            name="Eval",
            data_dir="N:/Dataset/ASV5/flac_E/",
            protocol_dir="N:/Dataset/ASV5/ASVspoof5.eval.track_1.tsv",
            apply_calibration=True,
            use_tta=True,
        )
    ])

    # Audio processing parameters
    sample_rate: int = 16000
    duration_sec: float = 2.0
    mono: bool = True
    normalize: bool = True

    # Evaluation parameters
    batch_size: int = 256  # Batch size
    num_workers: int = 8  # Number of data loading workers
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Model architecture parameters (should match training)
    n_mels: int = 160
    n_fft: int = 1024
    hop_length: int = 160
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    model_dropout: float = 0.3
    activation: str = "relu"
    pooling_method: str = "mean"  # Options: "mean", "attention", "top-k"
    top_k_ratio: float = 0.5  # For top-k pooling: ratio of frames to keep

    # Miscellaneous
    seed: int = 42  # Random seed

    # Whether to apply prior correction (only applied when calibration is enabled)
    apply_prior_correction: bool = True

    tta_num_crops: int = 5


# ============================================================================
# Data Loading
# ============================================================================

def create_dataloader(
    dataset_config: DatasetConfig,
    config: EvaluationConfig
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for a single dataset

    Args:
        dataset_config: Configuration for the dataset
        config: Main evaluation configuration

    Returns:
        DataLoader for the dataset
    """
    print(f"\n[Loading] Dataset: {dataset_config.name}")
    print(f"  - Data dir: {dataset_config.data_dir}")
    print(f"  - Protocol: {dataset_config.protocol_dir}")
    print(f"  - Apply calibration: {dataset_config.apply_calibration}")
    print(f"  - Use TTA: {dataset_config.use_tta}")

    # Create data processing args
    data_args = DataProcessArgs()
    # Set all three dirs to the same path (make_loaders expects all three)
    data_args.train_data_dir = dataset_config.data_dir
    data_args.dev_data_dir = dataset_config.data_dir
    data_args.eval_data_dir = dataset_config.data_dir
    data_args.train_protocol_dir = dataset_config.protocol_dir
    data_args.dev_protocol_dir = dataset_config.protocol_dir
    data_args.eval_protocol_dir = dataset_config.protocol_dir

    data_args.sample_rate = config.sample_rate
    data_args.duration_sec = config.duration_sec
    data_args.mono = config.mono
    data_args.normalize = config.normalize
    data_args.batch_size = config.batch_size
    data_args.num_workers = config.num_workers
    data_args.prefetch_factor = config.prefetch_factor
    data_args.pin_memory = config.pin_memory
    data_args.persistent_workers = config.persistent_workers
    data_args.train_shuffle = False  # No shuffle for evaluation
    data_args.seed = config.seed
    data_args.use_rawboost = False  # No augmentation for evaluation
    data_args.use_tta = dataset_config.use_tta
    data_args.tta_num_crops = config.tta_num_crops

    # Load data (use dev_loader from make_loaders for consistency)
    _, loader, _ = make_loaders(data_args)

    return loader


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str = "Dataset",
    use_tta: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset and return logits and labels
    Wrapper around evaluate_model() from utils.py for backward compatibility

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        dataset_name: Name of the dataset for display
        use_tta: Whether TTA is enabled (affects batch shape)

    Returns:
        (logits, labels) as numpy arrays
    """
    print(f"\n[Evaluating] {dataset_name} (TTA: {'Enabled' if use_tta else 'Disabled'})")

    # Use shared evaluate_model() from utils
    all_logits, all_labels = evaluate_model(
        model, dataloader, device,
        use_tta=use_tta,
        desc=f"Evaluating {dataset_name}"
    )

    # Convert to numpy for consistency with original API
    all_logits = all_logits.numpy()
    all_labels = all_labels.numpy()

    print(f"[SUCCESS] Collected {len(all_labels)} samples from {dataset_name}")

    return all_logits, all_labels


# Calibration functions moved to utils.py:
# - apply_platt_calibration()
# - apply_prior_correction()
# - compute_metrics_from_scores()


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    """
    Main evaluation function
    Loads model, evaluates on multiple datasets with optional calibration and prior correction
    """
    print("\n" + "="*80)
    print("ASVSPOOF5 MODEL EVALUATION")
    print("="*80)

    # Initialize configuration
    config = EvaluationConfig()

    print(f"\nConfiguration:")
    print(f"  - Model: {config.model_path}")
    print(f"  - Number of datasets: {len(config.datasets)}")

    # Find Dev dataset for calibration
    dev_dataset_idx = None
    for i, ds in enumerate(config.datasets):
        cal_status = "YES" if ds.apply_calibration else "NO"
        print(f"    [{i}] {ds.name} (calibration: {cal_status})")
        if ds.name == "Dev":
            dev_dataset_idx = i

    if dev_dataset_idx is None:
        print(f"\n[WARNING] No 'Dev' dataset found! Calibration will not be available.")
        print(f"  Please add a dataset named 'Dev' to enable calibration.")
    else:
        print(f"\n  - Calibration reference: Dev (index {dev_dataset_idx})")

    print(f"  - Apply prior correction: {config.apply_prior_correction}")

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Get device
    device = get_device()
    clear_cuda_cache()

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Create dataloaders for all datasets
    dataloaders = []
    for dataset_config in config.datasets:
        loader = create_dataloader(dataset_config, config)
        dataloaders.append(loader)

    print(f"\n[SUCCESS] Loaded {len(dataloaders)} datasets")

    # ========================================================================
    # Step 2: Create and Load Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING AND LOADING MODEL")
    print("="*80)

    # Create model args
    model_args = SpeechClassifierArgs()
    model_args.n_mels = config.n_mels
    model_args.n_fft = config.n_fft
    model_args.hop_length = config.hop_length
    model_args.sample_rate = config.sample_rate
    model_args.d_model = config.d_model
    model_args.nhead = config.nhead
    model_args.num_layers = config.num_layers
    model_args.dim_feedforward = config.dim_feedforward
    model_args.dropout = config.model_dropout
    model_args.activation = config.activation
    model_args.pooling_method = config.pooling_method
    model_args.top_k_ratio = config.top_k_ratio

    # Create model
    print("\n[Creating] Model architecture")
    model = create_model(model_args)

    # Load weights from checkpoint
    model = load_model_weights(model, config.model_path, device, strict=False)
    model = model.to(device)

    # ========================================================================
    # Step 3: Evaluate on All Datasets
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: EVALUATE ON ALL DATASETS")
    print("="*80)

    # Store results for all datasets
    all_results = {}

    for i, (dataset_config, dataloader) in enumerate(zip(config.datasets, dataloaders)):
        print(f"\n{'='*80}")
        print(f"Evaluating: {dataset_config.name}")
        print(f"{'='*80}")

        # Evaluate dataset (with TTA based on dataset config)
        logits, labels = evaluate_dataset(model, dataloader, device, dataset_config.name, use_tta=dataset_config.use_tta)

        # Compute initial metrics (without calibration)
        print(f"\n[Computing] Initial metrics for {dataset_config.name}")
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        bonafide_probs = probs[:, 1]
        initial_metrics = compute_metrics_from_scores(bonafide_probs, labels)

        # Store results
        all_results[dataset_config.name] = {
            'logits': logits,
            'labels': labels,
            'initial_metrics': initial_metrics,
            'initial_scores': bonafide_probs
        }

    # ========================================================================
    # Step 4: Apply Calibration and Prior Correction (if enabled)
    # ========================================================================
    if dev_dataset_idx is not None:
        # Check if any dataset needs calibration
        datasets_need_calibration = [ds for ds in config.datasets if ds.apply_calibration]

        if datasets_need_calibration:
            print("\n" + "="*80)
            print("STEP 4: APPLY CALIBRATION AND PRIOR CORRECTION")
            print("="*80)

            # Get calibration dataset (Dev)
            cal_dataset_name = config.datasets[dev_dataset_idx].name
            cal_logits = all_results[cal_dataset_name]['logits']
            cal_labels = all_results[cal_dataset_name]['labels']

            print(f"\nUsing '{cal_dataset_name}' as calibration reference")

            # Apply calibration to datasets that need it
            for dataset_config in config.datasets:
                if not dataset_config.apply_calibration:
                    # Skip datasets that don't need calibration
                    print(f"\n[Skipping] {dataset_config.name} (calibration disabled)")
                    continue

                print(f"\n{'-'*80}")
                print(f"Processing: {dataset_config.name}")
                print(f"{'-'*80}")

                test_logits = all_results[dataset_config.name]['logits']
                test_labels = all_results[dataset_config.name]['labels']

                # Apply Platt calibration
                print(f"\n[Calibration] Applying Platt calibration to {dataset_config.name}")
                calibrated_scores, calibrator = apply_platt_calibration(
                    cal_logits, cal_labels, test_logits
                )

                # Apply prior correction (if enabled)
                if config.apply_prior_correction:
                    if dataset_config.name == "Eval":
                        # For Eval dataset: don't use actual labels for prior correction to avoid label leakage
                        print(f"\n[Prior Correction] Using Dev prior for Eval (avoiding label leakage)")
                        dev_prior = np.mean(cal_labels == 1)
                        print(f"  - Dev set prior P(bonafide): {dev_prior:.4f}")
                        print(f"  - Assuming same prior for Eval")
                        # No correction needed when assuming same prior
                        final_scores = calibrated_scores
                    else:
                        print(f"\n[Prior Correction] Applying prior correction to {dataset_config.name}")
                        final_scores = apply_prior_correction(
                            cal_labels, test_labels, calibrated_scores
                        )
                else:
                    final_scores = calibrated_scores

                # Compute final metrics
                print(f"\n[Computing] Final metrics for {dataset_config.name}")
                final_metrics = compute_metrics_from_scores(final_scores, test_labels)

                # Store calibrated results
                all_results[dataset_config.name]['calibrated_scores'] = final_scores
                all_results[dataset_config.name]['calibrated_metrics'] = final_metrics

            print(f"\n[SUCCESS] Calibration and prior correction complete")
        else:
            print(f"\n[INFO] No datasets require calibration, skipping Step 4")
    else:
        print(f"\n[WARNING] Cannot perform calibration without 'Dev' dataset")

    # ========================================================================
    # Step 5: Print Final Results
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)

    for dataset_config in config.datasets:
        dataset_name = dataset_config.name
        results = all_results[dataset_name]

        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        # Print initial metrics
        print(f"\n{'-'*80}")
        print(f"{dataset_name} - Initial (No Calibration)")
        print(f"{'-'*80}")
        print_metrics(results['initial_metrics'], prefix="  ")

        # Print calibrated metrics if available
        if 'calibrated_metrics' in results:
            print(f"\n{'-'*80}")
            print(f"{dataset_name} - Final (After Calibration + Prior Correction)")
            print(f"{'-'*80}")
            print_metrics(results['calibrated_metrics'], prefix="  ")

            # Print classification report using calibrated scores
            eps = 1e-10
            calibrated_scores_clipped = np.clip(results['calibrated_scores'], eps, 1 - eps)
            calibrated_logits = np.stack([
                np.log(1 - calibrated_scores_clipped),
                np.log(calibrated_scores_clipped)
            ], axis=1)

            print_classification_report_wrapper(
                torch.from_numpy(calibrated_logits),
                torch.from_numpy(results['labels']),
                target_names=['spoof (AI)', 'bonafide (human)']
            )
        else:
            # Print classification report using initial scores
            print_classification_report_wrapper(
                torch.from_numpy(results['logits']),
                torch.from_numpy(results['labels']),
                target_names=['spoof (AI)', 'bonafide (human)']
            )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Evaluation interrupted by user")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[!] Evaluation failed with error:")
        print(f"  {str(e)}")
        raise
