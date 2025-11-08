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
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    compute_all_metrics, print_metrics, print_classification_report_wrapper,
    compute_cllr, compute_prior_log_odds_shift
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


@dataclass
class EvaluationConfig:
    """
    Evaluation configuration parameters

    Calibration logic (automatic based on dataset name):
    - "Dev" dataset: Used as calibration reference, no calibration applied to itself
    - "Eval" dataset: Automatically has apply_calibration=True (uses Dev as reference)
    - Other datasets: No calibration by default (can be manually enabled)
    """
    # Model path
    model_path: str = "./ce/best_model_eer_0.3947.pt"

    # Dataset configurations
    datasets: List[DatasetConfig] = field(default_factory=lambda: [
        DatasetConfig(
            name="Train",
            data_dir="H:/true_tone5/data/flac_T/",
            protocol_dir="H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.train.tsv"
        ),
        DatasetConfig(
            name="Dev",
            data_dir="H:/true_tone5/data/flac_D/",
            protocol_dir="H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv"
        ),

    ])

    # Audio processing parameters
    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True

    # Evaluation parameters
    batch_size: int = 32  # Batch size
    num_workers: int = 8  # Number of data loading workers
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Model architecture parameters (should match training)
    n_mels: int = 128
    n_fft: int = 768
    hop_length: int = 160
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    model_dropout: float = 0.3
    activation: str = "relu"

    # Miscellaneous
    seed: int = 42  # Random seed

    # Whether to apply prior correction (only applied when calibration is enabled)
    apply_prior_correction: bool = True

    def __post_init__(self):
        """
        Automatically configure calibration based on dataset names
        - "Dev": No calibration (used as reference)
        - "Eval": Enable calibration automatically
        - Others: Keep default (False)
        """
        for dataset in self.datasets:
            if dataset.name == "Eval":
                # Automatically enable calibration for Eval
                dataset.apply_calibration = True
            elif dataset.name == "Dev":
                # Dev is calibration reference, never apply calibration to itself
                dataset.apply_calibration = False
            # Other datasets keep their default apply_calibration value


# ============================================================================
# Model Loading
# ============================================================================

def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False
) -> nn.Module:
    """
    Load model weights from checkpoint with error handling
    Handles cases where checkpoint structure doesn't match current model

    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
        strict: Whether to strictly enforce state_dict matching

    Returns:
        Model with loaded weights
    """
    print(f"\n[Loading] Loading model from {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract state dict - handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  - Checkpoint format: standard (with model_state_dict)")
                # Print additional info if available
                if 'epoch' in checkpoint:
                    print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
                if 'metrics' in checkpoint:
                    print(f"  - Checkpoint metrics: {checkpoint['metrics']}")
            else:
                state_dict = checkpoint
                print(f"  - Checkpoint format: state_dict only")
        else:
            state_dict = checkpoint
            print(f"  - Checkpoint format: raw state_dict")

        # Try to load with strict=False to handle mismatches
        # This allows loading even if some keys don't match
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            print(f"  [WARNING] Missing keys in checkpoint: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"    - {key}")
            else:
                print(f"    - Showing first 5: {missing_keys[:5]}")

        if unexpected_keys:
            print(f"  [WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                print(f"    - Showing first 5: {unexpected_keys[:5]}")

        print(f"[SUCCESS] Model weights loaded successfully")

    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {str(e)}")
        print(f"  Attempting to load with strict=False...")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print(f"[SUCCESS] Model weights loaded with strict=False (some keys may be missing)")

        except Exception as e2:
            print(f"[ERROR] Failed to load model even with strict=False: {str(e2)}")
            raise

    return model


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
    data_args.use_tta = True  # Enable TTA for evaluation
    data_args.tta_num_crops = 5  # Number of crops per sample

    # Load data (we use dev_loader as it's typically used for validation/eval)
    _, loader, _, _ = make_loaders(data_args)

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

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        dataset_name: Name of the dataset for display
        use_tta: Whether TTA is enabled (affects batch shape)

    Returns:
        (logits, labels) as numpy arrays
    """
    model.eval()
    all_logits = []
    all_labels = []

    print(f"\n[Evaluating] {dataset_name} (TTA: {'Enabled' if use_tta else 'Disabled'})")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}", dynamic_ncols=True)
        for batch in pbar:
            waveforms = batch['waveforms'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            if use_tta:
                # TTA enabled: waveforms shape [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # Reshape to [B*num_crops, C, T] for batch processing
                waveforms_flat = waveforms.view(B * num_crops, C, T)
                lengths_flat = lengths.unsqueeze(1).expand(B, num_crops).reshape(B * num_crops)

                # Forward pass on all crops
                logits_flat = model(waveforms_flat, lengths_flat)  # [B*num_crops, 2]

                # Reshape back to [B, num_crops, 2]
                logits_crops = logits_flat.view(B, num_crops, 2)

                # Average logits across crops
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # Normal inference: waveforms shape [B, C, T]
                logits = model(waveforms, lengths)

            # Collect results
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"[SUCCESS] Collected {len(all_labels)} samples from {dataset_name}")

    return all_logits, all_labels


def apply_platt_calibration(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Apply Platt calibration on calibration set and transform test set
    Platt calibration: trains a logistic regression on calibration scores

    Args:
        cal_logits: Calibration logits [N_cal, 2]
        cal_labels: Calibration labels [N_cal]
        test_logits: Test logits [N_test, 2]

    Returns:
        (calibrated_test_scores, calibrator): Calibrated scores and the calibrator model
    """
    # Extract bonafide scores (logit for class 1)
    cal_scores = cal_logits[:, 1] - cal_logits[:, 0]  # Log odds
    test_scores = test_logits[:, 1] - test_logits[:, 0]

    # Fit logistic regression on calibration set
    # This learns: calibrated_score = a * score + b
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(cal_scores.reshape(-1, 1), cal_labels)

    print(f"  - Calibration parameters: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")

    # Get calibrated probabilities for test set
    calibrated_probs = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    return calibrated_probs, calibrator


def apply_prior_correction(
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    calibrated_scores: np.ndarray
) -> np.ndarray:
    """
    Apply prior correction based on label distribution difference
    Uses log-odds shift: corrected_logit = logit + log(P_test/P_cal * (1-P_cal)/(1-P_test))

    Args:
        cal_labels: Calibration labels [N_cal]
        test_labels: Test labels [N_test]
        calibrated_scores: Calibrated probability scores [N_test]

    Returns:
        corrected_scores: Prior-corrected probability scores [N_test]
    """
    # Compute class priors (proportion of bonafide samples)
    prior_cal = np.mean(cal_labels == 1)
    prior_test = np.mean(test_labels == 1)

    print(f"  - Calibration set prior P(bonafide): {prior_cal:.4f}")
    print(f"  - Test set prior P(bonafide): {prior_test:.4f}")

    # Compute log-odds shift
    shift = compute_prior_log_odds_shift(prior_cal, prior_test)
    print(f"  - Log-odds shift: {shift:.4f}")

    # Convert probabilities to log-odds, apply shift, convert back
    # logit = log(p / (1-p))
    eps = 1e-10  # Small value to avoid log(0)
    calibrated_scores = np.clip(calibrated_scores, eps, 1 - eps)

    logits = np.log(calibrated_scores / (1 - calibrated_scores))
    corrected_logits = logits + shift
    corrected_scores = 1 / (1 + np.exp(-corrected_logits))

    return corrected_scores


def compute_metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute all metrics from probability scores

    Args:
        scores: Probability scores (higher = more likely bonafide) [N]
        labels: Ground truth labels (0=spoof, 1=bonafide) [N]

    Returns:
        Dictionary containing all metrics including CLLR
    """
    # Convert scores to logits for compute_all_metrics
    # Reconstruct logits from probabilities
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # Create fake logits: [log(1-p), log(p)]
    logits = np.stack([np.log(1 - scores), np.log(scores)], axis=1)

    # Compute standard metrics using existing function
    metrics = compute_all_metrics(torch.from_numpy(logits), torch.from_numpy(labels))

    # Compute CLLR (Log-Likelihood Ratio cost)
    cllr = compute_cllr(scores, labels)
    metrics['cllr'] = cllr

    return metrics


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

        # Evaluate dataset (with TTA enabled)
        logits, labels = evaluate_dataset(model, dataloader, device, dataset_config.name, use_tta=True)

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
