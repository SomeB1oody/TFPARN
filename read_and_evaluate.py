"""
Model Evaluation Script for ASVspoof5 Competition
Handles: loading trained model, evaluating on dev and eval sets, computing all metrics with calibration
"""

from dataclasses import dataclass
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
class EvaluationConfig:
    """
    Evaluation configuration parameters
    """
    # Model path
    # IMPORTANT: Modify this to your model file path
    model_path: str = "H:/true_tone3/ce/best_model_eer_0.3947.pt"

    # Data paths (data and protocol order must correspond)
    data_dirs: List[str] = None
    protocol_dirs: List[str] = None

    # Audio processing parameters
    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True

    # Evaluation parameters
    batch_size: int = 96  # Batch size
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

    def __post_init__(self):
        """Initialize default data and protocol directories"""
        if self.data_dirs is None:
            self.data_dirs = ["H:/true_tone5/data/flac_D/", "H:/true_tone5/data/flac_E/"]
        if self.protocol_dirs is None:
            self.protocol_dirs = [
                "H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
                "H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv"
            ]


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
# Evaluation Functions
# ============================================================================

def evaluate_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset_name: str = "Dataset"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset and return logits and labels

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        dataset_name: Name of the dataset for display

    Returns:
        (logits, labels) as numpy arrays
    """
    model.eval()
    all_logits = []
    all_labels = []

    print(f"\n[Evaluating] {dataset_name}")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}", dynamic_ncols=True)
        for batch in pbar:
            waveforms = batch['waveforms'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
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
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    eval_logits: np.ndarray
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Apply Platt calibration on validation set and transform eval set
    Platt calibration: trains a logistic regression on validation scores

    Args:
        val_logits: Validation logits [N_val, 2]
        val_labels: Validation labels [N_val]
        eval_logits: Evaluation logits [N_eval, 2]

    Returns:
        (calibrated_eval_scores, calibrator): Calibrated scores and the calibrator model
    """
    print("\n[Calibration] Applying Platt calibration")

    # Extract bonafide scores (logit for class 1)
    val_scores = val_logits[:, 1] - val_logits[:, 0]  # Log odds
    eval_scores = eval_logits[:, 1] - eval_logits[:, 0]

    # Fit logistic regression on validation set
    # This learns: calibrated_score = a * score + b
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(val_scores.reshape(-1, 1), val_labels)

    print(f"  - Calibration parameters: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")

    # Get calibrated probabilities for eval set
    calibrated_probs = calibrator.predict_proba(eval_scores.reshape(-1, 1))[:, 1]

    print(f"[SUCCESS] Platt calibration complete")

    return calibrated_probs, calibrator


def apply_prior_correction(
    val_labels: np.ndarray,
    eval_labels: np.ndarray,
    calibrated_scores: np.ndarray
) -> np.ndarray:
    """
    Apply prior correction based on label distribution difference
    Uses log-odds shift: corrected_logit = logit + log(P_eval/P_val * (1-P_val)/(1-P_eval))

    Args:
        val_labels: Validation labels [N_val]
        eval_labels: Evaluation labels [N_eval]
        calibrated_scores: Calibrated probability scores [N_eval]

    Returns:
        corrected_scores: Prior-corrected probability scores [N_eval]
    """
    print("\n[Prior Correction] Applying prior correction")

    # Compute class priors (proportion of bonafide samples)
    prior_val = np.mean(val_labels == 1)
    prior_eval = np.mean(eval_labels == 1)

    print(f"  - Validation prior P(bonafide): {prior_val:.4f}")
    print(f"  - Evaluation prior P(bonafide): {prior_eval:.4f}")

    # Compute log-odds shift
    shift = compute_prior_log_odds_shift(prior_val, prior_eval)
    print(f"  - Log-odds shift: {shift:.4f}")

    # Convert probabilities to log-odds, apply shift, convert back
    # logit = log(p / (1-p))
    eps = 1e-10  # Small value to avoid log(0)
    calibrated_scores = np.clip(calibrated_scores, eps, 1 - eps)

    logits = np.log(calibrated_scores / (1 - calibrated_scores))
    corrected_logits = logits + shift
    corrected_scores = 1 / (1 + np.exp(-corrected_logits))

    print(f"[SUCCESS] Prior correction complete")

    return corrected_scores


def compute_llr_and_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    dataset_name: str = "Dataset"
) -> Dict[str, float]:
    """
    Compute LLR-based metrics (CLLR) and all other metrics

    Args:
        scores: Probability scores (higher = more likely bonafide) [N]
        labels: Ground truth labels (0=spoof, 1=bonafide) [N]
        dataset_name: Name of dataset for display

    Returns:
        Dictionary containing all metrics including CLLR
    """
    print(f"\n[Computing Metrics] {dataset_name}")

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

    print(f"[SUCCESS] Metrics computed for {dataset_name}")

    return metrics


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    """
    Main evaluation function
    Loads model, evaluates on dev and eval sets with calibration and prior correction
    """
    print("\n" + "="*80)
    print("ASVSPOOF5 MODEL EVALUATION")
    print("="*80)

    # Initialize configuration
    config = EvaluationConfig()

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

    # Create data processing args for validation set
    val_data_args = DataProcessArgs()
    val_data_args.train_data_dir = config.data_dirs[0]  # Use dev as validation
    val_data_args.dev_data_dir = config.data_dirs[0]
    val_data_args.eval_data_dir = config.data_dirs[0]
    val_data_args.train_protocol_dir = config.protocol_dirs[0]
    val_data_args.dev_protocol_dir = config.protocol_dirs[0]
    val_data_args.eval_protocol_dir = config.protocol_dirs[0]
    val_data_args.sample_rate = config.sample_rate
    val_data_args.duration_sec = config.duration_sec
    val_data_args.mono = config.mono
    val_data_args.normalize = config.normalize
    val_data_args.batch_size = config.batch_size
    val_data_args.num_workers = config.num_workers
    val_data_args.prefetch_factor = config.prefetch_factor
    val_data_args.pin_memory = config.pin_memory
    val_data_args.persistent_workers = config.persistent_workers
    val_data_args.train_shuffle = False
    val_data_args.seed = config.seed
    val_data_args.use_rawboost = False  # No augmentation for evaluation

    # Load validation data (dev set)
    print("\n[Loading] Validation set (Dev)")
    _, val_loader, _, _ = make_loaders(val_data_args)

    # Create data processing args for evaluation set
    eval_data_args = DataProcessArgs()
    eval_data_args.train_data_dir = config.data_dirs[1]  # Use eval set
    eval_data_args.dev_data_dir = config.data_dirs[1]
    eval_data_args.eval_data_dir = config.data_dirs[1]
    eval_data_args.train_protocol_dir = config.protocol_dirs[1]
    eval_data_args.dev_protocol_dir = config.protocol_dirs[1]
    eval_data_args.eval_protocol_dir = config.protocol_dirs[1]
    eval_data_args.sample_rate = config.sample_rate
    eval_data_args.duration_sec = config.duration_sec
    eval_data_args.mono = config.mono
    eval_data_args.normalize = config.normalize
    eval_data_args.batch_size = config.batch_size
    eval_data_args.num_workers = config.num_workers
    eval_data_args.prefetch_factor = config.prefetch_factor
    eval_data_args.pin_memory = config.pin_memory
    eval_data_args.persistent_workers = config.persistent_workers
    eval_data_args.train_shuffle = False
    eval_data_args.seed = config.seed
    eval_data_args.use_rawboost = False

    # Load evaluation data
    print("\n[Loading] Evaluation set (Eval)")
    _, eval_loader, _, _ = make_loaders(eval_data_args)

    print("\n[SUCCESS] Data loading complete")

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
    # Step 3: Evaluate on Validation Set (Dev)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: EVALUATE ON VALIDATION SET")
    print("="*80)

    val_logits, val_labels = evaluate_dataset(model, val_loader, device, "Validation (Dev)")

    # Compute initial metrics on validation set
    print("\n[Computing] Initial validation metrics (before calibration)")
    val_metrics_initial = compute_all_metrics(torch.from_numpy(val_logits), torch.from_numpy(val_labels))
    print_metrics(val_metrics_initial, prefix="  [VAL INITIAL] ")

    # ========================================================================
    # Step 4: Evaluate on Evaluation Set (Eval)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: EVALUATE ON EVALUATION SET")
    print("="*80)

    eval_logits, eval_labels = evaluate_dataset(model, eval_loader, device, "Evaluation (Eval)")

    # Compute initial metrics on evaluation set
    print("\n[Computing] Initial evaluation metrics (before calibration)")
    eval_metrics_initial = compute_all_metrics(torch.from_numpy(eval_logits), torch.from_numpy(eval_labels))
    print_metrics(eval_metrics_initial, prefix="  [EVAL INITIAL] ")

    # ========================================================================
    # Step 5: Platt Calibration
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: PLATT CALIBRATION")
    print("="*80)

    # Apply Platt calibration: train on val, apply to eval
    calibrated_eval_scores, calibrator = apply_platt_calibration(
        val_logits, val_labels, eval_logits
    )

    # ========================================================================
    # Step 6: Prior Correction
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: PRIOR CORRECTION")
    print("="*80)

    # Apply prior correction based on label distribution
    corrected_eval_scores = apply_prior_correction(
        val_labels, eval_labels, calibrated_eval_scores
    )

    # ========================================================================
    # Step 7: Compute Final Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: COMPUTE FINAL METRICS WITH CALIBRATION")
    print("="*80)

    # Compute final metrics on calibrated and corrected scores
    final_metrics = compute_llr_and_metrics(corrected_eval_scores, eval_labels, "Evaluation (Final)")

    # ========================================================================
    # Step 8: Print Final Results
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)

    print("\n" + "-"*80)
    print("VALIDATION SET (DEV) - INITIAL")
    print("-"*80)
    print_metrics(val_metrics_initial, prefix="  ")

    print("\n" + "-"*80)
    print("EVALUATION SET (EVAL) - INITIAL (Before Calibration)")
    print("-"*80)
    print_metrics(eval_metrics_initial, prefix="  ")

    print("\n" + "-"*80)
    print("EVALUATION SET (EVAL) - FINAL (After Platt Calibration + Prior Correction)")
    print("-"*80)
    print_metrics(final_metrics, prefix="  ")

    # Print detailed classification report
    print_classification_report_wrapper(
        torch.from_numpy(eval_logits),
        torch.from_numpy(eval_labels),
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
