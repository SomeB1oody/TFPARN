"""
Utility Functions for ASVspoof5 Training
Includes: seed fixing, device management, evaluation metrics, loss functions
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, classification_report,
    accuracy_score, f1_score, recall_score
)
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Random Seed
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Fix random seed for reproducibility

    Args:
        seed: Random seed
    """
    print(f"\n[Seed] Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure cudnn determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed fixed to {seed}")


# ============================================================================
# Device Management
# ============================================================================

def get_device() -> torch.device:
    """
    Automatically select available device: cuda > cpu

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f" Using device: CUDA ({device_name})")
    else:
        device = torch.device("cpu")
        print(f" Using device: CPU")

    return device


def clear_cuda_cache() -> None:
    """
    Clear CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Loss Functions
# ============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for imbalanced datasets
    """

    def __init__(self, weights: torch.Tensor):
        """
        Args:
            weights: Class weights [num_classes]
        """
        super().__init__()
        # Register as buffer so it moves with the module to device
        self.register_buffer('weights', weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes]
            labels: [B]

        Returns:
            loss: scalar
        """
        return F.cross_entropy(logits, labels, weight=self.weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for each class [num_classes]
        gamma: Focusing parameter (default: 2.0)
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes]
            labels: [B]

        Returns:
            loss: scalar
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather probabilities for true labels
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
        probs_t = (probs * labels_one_hot).sum(dim=1)

        # Compute focal loss
        alpha_t = (self.alpha.to(logits.device) * labels_one_hot).sum(dim=1)
        focal_weight = alpha_t * (1 - probs_t) ** self.gamma
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for AUC/pAUC optimization

    Encourages bonafide samples to have higher scores than spoof samples
    by a margin. This helps optimize ranking metrics like AUC.

    Loss = max(0, margin - (score_bonafide - score_spoof))

    Args:
        margin: Minimum desired score difference between bonafide and spoof
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 2] - model outputs
            labels: [B] - ground truth (0=spoof, 1=bonafide)

        Returns:
            loss: scalar pairwise ranking loss
        """
        # Extract bonafide scores (probability/logit for class 1)
        scores = logits[:, 1]  # [B]

        # Get indices for each class
        bonafide_mask = (labels == 1)
        spoof_mask = (labels == 0)

        bonafide_indices = torch.where(bonafide_mask)[0]
        spoof_indices = torch.where(spoof_mask)[0]

        # If either class is missing, return 0 loss
        if len(bonafide_indices) == 0 or len(spoof_indices) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Extract scores for each class
        bonafide_scores = scores[bonafide_indices]  # [N_bonafide]
        spoof_scores = scores[spoof_indices]  # [N_spoof]

        # Create all pairwise differences
        # bonafide_scores[:, None] - spoof_scores[None, :] creates [N_bonafide, N_spoof] matrix
        score_diff = bonafide_scores[:, None] - spoof_scores[None, :]  # [N_bonafide, N_spoof]

        # Apply hinge loss: max(0, margin - score_diff)
        pairwise_loss = F.relu(self.margin - score_diff)

        # Average over all pairs
        return pairwise_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with main classification loss + pairwise ranking loss

    Total Loss = main_loss + pairwise_weight * pairwise_loss

    Args:
        main_criterion: Main loss function (CE or Focal)
        pairwise_margin: Margin for pairwise ranking
        pairwise_weight: Weight for pairwise loss component
    """

    def __init__(
        self,
        main_criterion: nn.Module,
        pairwise_margin: float = 1.0,
        pairwise_weight: float = 0.1
    ):
        super().__init__()
        self.main_criterion = main_criterion
        self.pairwise_loss = PairwiseRankingLoss(margin=pairwise_margin)
        self.pairwise_weight = pairwise_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 2]
            labels: [B]

        Returns:
            combined_loss: scalar
        """
        # Main classification loss
        main_loss = self.main_criterion(logits, labels)

        # Pairwise ranking loss
        pairwise_loss = self.pairwise_loss(logits, labels)

        # Combined loss
        total_loss = main_loss + self.pairwise_weight * pairwise_loss

        return total_loss


def create_loss_function(
    loss_type: str,
    class_weights: torch.Tensor,
    focal_alpha: torch.Tensor = None,
    focal_gamma: float = 2.0,
    enable_pairwise: bool = False,
    pairwise_margin: float = 1.0,
    pairwise_weight: float = 0.1
) -> nn.Module:
    """
    Create loss function based on type

    Args:
        loss_type: 'ce' or 'focal'
        class_weights: Class weights for CE loss [num_classes]
        focal_alpha: Alpha parameter for focal loss [num_classes]
        focal_gamma: Gamma parameter for focal loss
        enable_pairwise: Whether to add pairwise ranking loss
        pairwise_margin: Margin for pairwise ranking loss
        pairwise_weight: Weight for pairwise loss component

    Returns:
        Loss function module
    """
    print(f"\n[Loss Function] Creating loss function: {loss_type}")

    # Create main loss
    if loss_type == 'ce':
        print(f"  - Type: Weighted Cross Entropy")
        print(f"  - Class weights: {class_weights.tolist()}")
        main_criterion = WeightedCrossEntropyLoss(class_weights)

    elif loss_type == 'focal':
        if focal_alpha is None:
            focal_alpha = class_weights
        print(f"  - Type: Focal Loss")
        print(f"  - Alpha: {focal_alpha.tolist()}")
        print(f"  - Gamma: {focal_gamma}")
        main_criterion = FocalLoss(focal_alpha, focal_gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'ce' or 'focal'")

    # Add pairwise ranking loss if enabled
    if enable_pairwise:
        print(f"  - Pairwise Ranking Loss: ENABLED")
        print(f"    • Margin: {pairwise_margin}")
        print(f"    • Weight: {pairwise_weight}")
        return CombinedLoss(main_criterion, pairwise_margin, pairwise_weight)

    return main_criterion


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and threshold

    Args:
        scores: Prediction scores (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find EER point where FPR = FNR
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_target: float = 0.05
) -> Tuple[float, float]:
    """
    Compute minimum Detection Cost Function (minDCF)

    Args:
        scores: Prediction scores (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of miss (false negative)
        c_fa: Cost of false alarm (false positive)
        p_target: Prior probability of target (bonafide)

    Returns:
        (min_dcf, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Compute DCF for each threshold
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

    # Find minimum DCF
    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]

    return min_dcf, min_dcf_threshold


def compute_act_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05
) -> float:
    """
    Compute actual Detection Cost Function (actDCF) at EER threshold

    Args:
        scores: Prediction scores (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of miss
        c_fa: Cost of false alarm
        p_target: Prior probability of target

    Returns:
        act_dcf: Actual DCF at EER threshold
    """
    _, eer_threshold = compute_eer(scores, labels)

    # Make predictions at EER threshold
    predictions = (scores >= eer_threshold).astype(int)

    # Compute confusion matrix elements
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    # Compute rates
    fnr = fn / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    # Compute actual DCF
    act_dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

    return act_dcf


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Calibrated Log-Likelihood Ratio (CLLR) cost

    CLLR measures the calibration quality of scores as posterior probabilities.
    Lower values are better (0 is perfect, higher means worse calibration).

    CLLR = 0.5 * (C_llr_target + C_llr_nontarget)
    where:
    - C_llr_target = mean(-log2(P(target|score))) for target samples
    - C_llr_nontarget = mean(-log2(1 - P(target|score))) for nontarget samples

    Args:
        scores: Posterior probability scores P(bonafide|x) in [0, 1] (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)

    Returns:
        cllr: Calibrated Log-Likelihood Ratio cost
    """
    # Ensure scores are in valid range
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # Separate bonafide and spoof samples
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    bonafide_scores = scores[bonafide_mask]
    spoof_scores = scores[spoof_mask]

    # Compute log-likelihood ratio costs
    # For bonafide samples: want high scores (close to 1)
    # Cost = -log2(score) = negative log-likelihood
    c_llr_bonafide = -np.log2(bonafide_scores).mean()

    # For spoof samples: want low scores (close to 0)
    # Cost = -log2(1 - score)
    c_llr_spoof = -np.log2(1 - spoof_scores).mean()

    # CLLR is the average of both costs
    cllr = 0.5 * (c_llr_bonafide + c_llr_spoof)

    return cllr


def compute_prior_log_odds_shift(
    prior_cal: float,
    prior_eval: float
) -> float:
    """
    Compute log-odds shift for prior correction

    When calibrating scores on one dataset (e.g., validation) and evaluating
    on another with different class priors, we need to adjust the scores.

    The shift is: log(P_eval / P_cal * (1 - P_cal) / (1 - P_eval))

    This is added to the log-odds: logit_corrected = logit + shift

    Args:
        prior_cal: Prior probability of positive class in calibration set
        prior_eval: Prior probability of positive class in evaluation set

    Returns:
        shift: Log-odds shift value
    """
    eps = 1e-10
    prior_cal = np.clip(prior_cal, eps, 1 - eps)
    prior_eval = np.clip(prior_eval, eps, 1 - eps)

    # Compute log-odds shift
    # shift = log(P_eval / (1 - P_eval)) - log(P_cal / (1 - P_cal))
    shift = np.log(prior_eval / (1 - prior_eval)) - np.log(prior_cal / (1 - prior_cal))

    return shift


def compute_all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        logits: Model outputs [N, 2]
        labels: Ground truth labels [N] (0=spoof, 1=bonafide)

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Get probabilities and predictions
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # Softmax
    bonafide_scores = probs[:, 1]  # Probability of bonafide (class 1)
    predictions = np.argmax(logits, axis=1)

    # Compute metrics
    eer, _ = compute_eer(bonafide_scores, labels)
    min_dcf, _ = compute_min_dcf(bonafide_scores, labels)
    act_dcf = compute_act_dcf(bonafide_scores, labels)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    # Compute AUC-ROC
    fpr, tpr, _ = roc_curve(labels, bonafide_scores, pos_label=1)
    auc_roc = auc(fpr, tpr)

    metrics = {
        'eer': eer,
        'min_dcf': min_dcf,
        'act_dcf': act_dcf,
        'accuracy': accuracy,
        'f1_macro': f1,
        'recall_macro': recall,
        'auc_roc': auc_roc
    }

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics in a formatted way

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for print statements
    """
    print(f"{prefix}Metrics:")
    print(f"{prefix}  - EER: {metrics['eer']:.4f}")
    print(f"{prefix}  - minDCF: {metrics['min_dcf']:.4f}")
    print(f"{prefix}  - actDCF: {metrics['act_dcf']:.4f}")
    if 'cllr' in metrics:
        print(f"{prefix}  - CLLR: {metrics['cllr']:.4f}")
    print(f"{prefix}  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}  - F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"{prefix}  - Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"{prefix}  - AUC-ROC: {metrics['auc_roc']:.4f}")


def print_classification_report_wrapper(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_names: list = None
) -> None:
    """
    Print sklearn classification report

    Args:
        logits: Model outputs [N, 2]
        labels: Ground truth labels [N]
        target_names: Class names for display
    """
    if target_names is None:
        target_names = ['spoof', 'bonafide']

    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    predictions = np.argmax(logits, axis=1)

    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        labels,
        predictions,
        target_names=target_names,
        digits=4
    ))
    print("="*80)


# ============================================================================
# Model Utilities
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


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = False,
    desc: str = "Evaluation"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate model on a dataset and return logits and labels
    Supports Test-Time Augmentation (TTA)

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        use_tta: Whether TTA is enabled (affects batch shape)
        desc: Description for progress bar

    Returns:
        (all_logits, all_labels) as torch.Tensors
    """
    from tqdm import tqdm

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True)
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
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def apply_platt_calibration(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray
) -> Tuple[np.ndarray, 'LogisticRegression']:
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
    from sklearn.linear_model import LogisticRegression

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


def evaluate_with_calibration(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        dev_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        device: torch.device,
        apply_calibration: bool = True,
        enable_prior_correction: bool = True
) -> Dict[str, Dict[str, any]]:
    """
    Complete evaluation pipeline with calibration and prior correction
    Evaluates on train/dev/eval sets, applies calibration using dev set as reference

    Args:
        model: Model to evaluate
        train_loader: Training data loader
        dev_loader: Development/validation data loader (used as calibration reference)
        eval_loader: Evaluation/test data loader
        device: Device to evaluate on
        apply_calibration: Whether to apply Platt calibration
        enable_prior_correction: Whether to apply prior correction (only if calibration enabled)

    Returns:
        Dictionary with results for each dataset:
        {
            'train': {'logits': ..., 'labels': ..., 'initial_metrics': ..., 'calibrated_metrics': ...},
            'dev': {'logits': ..., 'labels': ..., 'initial_metrics': ..., 'calibrated_metrics': ...},
            'eval': {'logits': ..., 'labels': ..., 'initial_metrics': ..., 'calibrated_metrics': ...}
        }
    """
    print("\n" + "="*80)
    print("COMPLETE EVALUATION WITH CALIBRATION PIPELINE")
    print("="*80)
    print(f"  - Apply calibration: {apply_calibration}")
    print(f"  - Apply prior correction: {enable_prior_correction}")

    results = {}

    # Step 1: Evaluate on all datasets
    print("\n" + "="*80)
    print("STEP 1: EVALUATE ON ALL DATASETS")
    print("="*80)

    datasets = {
        'train': (train_loader, False),  # (loader, use_tta)
        'dev': (dev_loader, True),
        'eval': (eval_loader, True)
    }

    for dataset_name, (loader, use_tta) in datasets.items():
        print(f"\n[Evaluating] {dataset_name.upper()} set (TTA: {'Enabled' if use_tta else 'Disabled'})")
        logits, labels = evaluate_model(model, loader, device, use_tta=use_tta, desc=f"Evaluating {dataset_name}")

        # Convert to numpy
        logits_np = logits.numpy()
        labels_np = labels.numpy()

        # Compute initial metrics (without calibration)
        print(f"[Computing] Initial metrics for {dataset_name}")
        probs = np.exp(logits_np) / np.exp(logits_np).sum(axis=1, keepdims=True)
        bonafide_probs = probs[:, 1]
        initial_metrics = compute_metrics_from_scores(bonafide_probs, labels_np)

        # Store results
        results[dataset_name] = {
            'logits': logits,  # Keep torch format
            'logits_np': logits_np,
            'labels': labels,  # Keep torch format
            'labels_np': labels_np,
            'initial_metrics': initial_metrics,
            'initial_scores': bonafide_probs
        }

        print(f"[SUCCESS] {dataset_name.upper()} - Collected {len(labels_np)} samples")

    # Step 2: Apply calibration and prior correction (if enabled)
    if apply_calibration:
        print("\n" + "="*80)
        print("STEP 2: APPLY CALIBRATION AND PRIOR CORRECTION")
        print("="*80)

        # Use dev set as calibration reference
        cal_logits = results['dev']['logits_np']
        cal_labels = results['dev']['labels_np']

        print(f"\nUsing 'dev' as calibration reference")

        # Apply calibration to eval set (and optionally dev for comparison)
        for dataset_name in ['dev', 'eval']:
            print(f"\n{'-'*80}")
            print(f"Processing: {dataset_name.upper()}")
            print(f"{'-'*80}")

            test_logits = results[dataset_name]['logits_np']
            test_labels = results[dataset_name]['labels_np']

            # Apply Platt calibration
            print(f"\n[Calibration] Applying Platt calibration to {dataset_name}")
            calibrated_scores, calibrator = apply_platt_calibration(
                cal_logits, cal_labels, test_logits
            )

            # Apply prior correction (if enabled)
            if enable_prior_correction:
                print(f"\n[Prior Correction] Applying prior correction to {dataset_name}")
                final_scores = apply_prior_correction(
                    cal_labels, test_labels, calibrated_scores
                )
            else:
                final_scores = calibrated_scores

            # Compute final metrics
            print(f"\n[Computing] Final metrics for {dataset_name}")
            final_metrics = compute_metrics_from_scores(final_scores, test_labels)

            # Store calibrated results
            results[dataset_name]['calibrated_scores'] = final_scores
            results[dataset_name]['calibrated_metrics'] = final_metrics

        print(f"\n[SUCCESS] Calibration and prior correction complete")
    else:
        print(f"\n[INFO] Calibration disabled, skipping calibration step")

    return results


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
) -> None:
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        load_path: str,
        device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict[str, float]]:
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        load_path: Path to checkpoint
        device: Device to load to

    Returns:
        (model, optimizer, epoch, metrics)
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    print(f" Checkpoint loaded from {load_path}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Metrics: {metrics}")

    return model, optimizer, epoch, metrics


def save_model(
    model_save_dir: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_args,
    model_args,
    train_args,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float]
) -> None:
    """
    Save model weights and all configuration/metrics to JSON

    Args:
        model_save_dir: Directory to save model and JSON
        model_name: Name for the model (without extension)
        model: PyTorch model
        optimizer: Optimizer
        data_args: Data processing arguments
        model_args: Model architecture arguments
        train_args: Training arguments
        train_metrics: Metrics on training set
        val_metrics: Metrics on validation set
        test_metrics: Metrics on test set
    """
    import json
    import os
    from dataclasses import asdict

    def convert_to_python_types(obj):
        """Recursively convert numpy/torch types to native Python types"""
        if isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj

    # Create model directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Prepare data to save
    save_data = {
        'data_process_args': asdict(data_args) if hasattr(data_args, '__dataclass_fields__') else vars(data_args),
        'model_args': asdict(model_args) if hasattr(model_args, '__dataclass_fields__') else vars(model_args),
        'train_args': asdict(train_args) if hasattr(train_args, '__dataclass_fields__') else vars(train_args),
        'metrics': {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
    }

    # Convert all numpy/torch types to native Python types
    save_data = convert_to_python_types(save_data)

    # Save JSON file
    json_path = os.path.join(model_save_dir, f"{model_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    print(f"[✓] Configuration and metrics saved to {json_path}")

    # Save model weights
    model_path = os.path.join(model_save_dir, f"{model_name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"[✓] Model weights saved to {model_path}")


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Early stopping handler

    Args:
        patience: Number of epochs to wait before stopping
        mode: 'min' or 'max' (whether lower or higher is better)
        delta: Minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda new, best: new < best - delta
        else:
            self.is_better = lambda new, best: new > best + delta

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current metric score

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
