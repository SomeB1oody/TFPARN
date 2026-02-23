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


# Random Seed
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
    # Ensure cudnn determinism for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed fixed to {seed}")


# Device Management
def get_device() -> torch.device:
    """
    Automatically select available device: cuda > cpu

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using device: CUDA ({device_name})")
    else:
        device = torch.device("cpu")
        print(f"[Device] Using device: CPU")

    return device


def clear_cuda_cache() -> None:
    """
    Clear CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Loss Functions
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

        # Clamp probabilities for numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Gather probabilities for true labels
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
        probs_t = (probs * labels_one_hot).sum(dim=1)

        # Compute focal loss
        alpha_t = (self.alpha.to(logits.device) * labels_one_hot).sum(dim=1)
        focal_weight = alpha_t * torch.clamp((1 - probs_t) ** self.gamma, min=1e-7)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for optimizing ranking-based metrics

    Encourages bonafide samples to score higher than spoof samples by a margin
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
    focal_alpha: torch.Tensor = None,
    focal_gamma: float = 1.0,
    enable_pairwise: bool = False,
    pairwise_margin: float = 1.0,
    pairwise_weight: float = 0.1
) -> nn.Module:
    """
    Create loss function based on type

    Args:
        loss_type: 'ce' or 'focal'
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
        print(f"  - Type: Cross Entropy (no class weights)")
        main_criterion = nn.CrossEntropyLoss()

    elif loss_type == 'focal':
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


# Evaluation Metrics
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
    pi_spf: float = 0.05
) -> Tuple[float, float]:
    """
    Compute minimum Detection Cost Function (minDCF) for ASVspoof5 Track 1

    Following ASVspoof5 specification:
    - DCF(τ) = C_miss * (1 - π_spf) * P_miss(τ) + C_fa * π_spf * P_fa(τ)
    - Normalized: DCF'(τ) = β * P_miss(τ) + P_fa(τ)
    - where β = C_miss * (1 - π_spf) / (C_fa * π_spf) ≈ 1.90
    - minDCF = min_τ DCF'(τ)

    Args:
        scores: Prediction scores (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of missing a bonafide (false negative), default=1.0
        c_fa: Cost of false alarm on spoof (false positive), default=10.0
        pi_spf: Prior probability of spoofing attack (π_spf), default=0.05

    Returns:
        (min_dcf_normalized, threshold): Normalized minDCF and its threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Compute β factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # Compute normalized DCF for each threshold
    # DCF'(τ) = β * P_miss(τ) + P_fa(τ)
    dcf_normalized = beta * fnr + fpr

    # Find minimum normalized DCF
    min_dcf_idx = np.argmin(dcf_normalized)
    min_dcf_normalized = dcf_normalized[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]

    return min_dcf_normalized, min_dcf_threshold


def compute_act_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    pi_spf: float = 0.05
) -> float:
    """
    Compute actual Detection Cost Function (actDCF) at Bayes threshold for ASVspoof5 Track 1

    Following ASVspoof5 specification:
    - τ_bayes = -log(β) where β = C_miss * (1 - π_spf) / (C_fa * π_spf) ≈ 1.90
    - actDCF = DCF'(τ_bayes) (normalized actual DCF at Bayes-optimal threshold)

    Note: This assumes detection scores can be interpreted as log-likelihood ratios.
    If scores are probabilities, conversion may be needed.

    Args:
        scores: Prediction scores (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of missing a bonafide (false negative), default=1.0
        c_fa: Cost of false alarm on spoof (false positive), default=10.0
        pi_spf: Prior probability of spoofing attack (π_spf), default=0.05

    Returns:
        act_dcf_normalized: Normalized actual DCF at Bayes threshold
    """
    # Compute β (beta factor)
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # Bayes-optimal threshold τ_bayes = -log(β)
    # For probability scores in [0,1], we need to convert to log-likelihood ratios
    # Since scores are probabilities P(bonafide|x), we compute log-odds
    eps = 1e-10
    scores_clipped = np.clip(scores, eps, 1 - eps)

    # Convert probability scores to log-likelihood ratios
    # LLR = log(P(bonafide|x) / P(spoof|x))
    llr_scores = np.log(scores_clipped / (1 - scores_clipped))

    # Bayes threshold in log-likelihood ratio space
    tau_bayes = -np.log(beta)

    # Make predictions at Bayes threshold
    predictions = (llr_scores >= tau_bayes).astype(int)

    # Compute confusion matrix elements
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    # Compute error rates
    fnr = fn / (tp + fn + 1e-10)  # P_miss (miss rate for bonafide)
    fpr = fp / (fp + tn + 1e-10)  # P_fa (false alarm rate for spoof)

    # Compute normalized actual DCF
    # DCF'(τ) = β * P_miss(τ) + P_fa(τ)
    act_dcf_normalized = beta * fnr + fpr

    return act_dcf_normalized


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Cost of Log-Likelihood Ratio (C_llr)

    Following ASVspoof5 specification:
    C_llr = 1/(2*log(2)) * [1/|Bon| * Σ log(1 + e^(-s_i)) + 1/|Spf| * Σ log(1 + e^(s_j))]

    where s_i are scores for bonafide samples and s_j are scores for spoof samples.
    Scores should be log-likelihood ratios (LLRs). Lower values indicate better calibration (0 is perfect).

    Args:
        scores: Scores interpreted as log-likelihood ratios (higher = more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)

    Returns:
        cllr: Cost of Log-Likelihood Ratio
    """
    # Ensure scores are in valid range and convert to LLRs if they are probabilities
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # Convert probability scores to log-likelihood ratios if in [0,1] range
    # LLR = log(P(bonafide|x) / P(spoof|x))
    if np.all((scores >= 0) & (scores <= 1)):
        llr_scores = np.log(scores / (1 - scores))
    else:
        llr_scores = scores

    # Separate bonafide and spoof samples
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    bonafide_llrs = llr_scores[bonafide_mask]
    spoof_llrs = llr_scores[spoof_mask]

    # Compute log-likelihood ratio costs according to ASVspoof5 formula
    # For bonafide samples: log(1 + e^(-s_i))
    c_llr_bonafide = np.mean(np.log(1 + np.exp(-bonafide_llrs)))

    # For spoof samples: log(1 + e^(s_j))
    c_llr_spoof = np.mean(np.log(1 + np.exp(spoof_llrs)))

    # C_llr with normalization factor 1/(2*log(2))
    cllr = (1 / (2 * np.log(2))) * (c_llr_bonafide + c_llr_spoof)

    return cllr


def compute_prior_log_odds_shift(
    prior_cal: float,
    prior_eval: float
) -> float:
    """
    Compute log-odds shift for correcting class prior mismatch

    When calibration set and evaluation set have different class distributions,
    scores need adjustment: logit_corrected = logit + shift

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


# Model Utilities
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

        # Try to load - handle both _IncompatibleKeys and None return types
        res = model.load_state_dict(state_dict, strict=strict)

        # Compatible with both PyTorch versions (some return _IncompatibleKeys, some return None)
        if hasattr(res, "missing_keys"):
            missing_keys, unexpected_keys = res.missing_keys, res.unexpected_keys
            if missing_keys:
                print(f"  [WARNING] Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"    - {key}")
                else:
                    print(f"    - Showing first 5: {missing_keys[:5]}")
            if unexpected_keys:
                print(f"  [WARNING] Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"    - {key}")
                else:
                    print(f"    - Showing first 5: {unexpected_keys[:5]}")
        else:
            print("[SUCCESS] Model weights loaded successfully (strict match).")

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

                # Forward pass on all crops
                logits_flat = model(waveforms_flat)

                # Reshape back to [B, num_crops, 2]
                logits_crops = logits_flat.view(B, num_crops, 2)

                # Average logits across crops
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # Normal inference: waveforms shape [B, C, T]
                logits = model(waveforms)

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
    Apply Platt calibration to convert model scores to calibrated probabilities
    Fits logistic regression on calibration set, then transforms test scores

    Args:
        cal_logits: Calibration logits [N_cal, 2]
        cal_labels: Calibration labels [N_cal]
        test_logits: Test logits [N_test, 2]

    Returns:
        (calibrated_test_scores, calibrator): Calibrated probabilities and fitted calibrator
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
    Adjust calibrated scores for different class prior distributions
    Applies log-odds shift to correct for prior mismatch

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
    Complete evaluation with Platt calibration and prior correction
    Uses dev set as calibration reference, applies to dev and eval sets

    Args:
        model: Model to evaluate
        train_loader: Training data loader
        dev_loader: Development data loader (calibration reference)
        eval_loader: Evaluation data loader
        device: Device to evaluate on
        apply_calibration: Whether to apply Platt calibration
        enable_prior_correction: Whether to correct for prior mismatch

    Returns:
        Dict with results for train/dev/eval:
        {
            'train': {'logits': ..., 'labels': ..., 'initial_metrics': ..., ...},
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
        'dev': (dev_loader, True),  # TTA enabled for final evaluation
        'eval': (eval_loader, True)  # TTA enabled for final evaluation
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


# Early Stopping
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
