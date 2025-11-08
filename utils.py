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
