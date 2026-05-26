"""
Helper functions for ASVspoof5 training: seeding, device setup, metrics, and loss functions
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
    Set the random seed so runs are reproducible.

    Args:
        seed: the random seed
    """
    print(f"\n[Seed] Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # make cudnn deterministic too
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed fixed to {seed}")


# Device Management
def get_device() -> torch.device:
    """
    Pick the device to use, preferring CUDA over CPU.

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
    Free up the CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Evaluation Metrics
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Equal Error Rate (EER) and its threshold.

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: true labels (0=spoof, 1=bonafide)

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # the EER is where FPR and FNR are closest
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
    Compute the minimum Detection Cost Function (minDCF) for ASVspoof5 Track 1.

    Follows the ASVspoof5 spec:
    - DCF(t) = c_miss * (1 - pi_spf) * P_miss(t) + c_fa * pi_spf * P_fa(t)
    - normalized: DCF'(t) = beta * P_miss(t) + P_fa(t)
    - beta = c_miss * (1 - pi_spf) / (c_fa * pi_spf), about 1.90
    - minDCF is the smallest DCF' over all thresholds

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: true labels (0=spoof, 1=bonafide)
        c_miss: cost of missing a bonafide (false negative), default 1.0
        c_fa: cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: prior probability of a spoofing attack, default 0.05

    Returns:
        (min_dcf_normalized, threshold): the normalized minDCF and its threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # the beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # normalized DCF at each threshold: beta * P_miss + P_fa
    dcf_normalized = beta * fnr + fpr

    # take the smallest one
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
    Compute the actual Detection Cost Function (actDCF) at the Bayes threshold for ASVspoof5 Track 1.

    Follows the ASVspoof5 spec:
    - the Bayes threshold is -log(beta), where beta = c_miss * (1 - pi_spf) / (c_fa * pi_spf), about 1.90
    - actDCF is the normalized DCF' evaluated at that threshold

    Note: this assumes the scores are log-likelihood ratios. If they are
    probabilities, they need to be converted first.

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: true labels (0=spoof, 1=bonafide)
        c_miss: cost of missing a bonafide (false negative), default 1.0
        c_fa: cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: prior probability of a spoofing attack, default 0.05

    Returns:
        act_dcf_normalized: the normalized actual DCF at the Bayes threshold
    """
    # the beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # the scores are probabilities P(bonafide|x), so turn them into log-odds first
    eps = 1e-10
    scores_clipped = np.clip(scores, eps, 1 - eps)

    # probability becomes log-likelihood ratio: log(P(bonafide|x) / P(spoof|x))
    llr_scores = np.log(scores_clipped / (1 - scores_clipped))

    # Bayes threshold in LLR space
    tau_bayes = -np.log(beta)

    # decide at the Bayes threshold
    predictions = (llr_scores >= tau_bayes).astype(int)

    # confusion matrix counts
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    # error rates
    fnr = fn / (tp + fn + 1e-10)  # miss rate for bonafide
    fpr = fp / (fp + tn + 1e-10)  # false alarm rate for spoof

    # normalized actual DCF: beta * P_miss + P_fa
    act_dcf_normalized = beta * fnr + fpr

    return act_dcf_normalized


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute the Cost of Log-Likelihood Ratio (C_llr).

    Follows the ASVspoof5 formula. The scores should be log-likelihood ratios.
    Lower is better calibrated, and 0 is perfect.

    Args:
        scores: scores read as log-likelihood ratios (higher means more likely bonafide)
        labels: true labels (0=spoof, 1=bonafide)

    Returns:
        cllr: the Cost of Log-Likelihood Ratio
    """
    # clip into range, then convert to LLRs if these are probabilities
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # if the scores sit in [0,1] they are probabilities, so turn them into log-odds
    if np.all((scores >= 0) & (scores <= 1)):
        llr_scores = np.log(scores / (1 - scores))
    else:
        llr_scores = scores

    # split into bonafide and spoof
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    bonafide_llrs = llr_scores[bonafide_mask]
    spoof_llrs = llr_scores[spoof_mask]

    # cost on the bonafide side
    c_llr_bonafide = np.mean(np.log(1 + np.exp(-bonafide_llrs)))

    # cost on the spoof side
    c_llr_spoof = np.mean(np.log(1 + np.exp(spoof_llrs)))

    # combine with the 1/(2*log(2)) normalization
    cllr = (1 / (2 * np.log(2))) * (c_llr_bonafide + c_llr_spoof)

    return cllr


def compute_all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all the evaluation metrics at once.

    Args:
        logits: model outputs [N, 2]
        labels: true labels [N] (0=spoof, 1=bonafide)

    Returns:
        a dict of metrics
    """
    # to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # softmax to probabilities, then the predicted class
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    bonafide_scores = probs[:, 1]  # probability of bonafide (class 1)
    predictions = np.argmax(logits, axis=1)

    # the detection metrics
    eer, _ = compute_eer(bonafide_scores, labels)
    min_dcf, _ = compute_min_dcf(bonafide_scores, labels)
    act_dcf = compute_act_dcf(bonafide_scores, labels)
    cllr = compute_cllr(bonafide_scores, labels)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    # AUC-ROC
    fpr, tpr, _ = roc_curve(labels, bonafide_scores, pos_label=1)
    auc_roc = auc(fpr, tpr)

    metrics = {
        'eer': eer,
        'min_dcf': min_dcf,
        'act_dcf': act_dcf,
        'cllr': cllr,
        'accuracy': accuracy,
        'f1_macro': f1,
        'recall_macro': recall,
        'auc_roc': auc_roc
    }

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print the metrics in a readable layout.

    Args:
        metrics: a dict of metrics
        prefix: text to put in front of each line
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

def count_parameters(model: nn.Module) -> int:
    """
    Count the model's trainable parameters.

    Args:
        model: a PyTorch model

    Returns:
        the number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Loss Functions
class FocalLoss(nn.Module):
    """
    Focal loss, which helps when the classes are imbalanced.

    Reference: "Focal Loss for Dense Object Detection"
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Set up the focal loss.

        Args:
            alpha: class weights [C] or a single number
            gamma: focusing parameter (default 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Args:
            inputs: logits [B, C]
            targets: target labels [B]

        Returns:
            the focal loss as a scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type: str, class_weights: torch.Tensor = None,
                      focal_alpha: float = 0.25, focal_gamma: float = 2.0, device=None):
    """
    Pick the loss function by name.

    Args:
        loss_type: 'ce' or 'focal'
        class_weights: class weights for the CE loss
        focal_alpha: alpha for the focal loss
        focal_gamma: gamma for the focal loss
        device: torch device

    Returns:
        the loss function
    """
    if loss_type == 'ce':
        if class_weights is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_cllr_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute CLLR straight from logits.

    Args:
        logits: model outputs [N, 2]
        labels: true labels [N]

    Returns:
        the CLLR value
    """
    # to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # softmax to probabilities
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    bonafide_scores = probs[:, 1]

    return compute_cllr(bonafide_scores, labels)