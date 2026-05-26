"""
Utility functions: seed setup, device selection, evaluation metrics, and loss functions
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc,
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
        seed: Random seed
    """
    print(f"\n[Seed] Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic so results don't vary between runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed fixed to {seed}")


# Device Management
def get_device() -> torch.device:
    """
    Pick a device, preferring CUDA over CPU.

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
    Clear the CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Loss Functions
class FocalLoss(nn.Module):
    """
    Focal loss, used to deal with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weight [num_classes]
        gamma: Focusing parameter (default 2.0)
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
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Clamp to keep things numerically stable
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Pick out the probability of the true label
        labels_one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
        probs_t = (probs * labels_one_hot).sum(dim=1)

        # Focal loss
        alpha_t = (self.alpha.to(logits.device) * labels_one_hot).sum(dim=1)
        focal_weight = alpha_t * torch.clamp((1 - probs_t) ** self.gamma, min=1e-7)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss, useful for ranking-based metrics.

    Pushes bonafide samples to score higher than spoof samples by at least a margin.
    Loss = max(0, margin - (score_bonafide - score_spoof))

    Args:
        margin: Minimum wanted score gap between bonafide and spoof
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
        # Bonafide score (logit for class 1)
        scores = logits[:, 1]  # [B]

        # Indices for each class
        bonafide_mask = (labels == 1)
        spoof_mask = (labels == 0)

        bonafide_indices = torch.where(bonafide_mask)[0]
        spoof_indices = torch.where(spoof_mask)[0]

        # No pairs possible if either class is missing, so loss is 0
        if len(bonafide_indices) == 0 or len(spoof_indices) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Scores split by class
        bonafide_scores = scores[bonafide_indices]  # [N_bonafide]
        spoof_scores = scores[spoof_indices]  # [N_spoof]

        # Difference of every bonafide score against every spoof score,
        # giving an [N_bonafide, N_spoof] matrix
        score_diff = bonafide_scores[:, None] - spoof_scores[None, :]  # [N_bonafide, N_spoof]

        # Hinge loss: max(0, margin - score_diff)
        pairwise_loss = F.relu(self.margin - score_diff)

        # Average over all pairs
        return pairwise_loss.mean()


class CombinedLoss(nn.Module):
    """
    Main classification loss plus a pairwise ranking term.

    Total loss = main_loss + pairwise_weight * pairwise_loss

    Args:
        main_criterion: Main loss function (CE or Focal)
        pairwise_margin: Margin for the pairwise ranking term
        pairwise_weight: Weight on the pairwise term
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
        main_loss = self.main_criterion(logits, labels)
        pairwise_loss = self.pairwise_loss(logits, labels)
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
    Build the loss function for the given type.

    Args:
        loss_type: 'ce' or 'focal'
        focal_alpha: Alpha for focal loss [num_classes]
        focal_gamma: Gamma for focal loss
        enable_pairwise: Whether to add the pairwise ranking term
        pairwise_margin: Margin for the pairwise ranking term
        pairwise_weight: Weight on the pairwise term

    Returns:
        Loss function module
    """
    print(f"\n[Loss Function] Creating loss function: {loss_type}")

    # Build the main loss
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

    # Wrap with the pairwise ranking term if requested
    if enable_pairwise:
        print(f"  - Pairwise Ranking Loss: ENABLED")
        print(f"    • Margin: {pairwise_margin}")
        print(f"    • Weight: {pairwise_weight}")
        return CombinedLoss(main_criterion, pairwise_margin, pairwise_weight)

    return main_criterion


# Evaluation Metrics
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Equal Error Rate (EER) and its threshold.

    Args:
        scores: Prediction scores (higher means more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # EER is the point where the false positive and false negative rates meet
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

    Following the ASVspoof5 spec:
    - DCF(t) = c_miss * (1 - pi_spf) * P_miss(t) + c_fa * pi_spf * P_fa(t)
    - Normalized: DCF'(t) = beta * P_miss(t) + P_fa(t)
    - where beta = c_miss * (1 - pi_spf) / (c_fa * pi_spf), about 1.90
    - minDCF is the smallest DCF'(t) over all thresholds t

    Args:
        scores: Prediction scores (higher means more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of missing a bonafide (false negative), default 1.0
        c_fa: Cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: Prior probability of a spoofing attack, default 0.05

    Returns:
        (min_dcf_normalized, threshold): the normalized minDCF and its threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # The beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # Normalized DCF at each threshold: beta * P_miss + P_fa
    dcf_normalized = beta * fnr + fpr

    # Smallest normalized DCF
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
    Compute the actual DCF (actDCF) at the Bayes threshold for ASVspoof5 Track 1.

    Following the ASVspoof5 spec:
    - The Bayes threshold is -log(beta), where
      beta = c_miss * (1 - pi_spf) / (c_fa * pi_spf), about 1.90.
    - actDCF is the normalized DCF evaluated at that threshold.

    Note: this assumes the scores are log-likelihood ratios. If they are
    probabilities, they need to be converted first.

    Args:
        scores: Prediction scores (higher means more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)
        c_miss: Cost of missing a bonafide (false negative), default 1.0
        c_fa: Cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: Prior probability of a spoofing attack, default 0.05

    Returns:
        act_dcf_normalized: the normalized actual DCF at the Bayes threshold
    """
    # The beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # The Bayes threshold is -log(beta). The scores here are probabilities
    # P(bonafide|x) in [0,1], so first turn them into log-odds.
    eps = 1e-10
    scores_clipped = np.clip(scores, eps, 1 - eps)

    # Probabilities to log-likelihood ratios: log(P(bonafide) / P(spoof))
    llr_scores = np.log(scores_clipped / (1 - scores_clipped))

    # Bayes threshold in LLR space
    tau_bayes = -np.log(beta)

    # Predict at the Bayes threshold
    predictions = (llr_scores >= tau_bayes).astype(int)

    # Confusion matrix counts
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))

    # Error rates
    fnr = fn / (tp + fn + 1e-10)  # miss rate for bonafide
    fpr = fp / (fp + tn + 1e-10)  # false alarm rate for spoof

    # Normalized actual DCF: beta * P_miss + P_fa
    act_dcf_normalized = beta * fnr + fpr

    return act_dcf_normalized


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute the Cost of Log-Likelihood Ratio (C_llr).

    Following the ASVspoof5 spec:
    C_llr = 1/(2*log(2)) * [mean over bonafide of log(1 + exp(-s)) +
                            mean over spoof of log(1 + exp(s))]

    where s are the scores. The scores should be log-likelihood ratios (LLRs).
    Lower is better calibrated, with 0 being perfect.

    Args:
        scores: Scores treated as log-likelihood ratios (higher means more likely bonafide)
        labels: Ground truth labels (0=spoof, 1=bonafide)

    Returns:
        cllr: Cost of Log-Likelihood Ratio
    """
    # If the scores are probabilities in [0,1], turn them into LLRs
    # Done carefully to avoid overflow
    if np.all((scores >= 0) & (scores <= 1)):
        # Keep scores away from exact 0/1 so the LLR doesn't blow up
        eps = 1e-7
        scores = np.clip(scores, eps, 1.0 - eps)
        # LLR = log(p / (1-p)), then clip to keep exp from overflowing later
        llr_scores = np.log(scores / (1.0 - scores))
        llr_scores = np.clip(llr_scores, -50.0, 50.0)
    else:
        llr_scores = scores

    # Split into bonafide and spoof samples
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    bonafide_llrs = llr_scores[bonafide_mask]
    spoof_llrs = llr_scores[spoof_mask]

    # Use log1p so large values of s don't overflow exp
    def safe_logloss1p(x):
        return np.log1p(np.exp(-np.clip(x, -50, 50)))

    def safe_logsumexp_1p(x):
        return np.log1p(np.exp(np.clip(x, -50, 50)))

    c_llr_bonafide = np.mean(safe_logloss1p(bonafide_llrs))
    c_llr_spoof = np.mean(safe_logsumexp_1p(spoof_llrs))

    cllr = (1 / (2 * np.log(2))) * (c_llr_bonafide + c_llr_spoof)
    return float(cllr)


def compute_all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all the evaluation metrics.

    Args:
        logits: Model outputs [N, 2]
        labels: Ground truth labels [N] (0=spoof, 1=bonafide)

    Returns:
        Dictionary of metrics
    """
    # Move to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Probabilities and predictions (stable softmax)
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    logits_exp = np.exp(logits_shifted)
    probs = logits_exp / logits_exp.sum(axis=1, keepdims=True)
    bonafide_scores = probs[:, 1]  # probability of bonafide (class 1)
    predictions = np.argmax(logits, axis=1)

    # Metrics
    eer, _ = compute_eer(bonafide_scores, labels)
    min_dcf, _ = compute_min_dcf(bonafide_scores, labels)
    act_dcf = compute_act_dcf(bonafide_scores, labels)

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
        'cllr': compute_cllr(bonafide_scores, labels),
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
        metrics: Dictionary of metrics
        prefix: Prefix for each printed line
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

# Model Utilities
def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False
) -> nn.Module:
    """
    Load model weights from a checkpoint, with some error handling.
    Copes with checkpoints whose structure doesn't quite match the model.

    Args:
        model: Model to load the weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the weights onto
        strict: Whether to require an exact state_dict match

    Returns:
        Model with the weights loaded
    """
    print(f"\n[Loading] Loading model from {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Pull out the state dict, allowing for a few checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  - Checkpoint format: standard (with model_state_dict)")
                # Print extra info when it's there
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

        # load_state_dict may return _IncompatibleKeys or None depending on version
        res = model.load_state_dict(state_dict, strict=strict)

        # Handle both PyTorch versions: some return _IncompatibleKeys, some return None
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
    Run the model over a dataset and return the logits and labels.
    Supports Test-Time Augmentation (TTA).

    Args:
        model: Model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on
        use_tta: Whether TTA is on (changes the batch shape)
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
                # With TTA, waveforms are [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # Flatten the crops into the batch dim for one forward pass
                waveforms_flat = waveforms.view(B * num_crops, C, T)

                # Run all crops through the model
                logits_flat = model(waveforms_flat)

                # Reshape back and average over the crops
                logits_crops = logits_flat.view(B, num_crops, 2)
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # No TTA: waveforms are [B, C, T]
                logits = model(waveforms)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Join all the batches together
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels

def count_parameters(model: nn.Module) -> int:
    """
    Count the model's trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Early Stopping
class EarlyStopping:
    """
    Early stopping helper.

    Args:
        patience: How many epochs to wait without improvement before stopping
        mode: 'min' or 'max' (whether lower or higher is better)
        delta: Smallest change that counts as an improvement
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
        Decide whether to stop training.

        Args:
            score: Current metric score

        Returns:
            True if training should stop, otherwise False
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
        """Reset the early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
