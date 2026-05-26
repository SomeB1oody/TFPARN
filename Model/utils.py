"""
Helper functions for ASVspoof5 training.
Covers seed setup, device selection, evaluation metrics, and loss functions.
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
    Set the random seed so runs are reproducible

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
    # Make cudnn deterministic too
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed fixed to {seed}")


# Device Management
def get_device() -> torch.device:
    """
    Pick a device, preferring CUDA over CPU

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


# Loss Functions
class FocalLoss(nn.Module):
    """
    Focal loss, helps with class imbalance

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: per-class weight [num_classes]
        gamma: focusing parameter (default 2.0)
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

        # Clamp to avoid numerical issues
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Probability of the true label for each sample
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

    Pushes bonafide samples to score higher than spoof samples by a margin.
    Loss = max(0, margin - (score_bonafide - score_spoof))

    Args:
        margin: wanted score gap between bonafide and spoof
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 2], model outputs
            labels: [B], ground truth (0=spoof, 1=bonafide)

        Returns:
            loss: scalar pairwise ranking loss
        """
        # Bonafide score is the class 1 logit
        scores = logits[:, 1]  # [B]

        # Indices for each class
        bonafide_mask = (labels == 1)
        spoof_mask = (labels == 0)

        bonafide_indices = torch.where(bonafide_mask)[0]
        spoof_indices = torch.where(spoof_mask)[0]

        # If a class is missing, there are no pairs, so return 0
        if len(bonafide_indices) == 0 or len(spoof_indices) == 0:
            return torch.tensor(0.0, device=logits.device)

        # Scores for each class
        bonafide_scores = scores[bonafide_indices]  # [N_bonafide]
        spoof_scores = scores[spoof_indices]  # [N_spoof]

        # All pairwise differences, gives an [N_bonafide, N_spoof] matrix
        score_diff = bonafide_scores[:, None] - spoof_scores[None, :]  # [N_bonafide, N_spoof]

        # Hinge loss: max(0, margin - score_diff)
        pairwise_loss = F.relu(self.margin - score_diff)

        # Average over all pairs
        return pairwise_loss.mean()


class CombinedLoss(nn.Module):
    """
    Main classification loss plus a pairwise ranking loss

    Total Loss = main_loss + pairwise_weight * pairwise_loss

    Args:
        main_criterion: main loss function (CE or Focal)
        pairwise_margin: margin for the pairwise ranking loss
        pairwise_weight: weight on the pairwise loss
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

        # Combine them
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
    Build a loss function for the given type

    Args:
        loss_type: 'ce' or 'focal'
        focal_alpha: alpha for focal loss [num_classes]
        focal_gamma: gamma for focal loss
        enable_pairwise: also add the pairwise ranking loss
        pairwise_margin: margin for the pairwise ranking loss
        pairwise_weight: weight on the pairwise loss

    Returns:
        Loss function module
    """
    print(f"\n[Loss Function] Creating loss function: {loss_type}")

    # Main loss
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

    # Wrap with the pairwise ranking loss if requested
    if enable_pairwise:
        print(f"  - Pairwise Ranking Loss: ENABLED")
        print(f"    • Margin: {pairwise_margin}")
        print(f"    • Weight: {pairwise_weight}")
        return CombinedLoss(main_criterion, pairwise_margin, pairwise_weight)

    return main_criterion


# Evaluation Metrics
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute the Equal Error Rate (EER) and its threshold

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: ground truth labels (0=spoof, 1=bonafide)

    Returns:
        (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # EER is the point where FPR equals FNR
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

    Per the ASVspoof5 spec:
    - DCF(t) = C_miss * (1 - pi_spf) * P_miss(t) + C_fa * pi_spf * P_fa(t)
    - normalized: DCF'(t) = beta * P_miss(t) + P_fa(t)
    - where beta = C_miss * (1 - pi_spf) / (C_fa * pi_spf), about 1.90
    - minDCF is the smallest DCF'(t) over all thresholds t

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: ground truth labels (0=spoof, 1=bonafide)
        c_miss: cost of missing a bonafide (false negative), default 1.0
        c_fa: cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: prior probability of a spoofing attack, default 0.05

    Returns:
        (min_dcf_normalized, threshold): the normalized minDCF and its threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # Normalized DCF at each threshold
    # DCF'(t) = beta * P_miss(t) + P_fa(t)
    dcf_normalized = beta * fnr + fpr

    # Take the smallest normalized DCF
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

    Per the ASVspoof5 spec:
    - tau_bayes = -log(beta), where beta = C_miss * (1 - pi_spf) / (C_fa * pi_spf), about 1.90
    - actDCF = DCF'(tau_bayes), the normalized DCF at the Bayes-optimal threshold

    Note: this assumes scores are log-likelihood ratios. If they are probabilities, they need converting first.

    Args:
        scores: prediction scores (higher means more likely bonafide)
        labels: ground truth labels (0=spoof, 1=bonafide)
        c_miss: cost of missing a bonafide (false negative), default 1.0
        c_fa: cost of a false alarm on spoof (false positive), default 10.0
        pi_spf: prior probability of a spoofing attack, default 0.05

    Returns:
        act_dcf_normalized: normalized actual DCF at the Bayes threshold
    """
    # beta factor
    beta = (c_miss * (1 - pi_spf)) / (c_fa * pi_spf)

    # Bayes-optimal threshold is -log(beta)
    # Scores are probabilities in [0,1], so turn them into log-likelihood ratios (log-odds) first
    eps = 1e-10
    scores_clipped = np.clip(scores, eps, 1 - eps)

    # Probability to log-likelihood ratio
    # LLR = log(P(bonafide|x) / P(spoof|x))
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
    fnr = fn / (tp + fn + 1e-10)  # P_miss, miss rate for bonafide
    fpr = fp / (fp + tn + 1e-10)  # P_fa, false alarm rate for spoof

    # Normalized actual DCF
    # DCF'(t) = beta * P_miss(t) + P_fa(t)
    act_dcf_normalized = beta * fnr + fpr

    return act_dcf_normalized


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute the Cost of Log-Likelihood Ratio (C_llr).

    Per the ASVspoof5 spec:
    C_llr = 1/(2*log(2)) * [mean over bonafide of log(1 + e^(-s_i)) + mean over spoof of log(1 + e^(s_j))]

    where s_i are bonafide scores and s_j are spoof scores.
    Scores should be log-likelihood ratios. Lower is better calibrated (0 is perfect).

    Args:
        scores: scores treated as log-likelihood ratios (higher means more likely bonafide)
        labels: ground truth labels (0=spoof, 1=bonafide)

    Returns:
        cllr: Cost of Log-Likelihood Ratio
    """
    # Clip into a valid range, and convert to LLRs if these are probabilities
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # If scores are probabilities in [0,1], turn them into log-likelihood ratios
    # LLR = log(P(bonafide|x) / P(spoof|x))
    if np.all((scores >= 0) & (scores <= 1)):
        llr_scores = np.log(scores / (1 - scores))
    else:
        llr_scores = scores

    # Split bonafide and spoof samples
    bonafide_mask = (labels == 1)
    spoof_mask = (labels == 0)

    bonafide_llrs = llr_scores[bonafide_mask]
    spoof_llrs = llr_scores[spoof_mask]

    # Per the ASVspoof5 formula
    # Bonafide samples: log(1 + e^(-s_i))
    c_llr_bonafide = np.mean(np.log(1 + np.exp(-bonafide_llrs)))

    # Spoof samples: log(1 + e^(s_j))
    c_llr_spoof = np.mean(np.log(1 + np.exp(spoof_llrs)))

    # Normalize by 1/(2*log(2))
    cllr = (1 / (2 * np.log(2))) * (c_llr_bonafide + c_llr_spoof)

    return cllr


def compute_prior_log_odds_shift(
    prior_cal: float,
    prior_eval: float
) -> float:
    """
    Compute the log-odds shift that corrects for a class prior mismatch.

    When the calibration set and the evaluation set have different class
    balances, scores need a fix: logit_corrected = logit + shift.

    Args:
        prior_cal: prior probability of the positive class in the calibration set
        prior_eval: prior probability of the positive class in the evaluation set

    Returns:
        shift: the log-odds shift
    """
    eps = 1e-10
    prior_cal = np.clip(prior_cal, eps, 1 - eps)
    prior_eval = np.clip(prior_eval, eps, 1 - eps)

    # Log-odds shift
    # shift = log(P_eval / (1 - P_eval)) - log(P_cal / (1 - P_cal))
    shift = np.log(prior_eval / (1 - prior_eval)) - np.log(prior_cal / (1 - prior_cal))

    return shift


def compute_all_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all the evaluation metrics

    Args:
        logits: model outputs [N, 2]
        labels: ground truth labels [N] (0=spoof, 1=bonafide)

    Returns:
        Dict of metrics
    """
    # To numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Probabilities and predictions
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
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
        'accuracy': accuracy,
        'f1_macro': f1,
        'recall_macro': recall,
        'auc_roc': auc_roc
    }

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics in a tidy format

    Args:
        metrics: dict of metrics
        prefix: prefix for each printed line
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
    Print sklearn's classification report

    Args:
        logits: model outputs [N, 2]
        labels: ground truth labels [N]
        target_names: class names to show
    """
    if target_names is None:
        target_names = ['spoof', 'bonafide']

    # To numpy
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
    Load model weights from a checkpoint, with error handling.
    Also copes with checkpoints whose structure does not match the current model.

    Args:
        model: model to load the weights into
        checkpoint_path: path to the checkpoint file
        device: device to load the weights onto
        strict: enforce an exact state_dict match

    Returns:
        Model with the weights loaded
    """
    print(f"\n[Loading] Loading model from {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Pull out the state dict, handling the different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  - Checkpoint format: standard (with model_state_dict)")
                # Show extra info if it is there
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

        # Load. The return type varies, it can be _IncompatibleKeys or None
        res = model.load_state_dict(state_dict, strict=strict)

        # Works across PyTorch versions (some return _IncompatibleKeys, some return None)
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
    Run the model over a dataset and return logits and labels.
    Supports Test-Time Augmentation (TTA).

    Args:
        model: model to evaluate
        dataloader: DataLoader for the dataset
        device: device to run on
        use_tta: whether TTA is on (changes the batch shape)
        desc: label for the progress bar

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
                # TTA on: waveforms are [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # Flatten to [B*num_crops, C, T] so all crops go through at once
                waveforms_flat = waveforms.view(B * num_crops, C, T)

                # Forward pass on all crops
                logits_flat = model(waveforms_flat)

                # Reshape back to [B, num_crops, 2]
                logits_crops = logits_flat.view(B, num_crops, 2)

                # Average the logits over the crops
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # Normal inference: waveforms are [B, C, T]
                logits = model(waveforms)

            # Collect results
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Join all the batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


def apply_platt_calibration(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray
) -> Tuple[np.ndarray, 'LogisticRegression']:
    """
    Apply Platt calibration to turn model scores into calibrated probabilities.
    Fits a logistic regression on the calibration set, then transforms the test scores.

    Args:
        cal_logits: calibration logits [N_cal, 2]
        cal_labels: calibration labels [N_cal]
        test_logits: test logits [N_test, 2]

    Returns:
        (calibrated_test_scores, calibrator): calibrated probabilities and the fitted calibrator
    """
    from sklearn.linear_model import LogisticRegression

    # Bonafide score is the difference of the two logits (log odds)
    cal_scores = cal_logits[:, 1] - cal_logits[:, 0]  # log odds
    test_scores = test_logits[:, 1] - test_logits[:, 0]

    # Fit logistic regression on the calibration set
    # It learns calibrated_score = a * score + b
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(cal_scores.reshape(-1, 1), cal_labels)

    print(f"  - Calibration parameters: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")

    # Calibrated probabilities for the test set
    calibrated_probs = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]

    return calibrated_probs, calibrator


def apply_prior_correction(
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    calibrated_scores: np.ndarray
) -> np.ndarray:
    """
    Adjust calibrated scores when the class priors differ between sets.
    Applies a log-odds shift to fix the prior mismatch.

    Args:
        cal_labels: calibration labels [N_cal]
        test_labels: test labels [N_test]
        calibrated_scores: calibrated probability scores [N_test]

    Returns:
        corrected_scores: prior-corrected probability scores [N_test]
    """
    # Class priors (fraction of bonafide samples)
    prior_cal = np.mean(cal_labels == 1)
    prior_test = np.mean(test_labels == 1)

    print(f"  - Calibration set prior P(bonafide): {prior_cal:.4f}")
    print(f"  - Test set prior P(bonafide): {prior_test:.4f}")

    # Log-odds shift
    shift = compute_prior_log_odds_shift(prior_cal, prior_test)
    print(f"  - Log-odds shift: {shift:.4f}")

    # Convert probabilities to log-odds, add the shift, then convert back
    # logit = log(p / (1-p))
    eps = 1e-10  # small value to avoid log(0)
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
        scores: probability scores (higher means more likely bonafide) [N]
        labels: ground truth labels (0=spoof, 1=bonafide) [N]

    Returns:
        Dict with all metrics, including CLLR
    """
    # compute_all_metrics wants logits, so rebuild them from the probabilities
    eps = 1e-10
    scores = np.clip(scores, eps, 1 - eps)

    # Stand-in logits: [log(1-p), log(p)]
    logits = np.stack([np.log(1 - scores), np.log(scores)], axis=1)

    # Standard metrics via the existing function
    metrics = compute_all_metrics(torch.from_numpy(logits), torch.from_numpy(labels))

    # CLLR (Log-Likelihood Ratio cost)
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
    Full evaluation with Platt calibration and prior correction.
    Uses the dev set as the calibration reference and applies it to the dev and eval sets.

    Args:
        model: model to evaluate
        train_loader: training data loader
        dev_loader: dev data loader (the calibration reference)
        eval_loader: evaluation data loader
        device: device to run on
        apply_calibration: whether to apply Platt calibration
        enable_prior_correction: whether to correct for prior mismatch

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

    # Step 1: evaluate on all datasets
    print("\n" + "="*80)
    print("STEP 1: EVALUATE ON ALL DATASETS")
    print("="*80)

    datasets = {
        'train': (train_loader, False),  # (loader, use_tta)
        'dev': (dev_loader, True),  # TTA on for the final evaluation
        'eval': (eval_loader, True)  # TTA on for the final evaluation
    }

    for dataset_name, (loader, use_tta) in datasets.items():
        print(f"\n[Evaluating] {dataset_name.upper()} set (TTA: {'Enabled' if use_tta else 'Disabled'})")
        logits, labels = evaluate_model(model, loader, device, use_tta=use_tta, desc=f"Evaluating {dataset_name}")

        # To numpy
        logits_np = logits.numpy()
        labels_np = labels.numpy()

        # Initial metrics (no calibration)
        print(f"[Computing] Initial metrics for {dataset_name}")
        probs = np.exp(logits_np) / np.exp(logits_np).sum(axis=1, keepdims=True)
        bonafide_probs = probs[:, 1]
        initial_metrics = compute_metrics_from_scores(bonafide_probs, labels_np)

        # Store results
        results[dataset_name] = {
            'logits': logits,  # keep the torch version
            'logits_np': logits_np,
            'labels': labels,  # keep the torch version
            'labels_np': labels_np,
            'initial_metrics': initial_metrics,
            'initial_scores': bonafide_probs
        }

        print(f"[SUCCESS] {dataset_name.upper()} - Collected {len(labels_np)} samples")

    # Step 2: apply calibration and prior correction (if enabled)
    if apply_calibration:
        print("\n" + "="*80)
        print("STEP 2: APPLY CALIBRATION AND PRIOR CORRECTION")
        print("="*80)

        # Use the dev set as the calibration reference
        cal_logits = results['dev']['logits_np']
        cal_labels = results['dev']['labels_np']

        print(f"\nUsing 'dev' as calibration reference")

        # Calibrate the eval set, and the dev set too for comparison
        for dataset_name in ['dev', 'eval']:
            print(f"\n{'-'*80}")
            print(f"Processing: {dataset_name.upper()}")
            print(f"{'-'*80}")

            test_logits = results[dataset_name]['logits_np']
            test_labels = results[dataset_name]['labels_np']

            # Platt calibration
            print(f"\n[Calibration] Applying Platt calibration to {dataset_name}")
            calibrated_scores, calibrator = apply_platt_calibration(
                cal_logits, cal_labels, test_logits
            )

            # Prior correction (if enabled)
            if enable_prior_correction:
                print(f"\n[Prior Correction] Applying prior correction to {dataset_name}")
                final_scores = apply_prior_correction(
                    cal_labels, test_labels, calibrated_scores
                )
            else:
                final_scores = calibrated_scores

            # Final metrics
            print(f"\n[Computing] Final metrics for {dataset_name}")
            final_metrics = compute_metrics_from_scores(final_scores, test_labels)

            # Store the calibrated results
            results[dataset_name]['calibrated_scores'] = final_scores
            results[dataset_name]['calibrated_metrics'] = final_metrics

        print(f"\n[SUCCESS] Calibration and prior correction complete")
    else:
        print(f"\n[INFO] Calibration disabled, skipping calibration step")

    return results


def count_parameters(model: nn.Module) -> int:
    """
    Count the model's trainable parameters

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
    Save the model weights, plus all the config and metrics as JSON

    Args:
        model_save_dir: directory to save the model and JSON into
        model_name: name for the model (no extension)
        model: PyTorch model
        optimizer: optimizer
        data_args: data processing arguments
        model_args: model architecture arguments
        train_args: training arguments
        train_metrics: metrics on the training set
        val_metrics: metrics on the validation set
        test_metrics: metrics on the test set
    """
    import json
    import os
    from dataclasses import asdict

    def convert_to_python_types(obj):
        """Recursively turn numpy/torch types into plain Python types"""
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

    # Make the model directory
    os.makedirs(model_save_dir, exist_ok=True)

    # Gather the values to save
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

    # Turn any numpy/torch types into plain Python types
    save_data = convert_to_python_types(save_data)

    # Write the JSON file
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
    Early stopping helper

    Args:
        patience: epochs to wait before stopping
        mode: 'min' or 'max' (whether lower or higher is better)
        delta: smallest change that counts as an improvement
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
        Check whether training should stop

        Args:
            score: current metric score

        Returns:
            True if it should stop, otherwise False
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
