"""
Evaluation script. Loads a saved model and its JSON config, then evaluates on
the dev and eval sets.
"""

import os
import json
import torch
from dataclasses import dataclass

# Import custom modules
from data_process import make_loaders
from model import create_aasist_model, AASISTArgs
from utils import (
    set_seed, get_device, load_model_weights,
    evaluate_model, compute_all_metrics, print_metrics,
    compute_cllr
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Settings for running the evaluation"""
    # Paths to the model files
    model_json_path: str = ""
    model_weights_path: str = ""

    # Evaluation settings
    batch_size_override: int = None  # override batch size (None means use the value from the JSON)
    use_tta_override: bool = None    # override the TTA setting (None means use the value from the JSON)


# ============================================================================
# Helper Functions
# ============================================================================

def load_config_from_json(json_path: str) -> dict:
    """
    Load the model config from a JSON file.

    Args:
        json_path: Path to the JSON config file

    Returns:
        Dictionary with the configuration
    """
    print(f"\n[Loading] Reading configuration from {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"[Loading] Configuration loaded successfully")

    # Show the metrics that were saved with the model
    if 'metrics' in config:
        print(f"\n[Saved Metrics]")
        if 'train' in config['metrics']:
            print(f"  Train EER: {config['metrics']['train'].get('eer', 'N/A'):.4f}")
        if 'dev' in config['metrics']:
            print(f"  Dev EER: {config['metrics']['dev'].get('eer', 'N/A'):.4f}")
        if 'eval' in config['metrics']:
            print(f"  Eval EER: {config['metrics']['eval'].get('eer', 'N/A'):.4f}")

    if 'epoch' in config:
        print(f"  Saved at epoch: {config['epoch']}")

    return config


def reconstruct_args_from_config(config: dict, eval_config: EvaluationConfig):
    """
    Rebuild the model args from the loaded config.

    Args:
        config: The loaded config dictionary
        eval_config: Overrides used only for evaluation

    Returns:
        Namespace with all the arguments
    """
    # Get the args out of the config
    if 'args' in config:
        args_dict = config['args']
    else:
        raise ValueError("Configuration file does not contain 'args' field")

    # Wrap them in a namespace
    from types import SimpleNamespace
    args = SimpleNamespace(**args_dict)

    # Apply any evaluation overrides
    if eval_config.batch_size_override is not None:
        print(f"[Override] Batch size: {args.batch_size} -> {eval_config.batch_size_override}")
        args.batch_size = eval_config.batch_size_override

    if eval_config.use_tta_override is not None:
        print(f"[Override] TTA: {args.use_tta} -> {eval_config.use_tta_override}")
        args.use_tta = eval_config.use_tta_override

    return args


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_saved_model(eval_config: EvaluationConfig):
    """
    Load a saved model and evaluate it.

    Args:
        eval_config: Evaluation configuration
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Load the config from JSON
    config = load_config_from_json(eval_config.model_json_path)

    # Rebuild the args
    args = reconstruct_args_from_config(config, eval_config)

    set_seed(args.seed)
    device = get_device()

    # ===== Step 1: Load Data =====
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Only the dev and eval loaders are needed, but make_loaders returns all three,
    # so just call it and ignore the train loader
    train_loader, dev_loader, eval_loader = make_loaders(args)

    # ===== Step 2: Create Model =====
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)

    # Build the AASIST config
    aasist_args = AASISTArgs(
        first_conv=args.first_conv,
        filts=args.filts,
        gat_dims=args.gat_dims,
        pool_ratios=args.pool_ratios,
        temperatures=args.temperatures,
        num_classes=args.num_classes
    )

    model = create_aasist_model(aasist_args, device)

    # ===== Step 3: Load Model Weights =====
    print("\n" + "="*80)
    print("STEP 3: LOADING MODEL WEIGHTS")
    print("="*80)

    model = load_model_weights(
        model,
        eval_config.model_weights_path,
        device,
        strict=True
    )

    # ===== Step 4: Evaluate on Dev Set =====
    print("\n" + "="*80)
    print("STEP 4: EVALUATING ON DEV SET")
    print("="*80)

    dev_logits, dev_labels = evaluate_model(
        model, dev_loader, device,
        use_tta=args.use_tta,
        desc="Evaluating Dev"
    )

    dev_metrics = compute_all_metrics(dev_logits, dev_labels)

    # CLLR is computed on its own from the bonafide probabilities
    import numpy as np
    probs = np.exp(dev_logits.numpy()) / np.exp(dev_logits.numpy()).sum(axis=1, keepdims=True)
    bonafide_probs = probs[:, 1]
    dev_cllr = compute_cllr(bonafide_probs, dev_labels.numpy())
    dev_metrics['cllr'] = dev_cllr

    print(f"\n[Dev Set Results]")
    print_metrics(dev_metrics, prefix="  ")

    # ===== Step 5: Evaluate on Eval Set =====
    print("\n" + "="*80)
    print("STEP 5: EVALUATING ON EVAL SET")
    print("="*80)

    eval_logits, eval_labels = evaluate_model(
        model, eval_loader, device,
        use_tta=args.use_tta,
        desc="Evaluating Eval"
    )

    eval_metrics = compute_all_metrics(eval_logits, eval_labels)

    # CLLR again, computed separately
    probs = np.exp(eval_logits.numpy()) / np.exp(eval_logits.numpy()).sum(axis=1, keepdims=True)
    bonafide_probs = probs[:, 1]
    eval_cllr = compute_cllr(bonafide_probs, eval_labels.numpy())
    eval_metrics['cllr'] = eval_cllr

    print(f"\n[Eval Set Results]")
    print_metrics(eval_metrics, prefix="  ")

    # ===== Summary =====
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    print(f"\n[Dev Set]")
    print_metrics(dev_metrics, prefix="  ")

    print(f"\n[Eval Set]")
    print_metrics(eval_metrics, prefix="  ")

    print(f"\n{'='*80}")
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")


# ============================================================================
# Main Script
# ============================================================================

def main():
    """Run the evaluation"""

    # Set up the evaluation
    eval_config = EvaluationConfig(
        model_json_path="",
        model_weights_path="",
        batch_size_override=None,  # use batch size from the JSON
        use_tta_override=None      # use TTA setting from the JSON
    )

    evaluate_saved_model(eval_config)


if __name__ == "__main__":
    main()
