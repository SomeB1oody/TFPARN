"""
Read and evaluate trained models.
Load a model from its checkpoint and JSON config, then run it on the train, dev, and eval sets.
"""

import json
import torch
from dataclasses import dataclass
from typing import Dict

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    get_device, clear_cuda_cache, load_model_weights,
    compute_all_metrics, print_metrics, set_seed
)


@dataclass
class EvaluationConfig:
    """Settings for evaluating a model"""
    # Required: paths to the model and its config
    json_path: str = ""
    model_path: str = ""

    # Dataset paths
    train_data_dir: str = ""
    dev_data_dir: str = ""
    eval_data_dir: str = ""

    # Protocol paths
    train_protocol_dir: str = ""
    dev_protocol_dir: str = ""
    eval_protocol_dir: str = ""

    # TTA setting
    use_tta: bool = True  # only used for dev and eval


def load_config_from_json(json_path: str) -> Dict:
    """Load the config from a JSON file"""
    print(f"\n{'='*80}")
    print(f"Loading configuration from: {json_path}")
    print(f"{'='*80}")

    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f" Configuration loaded successfully")
    return config


def evaluate_model(eval_config: EvaluationConfig):
    """
    Evaluate a trained model on the train, dev, and eval sets.

    Args:
        eval_config: an EvaluationConfig with the paths and settings
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION PIPELINE")
    print("="*80)

    # Step 1: load the config from JSON
    if not eval_config.json_path:
        raise ValueError("json_path must be specified in EvaluationConfig")
    if not eval_config.model_path:
        raise ValueError("model_path must be specified in EvaluationConfig")

    config = load_config_from_json(eval_config.json_path)

    # Step 2: set the random seed
    seed = config.get('train_args', {}).get('seed', 42)
    set_seed(seed)

    # Step 3: pick the device
    device = get_device()
    clear_cuda_cache()

    # Step 4: load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Build the data processing args from the config
    data_args = DataProcessArgs()

    # Override the paths from eval_config to point at different data
    data_args.train_data_dir = eval_config.train_data_dir
    data_args.dev_data_dir = eval_config.dev_data_dir
    data_args.eval_data_dir = eval_config.eval_data_dir
    data_args.train_protocol_dir = eval_config.train_protocol_dir
    data_args.dev_protocol_dir = eval_config.dev_protocol_dir
    data_args.eval_protocol_dir = eval_config.eval_protocol_dir

    # Read the data processing settings from the JSON
    data_config = config.get('data_process_args', {})
    data_args.sample_rate = data_config.get('sample_rate', 16000)
    data_args.duration_sec = data_config.get('duration_sec', 4.0)
    data_args.mono = data_config.get('mono', True)
    data_args.normalize = data_config.get('normalize', True)
    data_args.batch_size = data_config.get('batch_size', 96)
    data_args.num_workers = data_config.get('num_workers', 8)
    data_args.prefetch_factor = data_config.get('prefetch_factor', 2)
    data_args.pin_memory = data_config.get('pin_memory', True)
    data_args.persistent_workers = data_config.get('persistent_workers', True)
    data_args.train_shuffle = False  # don't shuffle when evaluating
    data_args.seed = seed

    # TTA: off for train, optional for dev and eval
    data_args.use_rawboost = False  # no augmentation when evaluating
    data_args.rawboost_prob = 0.0
    data_args.use_tta = eval_config.use_tta
    data_args.tta_num_crops = data_config.get('tta_num_crops', 5)

    # Build the loaders
    train_loader, dev_loader, eval_loader = make_loaders(data_args)

    # Step 5: create the model
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)

    # Build the model args from the JSON
    model_config = config.get('model_args', {})
    model_args = SpeechClassifierArgs()
    model_args.n_mels = model_config.get('n_mels', 160)
    model_args.n_fft = model_config.get('n_fft', 1024)
    model_args.hop_length = model_config.get('hop_length', 160)
    model_args.sample_rate = model_config.get('sample_rate', 16000)
    model_args.d_model = model_config.get('d_model', 256)
    model_args.nhead = model_config.get('nhead', 8)
    model_args.num_layers = model_config.get('num_layers', 6)
    model_args.dim_feedforward = model_config.get('dim_feedforward', 1024)
    model_args.dropout = model_config.get('dropout', 0.3)
    model_args.activation = model_config.get('activation', 'relu')
    model_args.pooling_method = model_config.get('pooling_method', 'mean')
    model_args.top_k_ratio = model_config.get('top_k_ratio', 0.3)

    # Create the model
    model = create_model(model_args)
    model = model.to(device)

    # Step 6: load the model weights
    print("\n" + "="*80)
    print("STEP 3: LOADING MODEL WEIGHTS")
    print("="*80)
    print(f"  - Model path: {eval_config.model_path}")

    model = load_model_weights(model, eval_config.model_path, device, strict=True)
    print(f" Model weights loaded successfully")

    # Step 7: evaluate on every set
    print("\n" + "="*80)
    print("STEP 4: EVALUATING MODEL")
    print("="*80)

    model.eval()

    # Train set (no TTA)
    print(f"\n{'-'*80}")
    print("Evaluating on TRAIN set (no TTA)")
    print(f"{'-'*80}")
    train_metrics = evaluate_single_set(model, train_loader, device, use_tta=False)

    # Dev set (TTA if it's on)
    print(f"\n{'-'*80}")
    print(f"Evaluating on DEV set (TTA={'enabled' if eval_config.use_tta else 'disabled'})")
    print(f"{'-'*80}")
    dev_metrics = evaluate_single_set(model, dev_loader, device, use_tta=eval_config.use_tta)

    # Eval set (TTA if it's on)
    print(f"\n{'-'*80}")
    print(f"Evaluating on EVAL set (TTA={'enabled' if eval_config.use_tta else 'disabled'})")
    print(f"{'-'*80}")
    eval_metrics = evaluate_single_set(model, eval_loader, device, use_tta=eval_config.use_tta)

    # Step 8: print the summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)

    print(f"\n[TRAIN SET]")
    print_metrics(train_metrics, prefix="  ")

    print(f"\n[DEV SET]")
    print_metrics(dev_metrics, prefix="  ")

    print(f"\n[EVAL SET]")
    print_metrics(eval_metrics, prefix="  ")

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)


def evaluate_single_set(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model on one dataset.

    Args:
        model: model to evaluate
        data_loader: the data loader
        device: device to use
        use_tta: whether TTA is on (changes the batch shape)

    Returns:
        a dict of metrics
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(data_loader, desc="Evaluating", dynamic_ncols=True):
            waveforms = batch['waveforms'].to(device)
            labels = batch['labels'].to(device)

            if use_tta:
                # TTA on: waveforms come in as [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # Flatten to [B*num_crops, C, T] to run them in one batch
                waveforms_flat = waveforms.view(B * num_crops, C, T)

                # Run all crops through the model
                logits_flat = model(waveforms_flat)

                # Back to [B, num_crops, 2]
                logits_crops = logits_flat.view(B, num_crops, 2)

                # Average the crops
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # Plain inference: waveforms are [B, C, T]
                logits = model(waveforms)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Metrics
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return metrics


# Main entry point
if __name__ == "__main__":
    """
    How to use:
    1. Set json_path and model_path in EvaluationConfig
    2. Run this script
    """

    # Set up the evaluation
    config = EvaluationConfig()

    # Run it
    try:
        evaluate_model(config)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed:")
        print(f"  {str(e)}")
        raise
