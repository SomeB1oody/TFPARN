"""
Evaluation script for RawNet2 on ASVspoof5.
Loads a saved model and its config, then runs it on the dev and eval sets.
"""

import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Tuple

from data_process import make_loaders
from model import RawNet2Args, build_model
from utils import (
    set_seed, get_device,
    compute_all_metrics, print_metrics,
    get_loss_function, compute_cllr_from_logits
)


@dataclass
class EvalConfig:
    """
    Config for evaluation
    """
    # paths to the model and its config
    model_path: str = ""
    config_path: str = ""

    # optional path overrides (leave as None to use the paths from the config)
    dev_data_dir: str = None
    eval_data_dir: str = None
    dev_protocol_dir: str = None
    eval_protocol_dir: str = None


def load_model_and_config(model_path: str, config_path: str, device):
    """
    Load a trained model and its config.

    Args:
        model_path: path to the model checkpoint (.pt file)
        config_path: path to the config JSON file
        device: torch device

    Returns:
        (model, args_dict): the loaded model and its config dict
    """
    print("\n" + "="*80)
    print("LOADING MODEL AND CONFIGURATION")
    print("="*80)

    # make sure the files are there
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # load the config
    print(f"[Config] Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    model_args_dict = config_dict['model_args']
    print(f"[Config] Configuration loaded successfully")

    # rebuild RawNet2Args from the saved config
    rawnet2_args = RawNet2Args(
        sinc_out_channels=model_args_dict['sinc_out_channels'],
        sinc_kernel_size=model_args_dict['sinc_kernel_size'],
        gru_node=model_args_dict['gru_node'],
        nb_gru_layer=model_args_dict['nb_gru_layer'],
        nb_fc_node=model_args_dict['nb_fc_node'],
        nb_classes=model_args_dict['nb_classes']
    )

    # build the model
    print(f"\n[Model] Building RawNet2 model")
    model = build_model(rawnet2_args, device)

    # load the weights
    print(f"[Model] Loading model weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"[Model] Model loaded successfully")
    print("="*80 + "\n")

    return model, model_args_dict


def evaluate_model(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    split_name: str = "Dev",
    use_tta: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Run the model on a dataset.

    Args:
        model: RawNet2 model
        dataloader: data loader to evaluate on
        criterion: loss function
        device: torch device
        split_name: name of the split ('Dev' or 'Eval')
        use_tta: whether to use test-time augmentation

    Returns:
        (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Evaluating {split_name}", ncols=100)

    with torch.no_grad():
        for batch in pbar:
            labels = batch['labels'].to(device)

            if use_tta:
                # TTA: several crops per sample
                waveforms = batch['waveforms'].to(device)  # [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # flatten crops into the batch dim
                waveforms_flat = waveforms.view(B * num_crops, C, T)

                _, logits_flat = model(waveforms_flat, is_test=False)

                # split crops back out
                logits_crops = logits_flat.view(B, num_crops, -1)

                # average over the crops
                logits = logits_crops.mean(dim=1)  # [B, num_classes]
            else:
                # one sample at a time
                waveforms = batch['waveforms'].to(device)
                _, logits = model(waveforms, is_test=False)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

    avg_loss = total_loss / len(dataloader)

    # metrics over all batches (cllr included)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics


def main():
    """Run the evaluation pipeline"""
    print("\n" + "="*80)
    print("RAWNET2 EVALUATION FOR ASVSPOOF5")
    print("="*80)

    # ========== Configuration ==========
    eval_config = EvalConfig()

    # fix the random seed (use the saved config's seed if it has one)
    set_seed(42)

    # pick a device
    device = get_device()

    # ========== Load Model ==========
    model, model_args_dict = load_model_and_config(
        eval_config.model_path,
        eval_config.config_path,
        device
    )

    # ========== Prepare Arguments for Data Loading ==========
    # a small object to hold the args the data loader expects
    class DataArgs:
        pass

    args = DataArgs()

    # start from the saved config
    for key, value in model_args_dict.items():
        setattr(args, key, value)

    # then apply any path overrides from eval_config
    if eval_config.dev_data_dir is not None:
        args.dev_data_dir = eval_config.dev_data_dir
    if eval_config.eval_data_dir is not None:
        args.eval_data_dir = eval_config.eval_data_dir
    if eval_config.dev_protocol_dir is not None:
        args.dev_protocol_dir = eval_config.dev_protocol_dir
    if eval_config.eval_protocol_dir is not None:
        args.eval_protocol_dir = eval_config.eval_protocol_dir

    # ========== Data Loading ==========
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # only the dev and eval loaders are needed here
    _, dev_loader, eval_loader = make_loaders(args)

    # ========== Loss Function ==========
    print("\n[Loss] Initializing loss function")
    criterion = get_loss_function(
        loss_type=args.loss_type,
        class_weights=None,  # not needed when only evaluating
        focal_alpha=args.focal_alpha if hasattr(args, 'focal_alpha') else 0.25,
        focal_gamma=args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0,
        device=device
    )

    # ========== Evaluation on Dev Set ==========
    print("\n" + "="*80)
    print("EVALUATING ON DEV SET")
    print("="*80)

    dev_loss, dev_metrics = evaluate_model(
        model, dev_loader, criterion, device, "Dev",
        use_tta=args.use_tta if hasattr(args, 'use_tta') else False
    )

    print(f"\n[Dev] Loss: {dev_loss:.4f}")
    print_metrics(dev_metrics, prefix="[Dev] ")

    # ========== Evaluation on Eval Set ==========
    print("\n" + "="*80)
    print("EVALUATING ON EVAL SET")
    print("="*80)

    eval_loss, eval_metrics = evaluate_model(
        model, eval_loader, criterion, device, "Eval",
        use_tta=args.use_tta if hasattr(args, 'use_tta') else False
    )

    print(f"\n[Eval] Loss: {eval_loss:.4f}")
    print_metrics(eval_metrics, prefix="[Eval] ")

    # ========== Summary ==========
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*80)

    print("\n[Dev Set Results]")
    print("-" * 80)
    print_metrics(dev_metrics, prefix="  ")

    print("\n[Eval Set Results]")
    print("-" * 80)
    print_metrics(eval_metrics, prefix="  ")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
