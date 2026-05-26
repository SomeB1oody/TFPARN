"""
Main training script for the ASVspoof5 competition.
Handles data loading, training, validation, evaluation, and saving checkpoints.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple
import sys
import time
import csv
import numpy as np

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    create_loss_function, compute_all_metrics,
    print_metrics, print_classification_report_wrapper,
    count_parameters, save_model, EarlyStopping,
    load_model_weights, evaluate_with_calibration,
    compute_cllr
)


# Training config
@dataclass
class ModelArgs:
    """
    All settings for training: data processing, model setup, and hyperparameters
    """
    # Data paths
    train_data_dir: str = ""
    dev_data_dir: str = ""
    eval_data_dir: str = ""

    # Protocol file paths
    train_protocol_dir: str = ""
    dev_protocol_dir: str = ""
    eval_protocol_dir: str = ""

    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True
    batch_size: int = 64
    num_workers: int = 32
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    train_shuffle: bool = True
    use_rawboost: bool = True
    rawboost_prob: float = 0.5

    # TTA Parameters
    use_tta: bool = True
    tta_num_crops: int = 5

    # Model Parameters
    n_mels: int = 160
    n_fft: int = 1024
    hop_length: int = 160
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    model_dropout: float = 0.3
    activation: str = "relu"
    pooling_method: str = "attention"  # one of "mean", "attention", "top-k"
    top_k_ratio: float = 0.3  # for top-k pooling, fraction of frames to keep

    # Training Hyperparameters
    max_epochs: int = 100
    learning_rate: float = 0.667e-4
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"  # 'adam' or 'adamw'
    scheduler_type: str = "cosine"  # 'cosine', 'step', or 'none'
    scheduler_warmup_epochs: int = 5

    # Loss Function ('ce' or 'focal')
    loss_type: str = "focal"
    focal_alpha: float = 0.5  # focal loss alpha (weight for positive class, negative gets 1-alpha)
    focal_gamma: float = 2.0  # focal loss gamma

    # Pairwise AUC/pAUC loss
    enable_pairwise: bool = True  # turn pairwise loss on or off
    pairwise_margin: float = 1.0  # margin for pairwise ranking loss
    pairwise_weight: float = 0.3  # how much the pairwise term counts

    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "min_dcf"  # one of 'f1_macro', 'accuracy', 'recall_macro', 'eer', 'auc_roc', 'min_dcf'
    early_stopping_mode: str = "min"  # 'max' for f1/acc/recall/auc, 'min' for eer/min_dcf

    # Model Checkpoint
    save_dir: str = ""

    # Other
    seed: int = 42


# Training functions
def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, dict, torch.Tensor, torch.Tensor]:
    """
    Run one training epoch.

    Args:
        model: model to train
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: device to train on
        epoch: current epoch number

    Returns:
        (average_loss, metrics)
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        waveforms = batch['waveforms'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(waveforms)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss and predictions
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # Update the progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{loss.item():.5f}', 'avg_loss': f'{avg_loss:.5f}'})

    # Metrics for the whole epoch
    avg_loss = total_loss / len(train_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics, all_logits, all_labels


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_tta: bool = False
) -> Tuple[float, dict, torch.Tensor, torch.Tensor]:
    """
    Run validation.

    Args:
        model: model to validate
        val_loader: validation data loader
        criterion: loss function
        device: device to validate on
        epoch: current epoch number
        use_tta: whether TTA is on (changes the batch shape)

    Returns:
        (average_loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]", dynamic_ncols=True)
        for batch in pbar:
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

            # Loss
            loss = criterion(logits, labels)

            # Track loss and predictions
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            # Update the progress bar
            pbar.set_postfix({'loss': loss.item()})

    # Metrics
    avg_loss = total_loss / len(val_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics, all_logits, all_labels


# Main training loop
def main():
    """
    Main training entry point
    """
    print("\n" + "="*80)
    print("ASVspoof5 TRAINING PIPELINE")
    print("="*80)

    # Set up config
    args = ModelArgs()

    # Set the random seed
    set_seed(args.seed)

    # Pick the device
    device = get_device()
    clear_cuda_cache()

    # Step 1: load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Build the data processing args
    data_args = DataProcessArgs()
    data_args.train_data_dir = args.train_data_dir
    data_args.dev_data_dir = args.dev_data_dir
    data_args.eval_data_dir = args.eval_data_dir
    data_args.train_protocol_dir = args.train_protocol_dir
    data_args.dev_protocol_dir = args.dev_protocol_dir
    data_args.eval_protocol_dir = args.eval_protocol_dir
    data_args.sample_rate = args.sample_rate
    data_args.duration_sec = args.duration_sec
    data_args.mono = args.mono
    data_args.normalize = args.normalize
    data_args.batch_size = args.batch_size
    data_args.num_workers = args.num_workers
    data_args.prefetch_factor = args.prefetch_factor
    data_args.pin_memory = args.pin_memory
    data_args.persistent_workers = args.persistent_workers
    data_args.use_rawboost = args.use_rawboost
    data_args.rawboost_prob = args.rawboost_prob
    data_args.train_shuffle = args.train_shuffle
    data_args.seed = args.seed
    data_args.use_tta = args.use_tta
    data_args.tta_num_crops = args.tta_num_crops

    # Build the loaders
    train_loader, dev_loader, eval_loader = make_loaders(data_args)

    # Step 2: create the model
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)

    # Build the model args
    model_args = SpeechClassifierArgs()
    model_args.n_mels = args.n_mels
    model_args.n_fft = args.n_fft
    model_args.hop_length = args.hop_length
    model_args.sample_rate = args.sample_rate
    model_args.d_model = args.d_model
    model_args.nhead = args.nhead
    model_args.num_layers = args.num_layers
    model_args.dim_feedforward = args.dim_feedforward
    model_args.dropout = args.model_dropout
    model_args.activation = args.activation
    model_args.pooling_method = args.pooling_method
    model_args.top_k_ratio = args.top_k_ratio

    # Create the model
    model = create_model(model_args)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"[✓] Model created with {num_params:,} trainable parameters")

    # Step 3: build the loss function
    print("\n" + "="*80)
    print("STEP 3: CREATING LOSS FUNCTION")
    print("="*80)

    if args.loss_type == "focal":
        # Turn the single alpha into [negative_class, positive_class]
        focal_alpha = torch.tensor([1.0 - args.focal_alpha, args.focal_alpha], dtype=torch.float32)
        criterion = create_loss_function(
            args.loss_type,
            focal_alpha,
            args.focal_gamma,
            enable_pairwise=args.enable_pairwise,
            pairwise_margin=args.pairwise_margin,
            pairwise_weight=args.pairwise_weight
        )
    else:
        criterion = create_loss_function(
            args.loss_type,
            enable_pairwise=args.enable_pairwise,
            pairwise_margin=args.pairwise_margin,
            pairwise_weight=args.pairwise_weight
        )

    criterion = criterion.to(device)

    # Step 4: set up optimizer and scheduler
    print("\n" + "="*80)
    print("STEP 4: CREATING OPTIMIZER AND SCHEDULER")
    print("="*80)

    print(f"  - Optimizer: {args.optimizer_type}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Weight decay: {args.weight_decay}")

    if args.optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")

    # Pick the scheduler
    if args.scheduler_type == "cosine":
        print(f"  - Scheduler: Cosine Annealing")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs - args.scheduler_warmup_epochs,
            eta_min=1e-6
        )
    elif args.scheduler_type == "step":
        print(f"  - Scheduler: Step LR")
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        print(f"  - Scheduler: None")
        scheduler = None

    print(f"[✓] Optimizer and scheduler created")

    # Step 5: training loop
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80)
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Early stopping patience: {args.early_stopping_patience}")
    print(f"  - Early stopping metric: {args.early_stopping_metric}")
    print(f"  - Early stopping mode: {args.early_stopping_mode}")

    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode=args.early_stopping_mode,
        delta=0.0001
    )

    best_metric = -float('inf') if args.early_stopping_mode == 'max' else float('inf')
    best_epoch = 0

    # Where to keep the temporary best model
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    temp_best_path = f"{args.save_dir}/best_model.pt"

    # Timing records start empty
    timing_records = []
    train_run_num = 0
    dev_run_num = 0
    eval_run_num = 0

    for epoch in range(1, args.max_epochs + 1):
        print("\n" + "-"*80)
        print(f"EPOCH {epoch}/{args.max_epochs}")
        print("-"*80)

        # Train
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        train_start_time = time.time()
        train_loss, train_metrics, train_logits, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch
        )
        train_time = time.time() - train_start_time
        train_max_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
        train_run_num += 1

        # Cllr on the train set
        train_probs = torch.softmax(train_logits, dim=1)[:, 1].numpy()
        train_labels_np = train_labels.numpy()
        eps = 1e-10
        train_probs_clipped = np.clip(train_probs, eps, 1 - eps)
        train_cllr = compute_cllr(train_probs_clipped, train_labels_np)

        print(f"\nTrain Loss: {train_loss:.6f}")
        print_metrics(train_metrics, prefix="  [TRAIN] ")

        # Validate (with TTA)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        val_start_time = time.time()
        val_loss, val_metrics, val_logits, val_labels = validate(
            model, dev_loader, criterion, device, epoch, use_tta=True
        )
        val_time = time.time() - val_start_time
        val_max_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
        dev_run_num += 1

        # Cllr on the validation set
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].numpy()
        val_labels_np = val_labels.numpy()
        val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
        val_cllr = compute_cllr(val_probs_clipped, val_labels_np)

        print(f"\nValidation Loss: {val_loss:.6f}")
        print_metrics(val_metrics, prefix="  [VAL] ")

        # Update the learning rate, with warmup first
        if scheduler is not None:
            if epoch <= args.scheduler_warmup_epochs:
                # Warmup: ramp the LR up from 0 to the target
                warmup_lr = args.learning_rate * epoch / args.scheduler_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"\nLearning rate (warmup): {warmup_lr:.8f}")
            else:
                # Past warmup, let the scheduler take over
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nLearning rate: {current_lr:.8f}")

        # See if this epoch improved the metric
        current_metric = val_metrics[args.early_stopping_metric]

        is_better = False
        if args.early_stopping_mode == 'max':
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                is_better = True
        else:
            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                is_better = True

        if is_better:
            print(f"\n[✓] New best {args.early_stopping_metric}: {best_metric:.4f}")
            # Save it as the temporary best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_metric,
                'metric_name': args.early_stopping_metric
            }, temp_best_path)
            print(f"  [✓] Saved temporary best model to {temp_best_path}")

        # Save the timing for this epoch
        timing_records.append({
            'run_num': train_run_num,
            'type': 1,
            'time': round(train_time, 1),
            'max_memory': round(train_max_memory / 1024, 1),
            'if_best': 1 if is_better else 0,
            'eer': round(train_metrics.get('eer', ''), 6),
            'min_dcf': round(train_metrics.get('min_dcf', ''), 6),
            'cllr': round(train_cllr, 6),
            'act_dcf': round(train_metrics.get('act_dcf', ''), 6)
        })
        timing_records.append({
            'run_num': dev_run_num,
            'type': 2,
            'time': round(val_time, 1),
            'max_memory': round(val_max_memory / 1024, 1),
            'if_best': 1 if is_better else 0,
            'eer': round(val_metrics.get('eer', ''), 6),
            'min_dcf': round(val_metrics.get('min_dcf', ''), 6),
            'cllr': round(val_cllr, 6),
            'act_dcf': round(val_metrics.get('act_dcf', ''), 6)
        })

        # Stop early if it's time
        if early_stopping(current_metric):
            print(f"\n[!] Early stopping triggered at epoch {epoch}")
            print(f"  Best {args.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
            break

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"  - Total epochs: {epoch}")
    print(f"  - Best epoch: {best_epoch}")
    print(f"  - Best {args.early_stopping_metric}: {best_metric:.4f}")

    # Step 6: load the best model and evaluate on the test set
    print("\n" + "="*80)
    print("STEP 6: LOAD BEST MODEL AND EVALUATE ON TEST SET")
    print("="*80)

    # Load the best model back from the temporary checkpoint
    model = load_model_weights(model, temp_best_path, device, strict=True)

    # Rebuild the loaders with TTA on for the final evaluation
    print("\n[INFO] Recreating data loaders with TTA enabled for final evaluation")
    data_args.use_tta = True
    _, dev_loader_tta, eval_loader_tta = make_loaders(data_args)

    # Evaluate on raw (uncalibrated) scores with the TTA loaders
    # Calibration and prior correction are off, so the scores are raw logits
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    eval_start_time = time.time()
    results = evaluate_with_calibration(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader_tta,
        eval_loader=eval_loader_tta,
        device=device,
        apply_calibration=False,
        enable_prior_correction=False
    )
    eval_time = time.time() - eval_start_time
    eval_max_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    eval_run_num += 1

    # Pull out the results
    final_train_metrics = results['train']['initial_metrics']
    final_val_metrics = results['dev']['initial_metrics']
    eval_metrics = results['eval']['initial_metrics']

    # Save the final evaluation timing
    timing_records.append({
        'run_num': eval_run_num,
        'type': 3,
        'time': round(eval_time, 1),
        'max_memory': round(eval_max_memory / 1024, 1),
        'if_best': 1,
        'eer': round(eval_metrics.get('eer', ''), 6),
        'min_dcf': round(eval_metrics.get('min_dcf', ''), 6),
        'cllr': round(eval_metrics.get('cllr', ''), 6),
        'act_dcf': round(eval_metrics.get('act_dcf', ''), 6)
    })

    # Print the final evaluation results
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)

    # Metrics without calibration
    print("\n" + "-"*80)
    print("INITIAL METRICS (No Calibration)")
    print("-"*80)
    print_metrics(final_train_metrics, prefix="  [TRAIN] ")
    print_metrics(final_val_metrics, prefix="  [VAL] ")
    print_metrics(eval_metrics, prefix="  [TEST] ")

    # Classification report from the raw (uncalibrated) scores
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT (Initial)")
    print("-"*80)
    print_classification_report_wrapper(
        results['eval']['logits'],
        results['eval']['labels'],
        target_names=['spoof (AI)', 'bonafide (human)']
    )

    # Step 7: save the model
    print("\n" + "="*80)
    print("STEP 7: SAVE MODEL")
    print("="*80)

    # Name the model after the metric and its value
    model_name = f"best_model_{args.early_stopping_metric}_{best_metric:.4f}"

    # Put it in a directory with the same name
    model_dir = os.path.join(args.save_dir, model_name)

    print(f"\nAuto-saving model to directory: {model_dir}")

    # Save everything
    save_model(
        model_save_dir=model_dir,
        model_name=model_name,
        model=model,
        optimizer=optimizer,
        data_args=data_args,
        model_args=model_args,
        train_args=args,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        test_metrics=eval_metrics
    )

    print(f"\n[✓] Model saved successfully!")
    print(f"  - Directory: {model_dir}")
    print(f"  - Model weights: {model_name}.pt")
    print(f"  - Configuration & metrics: {model_name}.json")
    print(f"  - Metric: {args.early_stopping_metric}")
    print(f"  - Value on Dev set: {best_metric:.4f}")

    # Write the timing records into the final model directory
    timing_csv_path = os.path.join(model_dir, "timing_records.csv")
    with open(timing_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['run_num', 'type', 'time', 'max_memory', 'if_best', 'eer', 'min_dcf', 'cllr', 'act_dcf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_records)

    print(f"\n[✓] Timing records saved successfully!")
    print(f"  - Timing CSV: {timing_csv_path}")
    print(f"  - Total train runs: {train_run_num}")
    print(f"  - Total dev runs: {dev_run_num}")
    print(f"  - Total eval runs: {eval_run_num}")

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[!] Training failed with error:")
        print(f"  {str(e)}")
        raise
