"""
Training script for RawNet2 on ASVspoof5.
Covers config, data loading, the training loop, validation, and evaluation.
"""

import os
import json
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Dict, Tuple

from data_process import make_loaders, get_class_weights
from model import RawNet2Args, build_model
from utils import (
    set_seed, get_device, clear_cuda_cache,
    compute_all_metrics, print_metrics, count_parameters,
    get_loss_function, compute_cllr_from_logits
)


@dataclass
class ModelArgs:
    """
    All the training config in one place: data processing, model setup, and hyperparameters
    """
    # ========== Data Paths ==========
    train_data_dir: str = ""
    dev_data_dir: str = ""
    eval_data_dir: str = ""

    train_protocol_dir: str = ""
    dev_protocol_dir: str = ""
    eval_protocol_dir: str = ""

    # ========== Data Processing Parameters ==========
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

    # RawBoost augmentation
    use_rawboost: bool = True
    rawboost_prob: float = 0.5

    # Test-Time Augmentation (TTA)
    use_tta: bool = True
    tta_num_crops: int = 5

    # ========== Model Architecture Parameters ==========
    # RawNet2 parameters (defaults from the paper)
    sinc_out_channels: int = 20
    sinc_kernel_size: int = 1024
    gru_node: int = 1024
    nb_gru_layer: int = 3
    nb_fc_node: int = 1024
    nb_classes: int = 2

    # ========== Training Hyperparameters ==========
    max_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    optimizer_type: str = "adamw"  # 'adam' or 'adamw'
    scheduler_type: str = "cosine"  # 'cosine', 'step', or 'none'
    scheduler_warmup_epochs: int = 5

    # Loss Function
    loss_type: str = "ce"  # 'ce' or 'focal'
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0

    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "min_dcf"  # 'eer', 'f1_macro', 'accuracy', 'min_dcf'
    early_stopping_mode: str = "min"  # 'min' or 'max'

    # ========== Model Checkpoint ==========
    save_dir: str = ""

    # ========== Misc ==========
    seed: int = 42

    def get_rawnet2_args(self) -> RawNet2Args:
        """Build the RawNet2Args used to create the model"""
        return RawNet2Args(
            sinc_out_channels=self.sinc_out_channels,
            sinc_kernel_size=self.sinc_kernel_size,
            gru_node=self.gru_node,
            nb_gru_layer=self.nb_gru_layer,
            nb_fc_node=self.nb_fc_node,
            nb_classes=self.nb_classes
        )


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    use_tta: bool = False
) -> Tuple[float, Dict[str, float], float, float]:
    """
    Run one training epoch.

    Args:
        model: RawNet2 model
        dataloader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: torch device
        epoch: current epoch number
        use_tta: whether TTA is on (ignored during training)

    Returns:
        (average_loss, metrics_dict, elapsed_time, max_memory_gb)
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # start the timer and reset memory tracking
    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", ncols=100)

    for batch_idx, batch in enumerate(pbar):
        waveforms = batch['waveforms'].to(device)
        labels = batch['labels'].to(device)

        # forward
        optimizer.zero_grad()
        embeddings, logits = model(waveforms, is_test=False)

        loss = criterion(logits, labels)

        # backward
        loss.backward()
        optimizer.step()

        # keep running totals
        total_loss += loss.item()
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    elapsed_time = time.time() - start_time

    # peak memory used this epoch
    if device.type == 'cuda':
        max_memory_bytes = torch.cuda.max_memory_allocated(device)
        max_memory_gb = max_memory_bytes / (1024 * 1024 * 1024)
    else:
        max_memory_gb = 0.0

    avg_loss = total_loss / len(dataloader)

    # metrics over the whole epoch (cllr included)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics, elapsed_time, max_memory_gb


def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    split_name: str = "Dev",
    use_tta: bool = False
) -> Tuple[float, Dict[str, float], float, float]:
    """
    Run the model on the dev or eval set.

    Args:
        model: RawNet2 model
        dataloader: validation data loader
        criterion: loss function
        device: torch device
        split_name: name of the split ('Dev' or 'Eval')
        use_tta: whether to use test-time augmentation

    Returns:
        (average_loss, metrics_dict, elapsed_time, max_memory_gb)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # start the timer and reset memory tracking
    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(dataloader, desc=f"{split_name} Evaluation", ncols=100)

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

    elapsed_time = time.time() - start_time

    # peak memory used during this pass
    if device.type == 'cuda':
        max_memory_bytes = torch.cuda.max_memory_allocated(device)
        max_memory_gb = max_memory_bytes / (1024 * 1024 * 1024)
    else:
        max_memory_gb = 0.0

    avg_loss = total_loss / len(dataloader)

    # metrics over all batches (cllr included)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics, elapsed_time, max_memory_gb


def save_timing_records(csv_path: str, timing_records: list):
    """
    Write all the timing records to a CSV file in one go.

    Args:
        csv_path: path to the CSV file
        timing_records: list of tuples
            (run_num, dataset_type, elapsed_time, max_memory_gb, is_best, metrics_dict or None)
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # header row
        writer.writerow(['run_num', 'type', 'time', 'max_memory', 'if_best',
                         'eer', 'min_dcf', 'cllr', 'act_dcf'])

        # data rows
        for run_num, dataset_type, elapsed_time, max_memory_gb, is_best, metrics in timing_records:
            writer.writerow([run_num, dataset_type, f'{elapsed_time:.1f}',
                             f'{max_memory_gb:.1f}', is_best,
                             f"{metrics.get('eer', ''):.4f}" if metrics.get('eer') is not None else '',
                             f"{metrics.get('min_dcf', ''):.4f}" if metrics.get('min_dcf') is not None else '',
                             f"{metrics.get('cllr', ''):.4f}" if metrics.get('cllr') is not None else '',
                             f"{metrics.get('act_dcf', ''):.4f}" if metrics.get('act_dcf') is not None else ''])


def save_checkpoint(
    model: nn.Module,
    args: ModelArgs,
    metrics: Dict[str, Dict[str, float]],
    save_dir: str,
    filename: str = "model.pt"
):
    """
    Save the model checkpoint along with its config.

    Args:
        model: trained model
        args: ModelArgs config
        metrics: metrics for train/dev/eval
        save_dir: directory to save into
        filename: checkpoint filename
    """
    os.makedirs(save_dir, exist_ok=True)

    # save the weights
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"\n[Checkpoint] Model saved to: {model_path}")

    # save config and metrics as JSON
    config_path = os.path.join(save_dir, filename.replace('.pt', '_config.json'))
    save_dict = {
        'model_args': asdict(args),
        'metrics': metrics
    }

    with open(config_path, 'w') as f:
        json.dump(save_dict, f, indent=4)
    print(f"[Checkpoint] Configuration and metrics saved to: {config_path}")


def main():
    """Run the whole training pipeline"""
    print("\n" + "="*80)
    print("RAWNET2 TRAINING PIPELINE FOR ASVSPOOF5")
    print("="*80)

    # ========== Configuration ==========
    args = ModelArgs()

    # fix the random seed
    set_seed(args.seed)

    # pick a device
    device = get_device()
    clear_cuda_cache()

    # ========== Data Loading ==========
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)

    train_loader, dev_loader, eval_loader = make_loaders(args)

    # work out class weights for the loss
    from data_process import read_protocol
    train_items = read_protocol(args.train_protocol_dir)
    class_weights = get_class_weights(train_items)

    # ========== Model Initialization ==========
    print("\n" + "="*80)
    print("STEP 2: MODEL INITIALIZATION")
    print("="*80)

    rawnet2_args = args.get_rawnet2_args()
    model = build_model(rawnet2_args, device)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"[Model] RawNet2 initialized")
    print(f"[Model] Total trainable parameters: {num_params:,}")

    # ========== Loss Function ==========
    print("\n[Loss] Initializing loss function: {args.loss_type}")
    criterion = get_loss_function(
        loss_type=args.loss_type,
        class_weights=class_weights,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        device=device
    )
    print(f"[Loss] Loss function initialized: {args.loss_type}")

    # ========== Optimizer ==========
    print(f"\n[Optimizer] Initializing optimizer: {args.optimizer_type}")
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
    print(f"[Optimizer] Optimizer initialized: {args.optimizer_type} (lr={args.learning_rate})")

    # ========== Learning Rate Scheduler ==========
    print(f"\n[Scheduler] Initializing scheduler: {args.scheduler_type}")
    if args.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs - args.scheduler_warmup_epochs,
            eta_min=1e-6
        )
    elif args.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5
        )
    elif args.scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")

    if scheduler is not None:
        print(f"[Scheduler] Scheduler initialized: {args.scheduler_type}")
    else:
        print(f"[Scheduler] No scheduler used")

    # ========== Training Loop ==========
    print("\n" + "="*80)
    print("STEP 3: TRAINING")
    print("="*80)
    print(f"Training configuration:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Early stopping patience: {args.early_stopping_patience}")
    print(f"  - Early stopping metric: {args.early_stopping_metric} ({args.early_stopping_mode})")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print("="*80 + "\n")

    best_metric = float('inf') if args.early_stopping_mode == 'min' else float('-inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None  # keep the best weights here

    history = {
        'train_loss': [],
        'dev_loss': [],
        'train_metrics': [],
        'dev_metrics': []
    }

    # collect timing records along the way
    timing_records = []

    # run counters, one per dataset type
    train_run_num = 0
    dev_run_num = 0
    eval_run_num = 0

    for epoch in range(1, args.max_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.max_epochs}")
        print(f"{'='*80}")

        # train
        train_loss, train_metrics, train_time, train_memory = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.use_tta
        )

        # log train time and memory
        train_run_num += 1
        timing_records.append((train_run_num, 1, train_time, train_memory, 0, train_metrics))

        # check the dev set
        dev_loss, dev_metrics, dev_time, dev_memory = validate(
            model, dev_loader, criterion, device, "Dev", args.use_tta
        )

        dev_run_num += 1

        # update history
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        history['train_metrics'].append(train_metrics)
        history['dev_metrics'].append(dev_metrics)

        print(f"\n[Train] Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="[Train] ")

        print(f"\n[Dev] Loss: {dev_loss:.4f}")
        print_metrics(dev_metrics, prefix="[Dev] ")

        # update the learning rate: warm up first, then cosine annealing
        if scheduler is not None:
            if epoch <= args.scheduler_warmup_epochs:
                # warmup: ramp the LR up to the base value
                warmup_lr = args.learning_rate * epoch / args.scheduler_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"\n[Scheduler] Learning rate (warmup): {warmup_lr:.6f}")
            else:
                scheduler.step()
                print(f"\n[Scheduler] Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # early stopping check
        current_metric = dev_metrics[args.early_stopping_metric]

        if args.early_stopping_mode == 'min':
            is_better = current_metric < best_metric
        else:
            is_better = current_metric > best_metric

        # log dev time and memory, plus the best flag and metrics
        timing_records.append((dev_run_num, 2, dev_time, dev_memory, 1 if is_better else 0, dev_metrics))

        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0

            # remember the best weights
            best_model_state = model.state_dict().copy()

            print(f"\n[Early Stopping] New best {args.early_stopping_metric}: {best_metric:.4f} at epoch {epoch}")
        else:
            patience_counter += 1
            print(f"\n[Early Stopping] No improvement for {patience_counter} epoch(s)")
            print(f"[Early Stopping] Best {args.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")

            if patience_counter >= args.early_stopping_patience:
                print(f"\n[Early Stopping] Patience exhausted. Stopping training.")
                break

    # done training
    print(f"\n[Training] Training completed at epoch {epoch}")
    print(f"[Training] Best model from epoch {best_epoch} with {args.early_stopping_metric}={best_metric:.4f}")

    # ========== Final Evaluation on Eval Set ==========
    print("\n" + "="*80)
    print("STEP 4: FINAL EVALUATION ON EVAL SET")
    print("="*80)

    # load the best weights back
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Evaluation] Loaded best model from epoch {best_epoch}")
    else:
        print(f"[Evaluation] Warning: No best model saved, using current model state")

    # run the eval set
    eval_loss, eval_metrics, eval_time, eval_memory = validate(
        model, eval_loader, criterion, device, "Eval", args.use_tta
    )

    # log eval time, memory, and metrics
    eval_run_num += 1
    timing_records.append((eval_run_num, 3, eval_time, eval_memory, 0, eval_metrics))

    print(f"\n[Eval] Loss: {eval_loss:.4f}")
    print_metrics(eval_metrics, prefix="[Eval] ")

    # ========== Save Final Results ==========
    print("\n" + "="*80)
    print("STEP 5: SAVING FINAL RESULTS")
    print("="*80)

    # pull the metrics from the best epoch
    best_train_metrics = history['train_metrics'][best_epoch - 1]
    best_dev_metrics = history['dev_metrics'][best_epoch - 1]

    final_metrics = {
        'train': best_train_metrics,
        'dev': best_dev_metrics,
        'eval': eval_metrics,
        'best_epoch': best_epoch
    }

    # name the folder after the metric and its value
    model_dir_name = f"best_model_{args.early_stopping_metric}_{best_metric:.4f}"
    save_dir = os.path.join(args.save_dir, model_dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # save the final checkpoint with all metrics
    save_checkpoint(
        model, args, final_metrics, save_dir, "best_model.pt"
    )

    # write the timing CSV next to the model
    timing_csv_path = os.path.join(save_dir, "timing.csv")
    save_timing_records(timing_csv_path, timing_records)
    print(f"[Timing] Timing records saved to: {timing_csv_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Model directory: {save_dir}")
    print("\nFinal Metrics Summary:")
    print("-" * 80)
    print("\n[Train - Best Epoch]")
    print_metrics(best_train_metrics, prefix="  ")
    print("\n[Dev - Best Epoch]")
    print_metrics(best_dev_metrics, prefix="  ")
    print("\n[Eval - Final]")
    print_metrics(eval_metrics, prefix="  ")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
