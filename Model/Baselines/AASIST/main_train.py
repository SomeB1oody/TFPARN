"""
Training script for AASIST on ASVspoof5.
Handles config, data loading, training, validation, evaluation, and saving the model.
"""

import os
import json
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Dict

# Import custom modules
from data_process import make_loaders, get_class_weights
from model import create_aasist_model, AASISTArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    create_loss_function, compute_all_metrics, print_metrics,
    EarlyStopping, count_parameters
)


# ============================================================================
# Helper Functions
# ============================================================================

def safe_empty_cache():
    """Empty the CUDA cache if a GPU is available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class ModelArgs:
    """
    All settings for a training run: data processing, model architecture, and
    training hyperparameters
    """
    # ===== Data Processing Parameters =====
    # Data paths
    train_data_dir: str = ""
    dev_data_dir: str = ""
    eval_data_dir: str = ""

    # Protocol file paths
    train_protocol_dir: str = ""
    dev_protocol_dir: str = ""
    eval_protocol_dir: str = ""

    # Audio processing
    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True

    # DataLoader settings
    batch_size: int = 64
    num_workers: int = 32
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    train_shuffle: bool = True

    # Augmentation settings
    use_rawboost: bool = True  # RawBoost augmentation for training
    rawboost_prob: float = 0.5

    # Test-Time Augmentation
    use_tta: bool = True  # Enable TTA for dev/eval
    tta_num_crops: int = 5  # Number of crops per sample for TTA

    # ===== AASIST Model Parameters =====
    # Number of samples
    nb_samp: int = 64600

    # First conv kernel size
    first_conv: int = 128

    # Filter sizes for the residual blocks
    # filts[0] is the Sinc conv output channels (an integer)
    # filts[1:] are [input, output] for each residual block
    filts: list = None  # defaults to [70, [1, 32], [32, 32], [32, 64], [64, 64]]

    # GAT dimensions
    gat_dims: list = None  # defaults to [64, 32]

    # Pooling ratios
    pool_ratios: list = None  # defaults to [0.5, 0.7, 0.5, 0.5]

    # Attention temperatures
    temperatures: list = None  # defaults to [2.0, 2.0, 100.0, 100.0]

    # Number of output classes
    num_classes: int = 2

    # ===== Training Hyperparameters =====
    max_epochs: int = 100
    learning_rate: float = 2.667e-4
    weight_decay: float = 1e-2

    # Optimizer
    optimizer_type: str = "adamw"  # 'adam' or 'adamw'
    adam_betas: tuple = (0.9, 0.999)  # Adam optimizer betas
    amsgrad: bool = False  # AMSGrad variant of Adam

    # Learning rate scheduler
    scheduler_type: str = "cosine"  # 'cosine', 'step', or 'none'
    scheduler_warmup_epochs: int = 5
    scheduler_step_size: int = 10  # For step scheduler
    scheduler_gamma: float = 0.5   # For step scheduler
    lr_min: float = 1e-6  # Minimum learning rate for scheduler

    # Loss function
    loss_type: str = "ce"  # 'ce' or 'focal'
    focal_gamma: float = 2.0  # Gamma for focal loss
    enable_pairwise: bool = False  # Enable pairwise ranking loss
    pairwise_margin: float = 1.0
    pairwise_weight: float = 0.1

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "min_dcf"  # 'eer', 'min_dcf', 'f1_macro', 'accuracy'
    early_stopping_mode: str = "min"  # 'min' for eer/min_dcf, 'max' for f1/accuracy

    # Model checkpoint
    save_dir: str = ""

    # Other
    seed: int = 42
    freq_aug: bool = False  # Enable frequency augmentation during training

    def __post_init__(self):
        """Fill in the default list values"""
        if self.filts is None:
            # filts[0] is the Sinc conv output channels
            # filts[1:] are [input, output] pairs for the residual blocks
            self.filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        if self.gat_dims is None:
            self.gat_dims = [64, 32]
        if self.pool_ratios is None:
            self.pool_ratios = [0.5, 0.7, 0.5, 0.5]
        if self.temperatures is None:
            self.temperatures = [2.0, 2.0, 100.0, 100.0]


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    freq_aug: bool = False
) -> tuple[Dict[str, float], float]:
    """
    Run one training epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        freq_aug: Enable frequency augmentation

    Returns:
        (metrics dict, peak memory in MB)
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar):
        waveforms = batch['waveforms'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(waveforms, Freq_aug=freq_aug)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

    # Get peak memory usage
    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_bytes = torch.cuda.max_memory_allocated(device)
        max_memory_mb = max_memory_bytes / (1024 ** 2)  # bytes to MB

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_all_metrics(all_logits, all_labels)
    metrics['loss'] = avg_loss

    # Free memory before returning
    del all_logits, all_labels
    safe_empty_cache()

    return metrics, max_memory_mb


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = False,
    desc: str = "Validation"
) -> tuple[Dict[str, float], float]:
    """
    Run the model over a validation set.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
        use_tta: Whether TTA is enabled
        desc: Description for progress bar

    Returns:
        (metrics dict, peak memory in MB)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=desc, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            waveforms = batch['waveforms'].to(device)
            labels = batch['labels'].to(device)

            if use_tta:
                # With TTA, waveforms are [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape
                waveforms_flat = waveforms.view(B * num_crops, C, T)

                # Run all crops through the model
                logits_flat = model(waveforms_flat)

                # Average the logits over the crops
                logits_crops = logits_flat.view(B, num_crops, -1)
                logits = logits_crops.mean(dim=1)
            else:
                logits = model(waveforms)

            # Compute loss
            loss = criterion(logits, labels)

            # Accumulate
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix({"loss": loss.item()})

    # Get peak memory usage
    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_bytes = torch.cuda.max_memory_allocated(device)
        max_memory_mb = max_memory_bytes / (1024 ** 2)  # bytes to MB

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_all_metrics(all_logits, all_labels)
    metrics['loss'] = avg_loss

    # Free memory before returning
    del all_logits, all_labels
    safe_empty_cache()

    return metrics, max_memory_mb


def save_checkpoint(
    save_dir: str,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: ModelArgs,
    train_metrics: Dict[str, float],
    dev_metrics: Dict[str, float],
    eval_metrics: Dict[str, float],
    epoch: int,
    timing_records: list = None
) -> None:
    """
    Save the model checkpoint, the config JSON, and the timing records CSV.

    Args:
        save_dir: Directory to save files
        model_name: Name for the model files (also used as subdirectory name)
        model: PyTorch model
        optimizer: Optimizer
        args: ModelArgs configuration
        train_metrics: Training metrics
        dev_metrics: Development metrics
        eval_metrics: Evaluation metrics
        epoch: Current epoch
        timing_records: List of timing records to save
    """
    # Put everything in a subdirectory named after the model
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    args_dict = asdict(args)

    save_data = {
        'args': args_dict,
        'epoch': epoch,
        'metrics': {
            'train': train_metrics,
            'dev': dev_metrics,
            'eval': eval_metrics
        }
    }

    # Turn numpy/torch types into plain Python types so JSON can handle them
    def convert_types(obj):
        """Recursively convert numpy/torch types to plain Python types"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        else:
            return obj

    save_data = convert_types(save_data)

    # Save the config JSON
    json_path = os.path.join(model_dir, f"{model_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
    print(f"[Checkpoint] Configuration saved to {json_path}")

    # Save model weights
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'dev_metrics': dev_metrics,
        'eval_metrics': eval_metrics
    }, model_path)
    print(f"[Checkpoint] Model weights saved to {model_path}")

    # Save timing records if there are any
    if timing_records is not None:
        timing_csv_path = os.path.join(model_dir, "timing_records.csv")
        if len(timing_records) > 0:
            fieldnames = list(timing_records[0].keys())
        else:
            fieldnames = ['run_num', 'type', 'time', 'max_memory', 'if_best',
                          'eer', 'min_dcf', 'cllr', 'act_dcf']
        with open(timing_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timing_records)
        print(f"[Checkpoint] Timing records saved to {timing_csv_path}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Run the full training pipeline"""
    print("\n" + "="*80)
    print("AASIST TRAINING FOR ASVSPOOF5")
    print("="*80)

    # Initialize configuration
    args = ModelArgs()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # ===== Step 1: Load Data =====
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    train_loader, dev_loader, eval_loader = make_loaders(args)

    # Class weights for the loss function
    from data_process import read_protocol
    train_items = read_protocol(args.train_protocol_dir)
    class_weights = get_class_weights(train_items)

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

    # Count parameters
    num_params = count_parameters(model)
    print(f"\n[Model] Total trainable parameters: {num_params:,}")

    # ===== Step 3: Create Loss Function =====
    print("\n" + "="*80)
    print("STEP 3: CREATING LOSS FUNCTION")
    print("="*80)

    criterion = create_loss_function(
        loss_type=args.loss_type,
        focal_alpha=class_weights if args.loss_type == 'focal' else None,
        focal_gamma=args.focal_gamma,
        enable_pairwise=args.enable_pairwise,
        pairwise_margin=args.pairwise_margin,
        pairwise_weight=args.pairwise_weight
    )
    criterion = criterion.to(device)

    # ===== Step 4: Create Optimizer =====
    print("\n" + "="*80)
    print("STEP 4: CREATING OPTIMIZER AND SCHEDULER")
    print("="*80)

    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
        print(f"[Optimizer] Adam (lr={args.learning_rate}, betas={args.adam_betas}, weight_decay={args.weight_decay}, amsgrad={args.amsgrad})")
    elif args.optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
        print(f"[Optimizer] AdamW (lr={args.learning_rate}, betas={args.adam_betas}, weight_decay={args.weight_decay}, amsgrad={args.amsgrad})")
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")

    # Set up the learning rate scheduler
    if args.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs - args.scheduler_warmup_epochs,
            eta_min=args.lr_min
        )
        print(f"[Scheduler] CosineAnnealingLR (warmup_epochs={args.scheduler_warmup_epochs}, lr_min={args.lr_min})")
    elif args.scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
        print(f"[Scheduler] StepLR (step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma})")
    else:
        scheduler = None
        print(f"[Scheduler] None")

    # ===== Step 5: Training Loop =====
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80)
    print(f"[Training] Max epochs: {args.max_epochs}")
    print(f"[Training] Early stopping: {args.early_stopping_metric} ({args.early_stopping_mode}, patience={args.early_stopping_patience})")

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode=args.early_stopping_mode,
        delta=0.0
    )

    # Track the best metrics (eval runs once at the end)
    best_epoch = 0
    best_dev_metrics = None
    best_train_metrics = None

    # Run counters per dataset type
    train_run_num = 0
    dev_run_num = 0
    eval_run_num = 0

    timing_records = []

    # Path for the temporary best model
    os.makedirs(args.save_dir, exist_ok=True)
    temp_best_path = os.path.join(args.save_dir, "temp_best_model.pt")

    # Training loop
    for epoch in range(1, args.max_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.max_epochs}")
        print(f"{'='*80}")

        # Train with timing
        train_run_num += 1
        train_start_time = time.time()
        train_metrics, train_max_memory = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, freq_aug=args.freq_aug
        )
        train_elapsed_time = time.time() - train_start_time

        print(f"\n[Train Results]")
        print_metrics(train_metrics, prefix="  ")
        print(f"  Time: {train_elapsed_time:.1f}s")
        print(f"  Max Memory: {train_max_memory:.1f} MB")

        # Clear GPU cache before validation
        safe_empty_cache()

        # Validate on dev set with timing
        dev_run_num += 1
        dev_start_time = time.time()
        dev_metrics, dev_max_memory = validate(
            model, dev_loader, criterion, device,
            use_tta=args.use_tta, desc=f"Epoch {epoch} [Dev]"
        )
        dev_elapsed_time = time.time() - dev_start_time

        print(f"\n[Dev Results]")
        print_metrics(dev_metrics, prefix="  ")
        print(f"  Time: {dev_elapsed_time:.1f}s")
        print(f"  Max Memory: {dev_max_memory:.1f} MB")

        # Update the learning rate: linear warmup first, then cosine annealing
        if scheduler is not None:
            if epoch <= args.scheduler_warmup_epochs:
                # During warmup, ramp the LR up to the base value
                warmup_lr = args.learning_rate * epoch / args.scheduler_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"\n[Scheduler] Learning rate (warmup): {warmup_lr:.6f}")
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n[Scheduler] Learning rate updated to: {current_lr:.6f}")

        # Did the model improve?
        metric_value = dev_metrics[args.early_stopping_metric]
        print(f"\n[Early Stopping] Current {args.early_stopping_metric}: {metric_value:.4f}")

        # Is this the best model so far?
        is_best = False
        if best_dev_metrics is None:
            is_best = True
        elif args.early_stopping_mode == 'min':
            is_best = metric_value < best_dev_metrics[args.early_stopping_metric]
        else:
            is_best = metric_value > best_dev_metrics[args.early_stopping_metric]

        if is_best:
            print(f"[Early Stopping] New best model found! (previous: {best_dev_metrics[args.early_stopping_metric] if best_dev_metrics else 'N/A'})")
            best_epoch = epoch
            best_train_metrics = train_metrics
            best_dev_metrics = dev_metrics

            # Save temporary best model
            print(f"[Checkpoint] Saving temporary best model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'dev_metrics': dev_metrics
            }, temp_best_path)
            print(f"[Checkpoint] Temporary model saved to {temp_best_path}")

        # Record timing for this epoch
        timing_records.append({
            'run_num': train_run_num,
            'type': 1,
            'time': round(train_elapsed_time, 1),
            'max_memory': round(train_max_memory, 1),
            'if_best': 1 if is_best else 0,
            'eer': round(train_metrics.get('eer', 0.0), 4),
            'min_dcf': round(train_metrics.get('min_dcf', 0.0), 4),
            'cllr': round(train_metrics.get('cllr', 0.0), 4),
            'act_dcf': round(train_metrics.get('act_dcf', 0.0), 4),
        })
        timing_records.append({
            'run_num': dev_run_num,
            'type': 2,
            'time': round(dev_elapsed_time, 1),
            'max_memory': round(dev_max_memory, 1),
            'if_best': 1 if is_best else 0,
            'eer': round(dev_metrics.get('eer', 0.0), 4),
            'min_dcf': round(dev_metrics.get('min_dcf', 0.0), 4),
            'cllr': round(dev_metrics.get('cllr', 0.0), 4),
            'act_dcf': round(dev_metrics.get('act_dcf', 0.0), 4),
        })

        # Early stopping check
        should_stop = early_stopping(metric_value)
        if should_stop:
            print(f"\n[Early Stopping] Triggered! No improvement for {args.early_stopping_patience} epochs.")
            print(f"[Early Stopping] Best epoch: {best_epoch}")
            break

        # Clear CUDA cache
        clear_cuda_cache()

    # ===== Step 6: Final Evaluation on Eval Set =====
    print("\n" + "="*80)
    print("TRAINING COMPLETE - RUNNING FINAL EVALUATION")
    print("="*80)
    print(f"\nBest model found at epoch: {best_epoch}")
    print(f"Loading best model for final evaluation on eval set...")

    # Load the best model back from the temporary checkpoint
    from utils import load_model_weights
    model = load_model_weights(model, temp_best_path, device, strict=True)

    # Clear GPU cache before final evaluation
    safe_empty_cache()

    # Evaluate on eval set with timing
    print(f"\n[Final Evaluation] Running on eval set...")
    eval_run_num += 1
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    eval_start_time = time.time()
    eval_metrics, eval_max_memory = validate(
        model, eval_loader, criterion, device,
        use_tta=args.use_tta, desc="Final Eval"
    )
    eval_elapsed_time = time.time() - eval_start_time
    eval_max_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0

    print(f"\n[Eval Results]")
    print_metrics(eval_metrics, prefix="  ")
    print(f"  Time: {eval_elapsed_time:.1f}s")
    print(f"  Max Memory: {eval_max_memory:.1f} MB")

    # Record timing for the eval run
    timing_records.append({
        'run_num': eval_run_num,
        'type': 3,
        'time': round(eval_elapsed_time, 1),
        'max_memory': round(eval_max_memory, 1),
        'if_best': 1,
        'eer': round(eval_metrics.get('eer', 0.0), 4),
        'min_dcf': round(eval_metrics.get('min_dcf', 0.0), 4),
        'cllr': round(eval_metrics.get('cllr', 0.0), 4),
        'act_dcf': round(eval_metrics.get('act_dcf', 0.0), 4),
    })

    # ===== Step 7: Save Final Model =====
    print("\n" + "="*80)
    print("STEP 7: SAVING FINAL MODEL")
    print("="*80)

    # Best metric value from the dev set
    best_metric_value = best_dev_metrics[args.early_stopping_metric]

    # Build a model name that includes the metric and its value
    model_name = f"best_model_{args.early_stopping_metric}_{best_metric_value:.4f}"

    print(f"\nSaving final model with:")
    print(f"  - Metric: {args.early_stopping_metric}")
    print(f"  - Value: {best_metric_value:.4f}")
    print(f"  - Model name: {model_name}")

    # Save the final checkpoint with all metrics and timing records
    save_checkpoint(
        args.save_dir, model_name, model, optimizer,
        args, best_train_metrics, best_dev_metrics, eval_metrics, best_epoch,
        timing_records=timing_records
    )

    # Remove the temporary checkpoint
    if os.path.exists(temp_best_path):
        os.remove(temp_best_path)
        print(f"\n[Cleanup] Removed temporary checkpoint: {temp_best_path}")

    # ===== Summary =====
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n[Train - Epoch {best_epoch}]")
    print_metrics(best_train_metrics, prefix="  ")

    print(f"\n[Dev - Epoch {best_epoch}]")
    print_metrics(best_dev_metrics, prefix="  ")

    print(f"\n[Eval - Final]")
    print_metrics(eval_metrics, prefix="  ")

    # Output file paths
    model_dir = os.path.join(args.save_dir, model_name)
    model_pt_path = os.path.join(model_dir, f"{model_name}.pt")
    model_json_path = os.path.join(model_dir, f"{model_name}.json")
    timing_csv_path = os.path.join(model_dir, "timing_records.csv")

    print(f"\n{'='*80}")
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Model directory: {model_dir}")
    print(f"  - Model weights: {model_pt_path}")
    print(f"  - Configuration: {model_json_path}")
    print(f"  - Timing records: {timing_csv_path}")
    print(f"\nTotal runs:")
    print(f"  - Train runs: {train_run_num}")
    print(f"  - Dev runs: {dev_run_num}")
    print(f"  - Eval runs: {eval_run_num}")


if __name__ == "__main__":
    main()
