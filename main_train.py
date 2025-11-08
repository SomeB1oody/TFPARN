"""
Main Training Script for ASVspoof5 Competition
Handles: data loading, model training, validation, evaluation, and checkpoint saving
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Dict
import sys

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    create_loss_function, compute_all_metrics,
    print_metrics, print_classification_report_wrapper,
    count_parameters, save_model, EarlyStopping,
    load_model_weights, evaluate_with_calibration
)


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class ModelArgs:
    """
    Complete configuration for training
    Includes: data processing, model architecture, and training hyperparameters
    """
    # Data Processing Parameters (from data_process.py)
    train_data_dir: str = "H:/true_tone5/data/flac_T/"
    dev_data_dir: str = "H:/true_tone5/data/flac_D/"
    eval_data_dir: str = "H:/true_tone5/data/flac_E/"
    train_protocol_dir: str = "H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.train.tsv"
    dev_protocol_dir: str = "H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv"
    eval_protocol_dir: str = "H:/true_tone5/data/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv"

    sample_rate: int = 16000
    duration_sec: float = 4.0
    mono: bool = True
    normalize: bool = True
    batch_size: int = 96
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    train_shuffle: bool = True

    # Model Parameters (from model.py)
    n_mels: int = 128
    n_fft: int = 768
    hop_length: int = 160
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    model_dropout: float = 0.5
    activation: str = "relu"

    # Training Hyperparameters
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_type: str = "adamw"  # 'adam' or 'adamw'
    scheduler_type: str = "cosine"  # 'cosine', 'step', or 'none'
    scheduler_warmup_epochs: int = 5

    # Loss Function ('ce' or 'focal')
    loss_type: str = "ce"
    focal_alpha: float = 0.5  # Alpha for focal loss (positive class weight, negative uses 1-alpha)
    focal_gamma: float = 2.0  # Gamma for focal loss

    # Pairwise AUC/pAUC Loss
    enable_pairwise: bool = True  # Whether to enable pairwise loss
    pairwise_margin: float = 1.0  # Margin for pairwise ranking loss
    pairwise_weight: float = 0.5  # Weight for pairwise loss term

    # Early Stopping
    early_stopping_patience: int = 15
    early_stopping_metric: str = "eer"  # Options: 'f1_macro', 'accuracy', 'recall_macro', 'eer', 'auc_roc'
    early_stopping_mode: str = "min"  # 'max' for f1/acc/recall/auc, 'min' for eer

    # Model Checkpoint
    save_dir: str = "./checkpoints/"

    # Other
    seed: int = 42


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
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

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
        lengths = batch['lengths'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(waveforms, lengths)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'avg_loss': f'{avg_loss:.3f}'})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_tta: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Validate model

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        use_tta: Whether TTA is enabled (affects batch shape)

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
            lengths = batch['lengths'].to(device)
            labels = batch['labels'].to(device)

            if use_tta:
                # TTA enabled: waveforms shape [B, num_crops, C, T]
                B, num_crops, C, T = waveforms.shape

                # Reshape to [B*num_crops, C, T] for batch processing
                waveforms_flat = waveforms.view(B * num_crops, C, T)
                lengths_flat = lengths.unsqueeze(1).expand(B, num_crops).reshape(B * num_crops)

                # Forward pass on all crops
                logits_flat = model(waveforms_flat, lengths_flat)  # [B*num_crops, 2]

                # Reshape back to [B, num_crops, 2]
                logits_crops = logits_flat.view(B, num_crops, 2)

                # Average logits across crops
                logits = logits_crops.mean(dim=1)  # [B, 2]
            else:
                # Normal inference: waveforms shape [B, C, T]
                logits = model(waveforms, lengths)

            # Compute loss
            loss = criterion(logits, labels)

            # Update metrics
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_all_metrics(all_logits, all_labels)

    return avg_loss, metrics


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """
    Main training function
    """
    print("\n" + "="*80)
    print("ASVspoof5 TRAINING PIPELINE")
    print("="*80)

    # Initialize configuration
    args = ModelArgs()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    clear_cuda_cache()

    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    # Create data processing args
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
    data_args.train_shuffle = args.train_shuffle
    data_args.seed = args.seed
    data_args.use_tta = True  # Enable TTA for dev/eval
    data_args.tta_num_crops = 5  # Number of crops per sample

    # Load data
    train_loader, dev_loader, eval_loader, class_weights = make_loaders(data_args)

    # ========================================================================
    # Step 2: Create Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)

    # Create model args
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

    # Create model
    model = create_model(model_args)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"[✓] Model created with {num_params:,} trainable parameters")

    # ========================================================================
    # Step 3: Create Loss Function
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: CREATING LOSS FUNCTION")
    print("="*80)

    if args.loss_type == "focal":
        # Convert single alpha to [negative_class, positive_class] format
        focal_alpha = torch.tensor([1.0 - args.focal_alpha, args.focal_alpha], dtype=torch.float32)
        criterion = create_loss_function(
            args.loss_type,
            class_weights,
            focal_alpha=focal_alpha,
            focal_gamma=args.focal_gamma,
            enable_pairwise=args.enable_pairwise,
            pairwise_margin=args.pairwise_margin,
            pairwise_weight=args.pairwise_weight
        )
    else:
        criterion = create_loss_function(
            args.loss_type,
            class_weights,
            enable_pairwise=args.enable_pairwise,
            pairwise_margin=args.pairwise_margin,
            pairwise_weight=args.pairwise_weight
        )

    criterion = criterion.to(device)

    # ========================================================================
    # Step 4: Create Optimizer and Scheduler
    # ========================================================================
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

    # Create scheduler
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

    # ========================================================================
    # Step 5: Training Loop
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80)
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Early stopping patience: {args.early_stopping_patience}")
    print(f"  - Early stopping metric: {args.early_stopping_metric}")
    print(f"  - Early stopping mode: {args.early_stopping_mode}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode=args.early_stopping_mode,
        delta=0.0001
    )

    best_metric = -float('inf') if args.early_stopping_mode == 'max' else float('inf')
    best_epoch = 0

    # Path for temporary best model
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    temp_best_path = f"{args.save_dir}/best_model.pt"

    for epoch in range(1, args.max_epochs + 1):
        print("\n" + "-"*80)
        print(f"EPOCH {epoch}/{args.max_epochs}")
        print("-"*80)

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print_metrics(train_metrics, prefix="  [TRAIN] ")

        # Validate (without TTA)
        val_loss, val_metrics = validate(
            model, dev_loader, criterion, device, epoch, use_tta=False
        )

        print(f"\nValidation Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="  [VAL] ")

        # Update scheduler
        if scheduler is not None and epoch > args.scheduler_warmup_epochs:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nLearning rate: {current_lr:.6f}")

        # Check for improvement
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
            # Save temporary best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_metric,
                'metric_name': args.early_stopping_metric
            }, temp_best_path)
            print(f"  [✓] Saved temporary best model to {temp_best_path}")

        # Early stopping check
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

    # ========================================================================
    # Step 6: Load Best Model and Evaluate on Test Set
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: LOAD BEST MODEL AND EVALUATE ON TEST SET")
    print("="*80)

    # Load best model from temporary checkpoint
    model = load_model_weights(model, temp_best_path, device, strict=True)

    # Use complete evaluation pipeline with calibration
    results = evaluate_with_calibration(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        eval_loader=eval_loader,
        device=device,
        apply_calibration=True,
        enable_prior_correction=True
    )

    # Extract results for convenience
    final_train_metrics = results['train']['initial_metrics']
    final_val_metrics = results['dev']['initial_metrics']
    eval_metrics = results['eval']['initial_metrics']

    # Print final evaluation results
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)

    # Print initial metrics (without calibration)
    print("\n" + "-"*80)
    print("INITIAL METRICS (No Calibration)")
    print("-"*80)
    print_metrics(final_train_metrics, prefix="  [TRAIN] ")
    print_metrics(final_val_metrics, prefix="  [VAL] ")
    print_metrics(eval_metrics, prefix="  [TEST] ")

    # Print calibrated metrics (if available)
    if 'calibrated_metrics' in results['eval']:
        print("\n" + "-"*80)
        print("CALIBRATED METRICS (With Calibration + Prior Correction)")
        print("-"*80)
        print_metrics(results['dev']['calibrated_metrics'], prefix="  [VAL] ")
        print_metrics(results['eval']['calibrated_metrics'], prefix="  [TEST] ")

        # Print classification report using calibrated scores
        import numpy as np
        eps = 1e-10
        calibrated_scores_clipped = np.clip(results['eval']['calibrated_scores'], eps, 1 - eps)
        calibrated_logits = np.stack([
            np.log(1 - calibrated_scores_clipped),
            np.log(calibrated_scores_clipped)
        ], axis=1)

        print("\n" + "-"*80)
        print("CLASSIFICATION REPORT (Calibrated)")
        print("-"*80)
        print_classification_report_wrapper(
            torch.from_numpy(calibrated_logits),
            torch.from_numpy(results['eval']['labels_np']),
            target_names=['spoof (AI)', 'bonafide (human)']
        )
    else:
        # Print classification report using initial scores
        print("\n" + "-"*80)
        print("CLASSIFICATION REPORT (Initial)")
        print("-"*80)
        print_classification_report_wrapper(
            results['eval']['logits'],
            results['eval']['labels'],
            target_names=['spoof (AI)', 'bonafide (human)']
        )

    # ========================================================================
    # Step 7: Save Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: SAVE MODEL")
    print("="*80)

    response = input("\nDo you want to save the model? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        # Get the metric value from eval set
        eval_metric_value = eval_metrics[args.early_stopping_metric]

        # Create model name with metric and value
        model_name = f"best_model_{args.early_stopping_metric}_{eval_metric_value:.4f}"

        # Create directory with same name as model
        model_dir = os.path.join(args.save_dir, model_name)

        print(f"\nSaving model to directory: {model_dir}")

        # Call save_model function
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
        print(f"  - Value on Test set: {eval_metric_value:.4f}")
    else:
        print("\n[!] Model not saved")

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
