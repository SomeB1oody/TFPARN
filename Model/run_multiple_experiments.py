"""
Runs several ASVspoof5 training experiments in a row.
Define each experiment with its own ModelArgs, then run them one after another. Results are saved automatically.
"""

import torch
import torch.optim as optim
from typing import List, Dict, Tuple
import sys
import os
from datetime import datetime
import json
import time
import csv
import numpy as np

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    create_loss_function,
    print_metrics, count_parameters, save_model,
    EarlyStopping, load_model_weights, evaluate_with_calibration,
    compute_cllr
)
from main_train import ModelArgs, train_one_epoch, validate


# Experiment Configuration
def create_experiment_list() -> List[ModelArgs]:
    """
    Define the experiments to run here.
    Each one is a full ModelArgs config.

    Returns:
        List of ModelArgs configs to run
    """
    experiments = []

    # Experiment 1
    exp1 = ModelArgs()
    exp1.learning_rate = 0.667e-4
    exp1.weight_decay = 1e-2
    exp1.pooling_method = "attention"
    exp1.loss_type = "focal"
    exp1.enable_pairwise = True
    exp1.focal_alpha = 0.5
    exp1.focal_gamma = 2.0
    exp1.save_dir = ""

    # Experiment 2
    exp2 = ModelArgs()
    exp2.learning_rate = 0.667e-4
    exp2.weight_decay = 1e-2
    exp2.pooling_method = "mean"
    exp2.loss_type = "focal"
    exp2.enable_pairwise = True
    exp2.focal_alpha = 0.5
    exp2.focal_gamma = 2.0
    exp2.save_dir = ""

    # Experiment 3
    exp3 = ModelArgs()
    exp3.learning_rate = 0.667e-4
    exp3.weight_decay = 1e-2
    exp3.pooling_method = "mean"
    exp3.loss_type = "ce"
    exp3.enable_pairwise = True
    exp3.focal_alpha = 0.5
    exp3.focal_gamma = 2.0
    exp3.save_dir = ""

    # Experiment 4
    exp4 = ModelArgs()
    exp4.learning_rate = 0.667e-4
    exp4.weight_decay = 1e-2
    exp4.pooling_method = "mean"
    exp4.loss_type = "ce"
    exp4.enable_pairwise = False
    exp4.focal_alpha = 0.5
    exp4.focal_gamma = 2.0
    exp4.save_dir = ""

    experiments.append(exp1)
    experiments.append(exp2)
    experiments.append(exp3)
    experiments.append(exp4)

    return experiments


# Single Experiment Runner
def run_single_experiment(
    args: ModelArgs,
    experiment_idx: int,
    total_experiments: int
) -> Dict:
    """
    Run one experiment with the given ModelArgs

    Args:
        args: ModelArgs config for this experiment
        experiment_idx: Index of the current experiment (1-based)
        total_experiments: How many experiments in total

    Returns:
        Dict with the experiment results
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT {experiment_idx}/{total_experiments}")
    print("="*80)

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device()
    clear_cuda_cache()

    # Step 1: Load Data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

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
    data_args.use_tta = True  # use TTA for validation during training
    data_args.tta_num_crops = 5

    train_loader, dev_loader, eval_loader = make_loaders(data_args)

    # Step 2: Create Model
    print("\n" + "="*80)
    print("STEP 2: CREATING MODEL")
    print("="*80)

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

    model = create_model(model_args)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"[✓] Model created with {num_params:,} trainable parameters")

    # Step 3: Create Loss Function
    print("\n" + "="*80)
    print("STEP 3: CREATING LOSS FUNCTION")
    print("="*80)

    if args.loss_type == "focal":
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

    # Step 4: Create Optimizer and Scheduler
    print("\n" + "="*80)
    print("STEP 4: CREATING OPTIMIZER AND SCHEDULER")
    print("="*80)

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

    if args.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs - args.scheduler_warmup_epochs,
            eta_min=1e-6
        )
    elif args.scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    else:
        scheduler = None

    print(f"[✓] Optimizer and scheduler created")

    # Step 5: Training Loop
    print("\n" + "="*80)
    print("STEP 5: TRAINING")
    print("="*80)

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        mode=args.early_stopping_mode,
        delta=0.0001
    )

    best_metric = -float('inf') if args.early_stopping_mode == 'max' else float('inf')
    best_epoch = 0

    os.makedirs(args.save_dir, exist_ok=True)
    temp_best_path = f"{args.save_dir}/best_model.pt"

    # Initialize timing records
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

        # Compute Cllr for train
        train_probs = torch.softmax(train_logits, dim=1)[:, 1].numpy()
        train_labels_np = train_labels.numpy()
        eps = 1e-10
        train_probs_clipped = np.clip(train_probs, eps, 1 - eps)
        train_cllr = compute_cllr(train_probs_clipped, train_labels_np)

        print(f"\nTrain Loss: {train_loss:.6f}")
        print_metrics(train_metrics, prefix="  [TRAIN] ")

        # Validate
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        val_start_time = time.time()
        val_loss, val_metrics, val_logits, val_labels = validate(
            model, dev_loader, criterion, device, epoch, use_tta=True
        )
        val_time = time.time() - val_start_time
        val_max_memory = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
        dev_run_num += 1

        # Compute Cllr for validation
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].numpy()
        val_labels_np = val_labels.numpy()
        val_probs_clipped = np.clip(val_probs, eps, 1 - eps)
        val_cllr = compute_cllr(val_probs_clipped, val_labels_np)

        print(f"\nValidation Loss: {val_loss:.6f}")
        print_metrics(val_metrics, prefix="  [VAL] ")

        # Update learning rate: linear warmup, then cosine annealing
        if scheduler is not None:
            if epoch <= args.scheduler_warmup_epochs:
                # Warmup: ramp LR up from small to base
                warmup_lr = args.learning_rate * epoch / args.scheduler_warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"\nLearning rate (warmup): {warmup_lr:.8f}")
            else:
                # After warmup, use cosine annealing
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\nLearning rate: {current_lr:.8f}")

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': best_metric,
                'metric_name': args.early_stopping_metric
            }, temp_best_path)

        # Record timing for this epoch
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

        # Early stopping check
        if early_stopping(current_metric):
            print(f"\n[!] Early stopping triggered at epoch {epoch}")
            print(f"  Best {args.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
            break

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    # Step 6: Evaluation
    print("\n" + "="*80)
    print("STEP 6: EVALUATION")
    print("="*80)

    # Load best model
    model = load_model_weights(model, temp_best_path, device, strict=True)

    # Rebuild the data loaders with TTA on for the final evaluation
    print("\n[INFO] Recreating data loaders with TTA enabled for final evaluation")
    data_args.use_tta = True
    _, dev_loader_tta, eval_loader_tta = make_loaders(data_args)

    # Evaluate without calibration, using the TTA loaders
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

    final_train_metrics = results['train']['initial_metrics']
    final_val_metrics = results['dev']['initial_metrics']
    eval_metrics = results['eval']['initial_metrics']

    # Record final evaluation timing
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

    print("\n" + "-"*80)
    print("FINAL METRICS")
    print("-"*80)
    print_metrics(final_train_metrics, prefix="  [TRAIN] ")
    print_metrics(final_val_metrics, prefix="  [VAL] ")
    print_metrics(eval_metrics, prefix="  [TEST] ")

    # Step 7: Save Model
    print("\n" + "="*80)
    print("STEP 7: SAVE MODEL")
    print("="*80)

    model_name = f"best_model_{args.early_stopping_metric}_{best_metric:.4f}"
    model_dir = os.path.join(args.save_dir, model_name)

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

    print(f"[✓] Model saved to {model_dir}")

    # Save timing records to final model directory
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

    # Return experiment results
    return {
        'experiment_idx': experiment_idx,
        'model_name': model_name,
        'model_dir': model_dir,
        'best_epoch': best_epoch,
        'best_val_metric': best_metric,
        'train_metrics': final_train_metrics,
        'val_metrics': final_val_metrics,
        'test_metrics': eval_metrics,
        'model_params': num_params,
        'config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.model_dropout,
            'learning_rate': args.learning_rate,
            'loss_type': args.loss_type
        }
    }


# Main Runner
def main():
    """
    Run all the experiments
    """
    print("\n" + "="*80)
    print("MULTIPLE EXPERIMENTS RUNNER")
    print("="*80)

    # Get experiment list
    experiments = create_experiment_list()
    total_experiments = len(experiments)

    print(f"\nTotal experiments to run: {total_experiments}")
    print("\nExperiment configurations:")
    for i, exp in enumerate(experiments, 1):
        print(f"\n  Experiment {i}:")
        print(f"    - Model: d_model={exp.d_model}, layers={exp.num_layers}, heads={exp.nhead}")
        print(f"    - Learning rate: {exp.learning_rate}")
        print(f"    - Loss type: {exp.loss_type}")
        print(f"    - Max epochs: {exp.max_epochs}")

    # Run all experiments
    all_results = []
    start_time = datetime.now()

    for i, exp_args in enumerate(experiments, 1):
        try:
            result = run_single_experiment(exp_args, i, total_experiments)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n\n[!] Experiment interrupted by user")
            break
        except Exception as e:
            print(f"\n\n[!] Experiment {i} failed with error:")
            print(f"  {str(e)}")
            # Continue with next experiment
            continue

    end_time = datetime.now()
    duration = end_time - start_time

    # ========================================================================
    # Summary of All Experiments
    # ========================================================================
    print("\n" + "="*80)
    print("ALL EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"\nTotal time: {duration}")
    print(f"Completed experiments: {len(all_results)}/{total_experiments}")

    if all_results:
        # Save summary to JSON
        summary_path = os.path.join("./", f"experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Convert tensors to floats for JSON serialization
        json_results = []
        for result in all_results:
            json_result = result.copy()
            for key in ['train_metrics', 'val_metrics', 'test_metrics']:
                json_result[key] = {k: float(v) for k, v in result[key].items()}
            json_results.append(json_result)

        with open(summary_path, 'w') as f:
            json.dump({
                'total_experiments': total_experiments,
                'completed_experiments': len(all_results),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'results': json_results
            }, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")

        # Print comparison table
        print("\n" + "-"*80)
        print("RESULTS COMPARISON")
        print("-"*80)
        print(f"{'Exp':<5} {'Model':<20} {'Val EER':<10} {'Test EER':<10} {'Test F1':<10} {'Params':<12}")
        print("-"*80)

        for result in all_results:
            exp_idx = result['experiment_idx']
            model_desc = f"d{result['config']['d_model']}_l{result['config']['num_layers']}"
            val_eer = result['val_metrics']['eer']
            test_eer = result['test_metrics']['eer']
            test_f1 = result['test_metrics']['f1_macro']
            params = result['model_params']

            print(f"{exp_idx:<5} {model_desc:<20} {val_eer:<10.4f} {test_eer:<10.4f} {test_f1:<10.4f} {params:<12,}")

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Runner interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[!] Runner failed with error:")
        print(f"  {str(e)}")
        raise
