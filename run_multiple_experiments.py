"""
Multiple Experiments Runner for ASVspoof5 Training
Allows users to define multiple experiments with different ModelArgs and run them sequentially with automatic result saving
"""

import torch
import torch.optim as optim
from typing import List, Dict
import sys
import os
from datetime import datetime
import json

from data_process import make_loaders, DefaultArgs as DataProcessArgs
from model import create_model, SpeechClassifierArgs
from utils import (
    set_seed, get_device, clear_cuda_cache,
    create_loss_function,
    print_metrics, count_parameters, save_model,
    EarlyStopping, load_model_weights, evaluate_with_calibration
)
from main_train import ModelArgs, train_one_epoch, validate


# ============================================================================
# Experiment Configuration
# ============================================================================

def create_experiment_list() -> List[ModelArgs]:
    """
    Define multiple experiments here
    Each experiment is a complete ModelArgs configuration

    Returns:
        List of ModelArgs configurations to run
    """
    experiments = []

    # Experiment 1
    exp1 = ModelArgs()
    exp1.learning_rate = 1e-4
    exp1.loss_type = "ce"
    exp1.save_dir = "./ce_opt/"
    experiments.append(exp1)

    # Experiment 2
    exp2 = ModelArgs()
    exp2.learning_rate = 5e-5
    exp2.loss_type = "ce"
    exp2.save_dir = "./ce_low_lr/"
    experiments.append(exp2)

    # Experiment 3
    exp3 = ModelArgs()
    exp3.learning_rate = 1e-4
    exp3.weight_decay = 1e-2
    exp3.loss_type = "focal"
    exp3.focal_alpha = 0.6
    exp3.focal_gamma = 1.2
    exp3.save_dir = "./focal_0.6_1.2/"
    experiments.append(exp3)

    # Experiment 4
    exp4 = ModelArgs()
    exp4.learning_rate = 1e-4
    exp4.weight_decay = 1e-2
    exp4.loss_type = "focal"
    exp4.focal_alpha = 0.9
    exp4.focal_gamma = 1.2
    exp4.save_dir = "./focal_0.9_1.2/"
    experiments.append(exp4)

    return experiments


# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_single_experiment(
    args: ModelArgs,
    experiment_idx: int,
    total_experiments: int
) -> Dict:
    """
    Run a single experiment with given ModelArgs

    Args:
        args: ModelArgs configuration for this experiment
        experiment_idx: Index of current experiment (1-based)
        total_experiments: Total number of experiments

    Returns:
        Dictionary containing experiment results
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT {experiment_idx}/{total_experiments}")
    print("="*80)

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
    data_args.use_tta = False  # Disable TTA for training-time validation
    data_args.tta_num_crops = 5

    train_loader, dev_loader, eval_loader, class_weights = make_loaders(data_args)

    # ========================================================================
    # Step 2: Create Model
    # ========================================================================
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

    # ========================================================================
    # Step 5: Training Loop
    # ========================================================================
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
    temp_best_path = f"{args.save_dir}/best_model_exp{experiment_idx}.pt"

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

        # Validate
        val_loss, val_metrics = validate(
            model, dev_loader, criterion, device, epoch, use_tta=False
        )

        print(f"\nValidation Loss: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="  [VAL] ")

        # Update scheduler
        if scheduler is not None and epoch > args.scheduler_warmup_epochs:
            scheduler.step()

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

        # Early stopping check
        if early_stopping(current_metric):
            print(f"\n[!] Early stopping triggered at epoch {epoch}")
            print(f"  Best {args.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
            break

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    # ========================================================================
    # Step 6: Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: EVALUATION")
    print("="*80)

    # Load best model
    model = load_model_weights(model, temp_best_path, device, strict=True)

    # Recreate data loaders with TTA enabled for final evaluation
    print("\n[INFO] Recreating data loaders with TTA enabled for final evaluation")
    data_args.use_tta = True
    _, dev_loader_tta, eval_loader_tta, _ = make_loaders(data_args)

    # Evaluate with calibration (with TTA-enabled loaders)
    results = evaluate_with_calibration(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader_tta,
        eval_loader=eval_loader_tta,
        device=device,
        apply_calibration=True,
        enable_prior_correction=True
    )

    final_train_metrics = results['train']['initial_metrics']
    final_val_metrics = results['dev']['initial_metrics']
    eval_metrics = results['eval']['initial_metrics']

    print("\n" + "-"*80)
    print("FINAL METRICS")
    print("-"*80)
    print_metrics(final_train_metrics, prefix="  [TRAIN] ")
    print_metrics(final_val_metrics, prefix="  [VAL] ")
    print_metrics(eval_metrics, prefix="  [TEST] ")

    # ========================================================================
    # Step 7: Save Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: SAVE MODEL")
    print("="*80)

    eval_metric_value = eval_metrics[args.early_stopping_metric]
    model_name = f"exp{experiment_idx}_{args.early_stopping_metric}_{eval_metric_value:.4f}"
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


# ============================================================================
# Main Runner
# ============================================================================

def main():
    """
    Main function to run multiple experiments
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
        summary_path = os.path.join(experiments[0].save_dir, f"experiments_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

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
