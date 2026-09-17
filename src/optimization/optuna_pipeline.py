import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader, WeightedRandomSampler
import optuna
from optuna.pruners import MedianPruner

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.hybrid_1d_bi_cnn_gru import Hybrid1DBiCNNGRU
from src.engine.losses import FocalLoss, create_loss_function
from src.data.dataset import build_dataset_from_records, ArrhythmiaDataset
from src.utils.metrics import compute_metrics, print_metrics_report


def get_target_device() -> torch.device:
    """
    Selects target computing device explicitly for Apple Silicon Mac M1 acceleration (MPS)
    or falls back to CPU if unavailable.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device} (Apple Silicon MPS Acceleration: {torch.backends.mps.is_available()})")
    return device


def clean_mps_memory(device: torch.device):
    """Frees allocated memory caches on Apple Silicon MPS device."""
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_cached_dataloaders(
    train_ds: ArrhythmiaDataset,
    val_ds: ArrhythmiaDataset,
    test_ds: ArrhythmiaDataset,
    batch_size: int = 64,
    use_oversampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds efficient DataLoaders from pre-cached in-memory datasets.
    
    Args:
        train_ds: Cached Training dataset
        val_ds: Cached Validation dataset
        test_ds: Cached Test dataset
        batch_size: Mini-batch size
        use_oversampling: Whether to use WeightedRandomSampler for training batches
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    # Standard AAMI class weights: Normal=1.0, SVEB=10.0, VEB=3.0, Fusion=10.0, Unknown=50.0
    class_weights_np = np.array([1.0, 10.0, 3.0, 10.0, 50.0], dtype=np.float32)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    if use_oversampling:
        sample_weights = class_weights_np[train_ds.labels.numpy()]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_weights_tensor


class ModelCObjective:
    """
    Optuna Objective callable for tuning Model C (hybrid_1d_bi_cnn_gru).
    
    Optimizes for Macro F1-Score on the validation set across severe class imbalance.
    Integrates dynamic loss function tuning (Focal Loss vs Weighted Cross-Entropy),
    dynamic architectural hyperparameters, early stopping pruning via MedianPruner,
    and Mac M1 MPS memory management.
    """

    def __init__(
        self,
        train_ds: ArrhythmiaDataset,
        val_ds: ArrhythmiaDataset,
        config: Dict[str, Any],
        device: torch.device,
        epochs_per_trial: int = 15,
        batch_size: int = 64,
        best_model_save_dir: Optional[Path] = None
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.config = config
        self.device = device
        self.epochs_per_trial = epochs_per_trial
        self.batch_size = batch_size
        self.best_model_save_dir = best_model_save_dir or Path("./checkpoints")
        self.best_model_save_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = config["data"].get("num_classes", 5)

        # Pre-build validation loader once
        self.val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

        # Track global best Macro F1 score across all completed trials
        self.global_best_macro_f1 = -1.0
        self.global_best_trial_num = -1

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Executes a single optimization trial.
        
        Args:
            trial (optuna.Trial): Optuna trial instance.
            
        Returns:
            float: Best Validation Macro F1-Score achieved during the trial.
        """
        # ==========================================
        # 1. Architectural Search Space
        # ==========================================
        conv_out_channels = trial.suggest_categorical("conv_out_channels", [16, 32, 64])
        conv_kernel_size = trial.suggest_categorical("conv_kernel_size", [3, 5, 7])
        gru_hidden_size = trial.suggest_categorical("gru_hidden_size", [32, 64, 128])
        gru_num_layers = trial.suggest_int("gru_num_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # ==========================================
        # 2. Loss Function Search Space
        # ==========================================
        loss_type = trial.suggest_categorical("loss_type", ["focal", "weighted_ce"])
        if loss_type == "focal":
            focal_gamma = trial.suggest_float("focal_gamma", 1.5, 3.5)
        else:
            focal_gamma = None

        # Build DataLoader with WeightedRandomSampler for balanced class representation
        train_loader, _, _, class_weights = build_cached_dataloaders(
            self.train_ds, self.val_ds, self.val_ds,
            batch_size=self.batch_size,
            use_oversampling=True
        )
        class_weights = class_weights.to(self.device)

        # Instantiate Model C
        model = Hybrid1DBiCNNGRU(
            in_channels=1,
            num_classes=self.num_classes,
            conv_out_channels=conv_out_channels,
            conv_kernel_size=conv_kernel_size,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            dropout=dropout
        ).to(self.device)

        param_count = model.get_num_parameters()
        trial.set_user_attr("parameter_count", param_count)

        # Instantiate Loss Function
        if loss_type == "focal":
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer & Learning Rate Scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_val_macro_f1 = -1.0
        best_state_dict = None

        print(f"\n[Trial #{trial.number:02d} Start] Params: {param_count:,} | Conv: ({conv_out_channels}ch, k={conv_kernel_size}) | "
              f"GRU: (h={gru_hidden_size}, layers={gru_num_layers}) | Dropout: {dropout:.2f} | LR: {lr:.5f} | "
              f"Loss: {loss_type.upper()} {f'(gamma={focal_gamma:.2f})' if focal_gamma else ''}")

        try:
            for epoch in range(1, self.epochs_per_trial + 1):
                # -----------------
                # Training Phase
                # -----------------
                model.train()
                train_loss = 0.0
                train_total = 0

                for batch in train_loader:
                    # Dataset yields (batch_2d, batch_1d, batch_rr, targets)
                    _, batch_1d, _, targets = batch
                    batch_1d = batch_1d.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_1d)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * targets.size(0)
                    train_total += targets.size(0)

                avg_train_loss = train_loss / train_total

                # -----------------
                # Validation Phase
                # -----------------
                model.eval()
                val_loss = 0.0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for batch in self.val_loader:
                        _, batch_1d, _, targets = batch
                        batch_1d = batch_1d.to(self.device)
                        targets = targets.to(self.device)

                        outputs = model(batch_1d)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item() * targets.size(0)
                        preds = outputs.argmax(dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())

                avg_val_loss = val_loss / len(all_targets)
                y_true = np.array(all_targets)
                y_pred = np.array(all_preds)

                # Calculate Multi-Class Metrics (PRIMARY: Macro F1-Score)
                metrics = compute_metrics(y_true, y_pred, num_classes=self.num_classes)
                val_macro_f1 = metrics["macro_f1"]
                val_acc = metrics["accuracy"]
                val_macro_sens = metrics["macro_sensitivity"]

                # Step Scheduler on Macro F1
                scheduler.step(val_macro_f1)

                # Keep track of best trial epoch
                if val_macro_f1 > best_val_macro_f1:
                    best_val_macro_f1 = val_macro_f1
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                # Report metric for Optuna Pruning
                trial.report(val_macro_f1, epoch)

                # Print epoch summary
                print(f" Trial #{trial.number:02d} | Ep [{epoch:02d}/{self.epochs_per_trial:02d}] "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}% | Val Macro Sens: {val_macro_sens:.2f}% | "
                      f"Val Macro F1: {val_macro_f1:.2f}%")

                # Check if trial should be pruned
                if trial.should_prune():
                    print(f" [!] Trial #{trial.number:02d} PRUNED early by MedianPruner at epoch {epoch} (Val Macro F1: {val_macro_f1:.2f}%).")
                    raise optuna.TrialPruned()

            # Save best trial model state if it is global best
            if best_val_macro_f1 > self.global_best_macro_f1:
                self.global_best_macro_f1 = best_val_macro_f1
                self.global_best_trial_num = trial.number
                best_ckpt_path = self.best_model_save_dir / "model_c_optuna_best.pth"
                clean_params = {
                    k: (int(v) if isinstance(v, (int, np.integer)) else
                        float(v) if isinstance(v, (float, np.floating)) else v)
                    for k, v in trial.params.items()
                }
                torch.save({
                    "trial_number": trial.number,
                    "params": clean_params,
                    "val_macro_f1": float(best_val_macro_f1),
                    "model_state_dict": best_state_dict,
                    "config": self.config
                }, best_ckpt_path)
                print(f" [★] New Global Best Trial #{trial.number:02d}! Saved checkpoint to '{best_ckpt_path}' (Val Macro F1: {best_val_macro_f1:.2f}%)")

            return best_val_macro_f1

        finally:
            # ==========================================
            # Memory Management: Apple Mac M1 MPS Cache
            # ==========================================
            del model, optimizer, criterion
            clean_mps_memory(self.device)


def run_optuna_study(
    config_path: str = "config.yaml",
    n_trials: int = 20,
    epochs_per_trial: int = 15,
    batch_size: int = 64,
    study_name: str = "model_c_hybrid_1d_bi_cnn_gru_opt",
    storage: Optional[str] = None,
    results_dir: str = "./results"
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Orchestrates the complete Optuna hyperparameter optimization study for Model C.
    
    1. Sets target compute device to Apple Silicon MPS if available.
    2. Pre-caches datasets in memory (AAMI DS1 Train/Val and DS2 Test).
    3. Initializes MedianPruner for early pruning of non-promising trials.
    4. Executes optimization study maximizing Validation Macro F1-Score.
    5. Evaluates the best discovered architecture on unseen DS2 Test set.
    6. Generates full evaluation report, metrics summary, and exports results.
    """
    print("=" * 80)
    print("      MODEL C (hybrid_1d_bi_cnn_gru) OPTUNA HYPERPARAMETER OPTIMIZATION     ")
    print("=" * 80)

    # 1. Device Setup
    device = get_target_device()

    # 2. Load Configuration & Pre-cache Datasets
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["training"].get("checkpoint_dir", "./checkpoints"))
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print("\n[*] Pre-caching MIT-BIH Datasets into Memory (DS1 Train/Val, DS2 Test)...")
    start_data_time = time.time()
    train_ds, val_ds, test_ds, cfg = build_dataset_from_records(config_path)
    print(f"[✓] Data pre-caching completed in {time.time() - start_data_time:.2f}s.\n")

    # 3. Setup Pruner and Objective
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    
    objective = ModelCObjective(
        train_ds=train_ds,
        val_ds=val_ds,
        config=cfg,
        device=device,
        epochs_per_trial=epochs_per_trial,
        batch_size=batch_size,
        best_model_save_dir=checkpoint_path
    )

    # 4. Create and Run Study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Primary Metric: Maximize Validation Macro F1-Score
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )

    print(f"[*] Launching Optuna Study '{study_name}' ({n_trials} trials, {epochs_per_trial} epochs/trial)...")
    print(f"[*] Target Optimization Metric: Validation Macro F1-Score (Direction: MAXIMIZE)")
    print(f"[*] Pruner: MedianPruner (startup_trials=5, warmup_steps=3)\n")

    study_start_time = time.time()
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    study_duration = time.time() - study_start_time

    print("\n" + "=" * 80)
    print("                      OPTUNA STUDY COMPLETED                         ")
    print("=" * 80)
    print(f" Total Trials Run:       {len(study.trials)}")
    print(f" Completed Trials:       {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f" Pruned Trials:          {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f" Best Trial Number:      #{study.best_trial.number}")
    print(f" Best Val Macro F1:      {study.best_value:.2f}%")
    print(f" Total Study Duration:   {study_duration / 60:.2f} minutes")
    print("-" * 80)
    print(" Best Hyperparameters Found:")
    for param_name, param_val in study.best_trial.params.items():
        print(f"   - {param_name:<20}: {param_val}")
    print("=" * 80 + "\n")

    # 5. Export Study Results to CSV & JSON
    df_trials = study.trials_dataframe()
    csv_file = results_path / f"{study_name}_trials.csv"
    df_trials.to_csv(csv_file, index=False)
    print(f"[*] Saved trial logs to '{csv_file}'")

    best_json_file = results_path / f"{study_name}_best_params.json"
    with open(best_json_file, "w") as f:
        json.dump({
            "best_trial_number": study.best_trial.number,
            "best_val_macro_f1": study.best_value,
            "best_params": study.best_trial.params,
            "total_duration_sec": study_duration
        }, f, indent=4)
    print(f"[*] Saved best parameters to '{best_json_file}'")

    # 6. Evaluate Best Discovered Architecture on Unseen DS2 Test Set
    best_ckpt_file = checkpoint_path / "model_c_optuna_best.pth"
    test_metrics = {}
    if best_ckpt_file.exists():
        print("\n" + "=" * 80)
        print("    EVALUATING BEST MODEL C ARCHITECTURE ON UNSEEN DS2 TEST SET      ")
        print("=" * 80)

        ckpt_data = torch.load(best_ckpt_file, map_location=device, weights_only=False)
        best_params = ckpt_data["params"]

        best_model = Hybrid1DBiCNNGRU(
            in_channels=1,
            num_classes=cfg["data"].get("num_classes", 5),
            conv_out_channels=best_params["conv_out_channels"],
            conv_kernel_size=best_params["conv_kernel_size"],
            gru_hidden_size=best_params["gru_hidden_size"],
            gru_num_layers=best_params["gru_num_layers"],
            dropout=best_params["dropout"]
        ).to(device)

        best_model.load_state_dict(ckpt_data["model_state_dict"])
        best_model.eval()

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in test_loader:
                _, batch_1d, _, targets = batch
                batch_1d = batch_1d.to(device)
                outputs = best_model(batch_1d)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

        test_metrics = compute_metrics(np.array(all_targets), np.array(all_preds), num_classes=cfg["data"].get("num_classes", 5))
        class_names = cfg["data"]["classes"]
        print_metrics_report(test_metrics, class_names)

        # Save test evaluation report
        eval_report_file = results_path / f"{study_name}_test_evaluation.json"
        with open(eval_report_file, "w") as f:
            serializable_metrics = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in test_metrics.items()
            }
            json.dump(serializable_metrics, f, indent=4)
        print(f"[*] Saved test evaluation metrics to '{eval_report_file}'\n")

        # Cleanup test model
        del best_model
        clean_mps_memory(device)

    return study, test_metrics
