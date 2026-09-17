import os
import sys
from pathlib import Path
import unittest
import torch
import numpy as np
import optuna

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.hybrid_1d_bi_cnn_gru import Hybrid1DBiCNNGRU
from src.engine.losses import FocalLoss, create_loss_function
from src.data.dataset import ArrhythmiaDataset
from src.utils.metrics import compute_metrics
from src.optimization.optuna_pipeline import (
    get_target_device,
    clean_mps_memory,
    build_cached_dataloaders,
    ModelCObjective
)


class TestOptunaModelCPipeline(unittest.TestCase):

    def setUp(self):
        self.device = get_target_device()
        self.num_classes = 5
        self.batch_size = 16
        self.seq_len = 256

        # Create synthetic datasets for fast testing
        n_train, n_val, n_test = 64, 32, 32
        
        # Synthetic 1D ECG: (N, 256), 2D: (N, 16, 16), RR: (N, 2), Labels: (N,)
        train_1d = np.random.randn(n_train, self.seq_len).astype(np.float32)
        train_2d = np.random.randn(n_train, 16, 16).astype(np.float32)
        train_rr = np.random.randn(n_train, 2).astype(np.float32)
        train_y = np.random.randint(0, self.num_classes, size=(n_train,))

        val_1d = np.random.randn(n_val, self.seq_len).astype(np.float32)
        val_2d = np.random.randn(n_val, 16, 16).astype(np.float32)
        val_rr = np.random.randn(n_val, 2).astype(np.float32)
        val_y = np.random.randint(0, self.num_classes, size=(n_val,))

        test_1d = np.random.randn(n_test, self.seq_len).astype(np.float32)
        test_2d = np.random.randn(n_test, 16, 16).astype(np.float32)
        test_rr = np.random.randn(n_test, 2).astype(np.float32)
        test_y = np.random.randint(0, self.num_classes, size=(n_test,))

        self.train_ds = ArrhythmiaDataset(train_2d, train_1d, train_rr, train_y)
        self.val_ds = ArrhythmiaDataset(val_2d, val_1d, val_rr, val_y)
        self.test_ds = ArrhythmiaDataset(test_2d, test_1d, test_rr, test_y)

        self.test_config = {
            "data": {
                "num_classes": 5,
                "classes": {0: "N", 1: "SVEB", 2: "VEB", 3: "F", 4: "Q"}
            },
            "training": {
                "checkpoint_dir": "./scratch/test_checkpoints"
            }
        }

    def test_model_c_forward_and_params(self):
        """Test Model C forward pass and parameter count ~38.8k."""
        model = Hybrid1DBiCNNGRU(
            in_channels=1,
            num_classes=5,
            conv_out_channels=32,
            conv_kernel_size=5,
            gru_hidden_size=64,
            gru_num_layers=1,
            dropout=0.3
        ).to(self.device)

        params = model.get_num_parameters()
        print(f"\n[Test] Model C parameter count: {params:,}")
        self.assertGreater(params, 30000)
        self.assertLess(params, 50000)

        # Test forward pass with (B, 1, L)
        dummy_x = torch.randn(8, 1, self.seq_len).to(self.device)
        out = model(dummy_x)
        self.assertEqual(out.shape, (8, 5))

        # Test forward pass with (B, L)
        dummy_x2 = torch.randn(8, self.seq_len).to(self.device)
        out2 = model(dummy_x2)
        self.assertEqual(out2.shape, (8, 5))

    def test_focal_loss_and_factory(self):
        """Test Focal Loss and Weighted Cross-Entropy."""
        inputs = torch.randn(8, 5).to(self.device)
        targets = torch.randint(0, 5, (8,)).to(self.device)
        weights = torch.tensor([1.0, 10.0, 3.0, 10.0, 50.0], dtype=torch.float32).to(self.device)

        focal = create_loss_function("focal", gamma=2.5, class_weights=weights, device=self.device)
        fl_loss = focal(inputs, targets)
        self.assertTrue(torch.isfinite(fl_loss))
        self.assertGreater(fl_loss.item(), 0.0)

        wce = create_loss_function("weighted_ce", class_weights=weights, device=self.device)
        wce_loss = wce(inputs, targets)
        self.assertTrue(torch.isfinite(wce_loss))
        self.assertGreater(wce_loss.item(), 0.0)

    def test_cached_dataloaders(self):
        """Test build_cached_dataloaders with oversampling."""
        train_l, val_l, test_l, weights = build_cached_dataloaders(
            self.train_ds, self.val_ds, self.test_ds,
            batch_size=16, use_oversampling=True
        )
        self.assertEqual(len(weights), 5)
        batch = next(iter(train_l))
        m2d, s1d, rr, y = batch
        self.assertEqual(s1d.shape, (16, 1, self.seq_len))
        self.assertEqual(y.shape, (16,))

    def test_optuna_objective_and_pruner(self):
        """Test Optuna Objective end-to-end with 2 trials and MedianPruner."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        objective = ModelCObjective(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            config=self.test_config,
            device=self.device,
            epochs_per_trial=2,
            batch_size=16,
            best_model_save_dir=Path("./scratch/test_checkpoints")
        )

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
        )

        study.optimize(objective, n_trials=2)

        self.assertEqual(len(study.trials), 2)
        self.assertIsNotNone(study.best_value)
        self.assertIn("conv_out_channels", study.best_trial.params)
        self.assertIn("conv_kernel_size", study.best_trial.params)
        self.assertIn("gru_hidden_size", study.best_trial.params)
        self.assertIn("gru_num_layers", study.best_trial.params)
        self.assertIn("dropout", study.best_trial.params)
        self.assertIn("lr", study.best_trial.params)
        self.assertIn("loss_type", study.best_trial.params)
        print(f"\n[✓] Optuna test passed! Best Trial #{study.best_trial.number}: {study.best_value:.2f}% Macro F1")


if __name__ == "__main__":
    unittest.main()
