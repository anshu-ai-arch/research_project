#!/usr/bin/env python3
"""
Model C (hybrid_1d_bi_cnn_gru) Optuna Hyperparameter Optimization Script
Configured for Apple Silicon Mac M1 (MPS) Acceleration & Multi-Class Macro F1 Optimization.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import argparse
from src.optimization.optuna_pipeline import run_optuna_study


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna Optimization Pipeline for Model C (hybrid_1d_bi_cnn_gru) on Apple Silicon Mac M1 (MPS)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of Optuna optimization trials to execute (default: 15)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per trial (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Mini-batch size for DataLoader (default: 64)"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="model_c_hybrid_1d_bi_cnn_gru_opt",
        help="Optuna study name (default: model_c_hybrid_1d_bi_cnn_gru_opt)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional database storage URL (e.g., sqlite:///optuna_study.db)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save optimization results and metrics reports (default: ./results)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    study, test_metrics = run_optuna_study(
        config_path=args.config,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs,
        batch_size=args.batch_size,
        study_name=args.study_name,
        storage=args.storage,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
