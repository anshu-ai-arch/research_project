import argparse
import yaml
from pathlib import Path

from src.data.download_dataset import download_mitdb
from src.data.dataset import get_dataloaders
from src.models.factory import ModelFactory
from src.engine.trainer import Trainer, get_device
from src.engine.evaluate import evaluate_checkpoint

# Ensure all models register into ModelFactory
import src.models.cnn_model
import src.models.lstm_model
import src.models.gru_model
import src.models.hybrid_cnn_lstm_gru
import src.models.hybrid_1d_cnn_lstm_gru
import src.models.hybrid_cnn_gru
import src.models.hybrid_1d_cnn_gru
import src.models.hybrid_1d_bi_cnn_gru
import src.models.hybrid_1d_multiscale_se_bigru
import src.models.hybrid_1d_cnn_transformer
import src.models.hybrid_1d_cnn_transformer_fusion


def main():
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Classification System (Paper Replication & Swappable Architectures)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--download", action="store_true", help="Download PhysioNet MIT-BIH records")
    parser.add_argument("--model", type=str, default=None, help="Model name (cnn, lstm, gru, hybrid_cnn_lstm_gru, hybrid_1d_cnn_lstm_gru, hybrid_cnn_gru, hybrid_1d_cnn_gru)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--train", action="store_true", help="Train specified model")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization for Model C")
    parser.add_argument("--n_trials", type=int, default=15, help="Number of Optuna trials to run")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model checkpoint")
    parser.add_argument("--compare_all", action="store_true", help="Train and benchmark all registered models side-by-side")
    parser.add_argument("--no_preprocess", action="store_true", help="Bypass bandpass filter and train directly on raw ECG signals")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.no_preprocess:
        config["data"]["use_filtering"] = False
        print("[!] RAW ECG MODE ACTIVATED: Bandpass filtering is BYPASSED.")

    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    if args.download:
        download_mitdb(args.config)

    if args.compare_all:
        print("\n" + "=" * 70)
        print("          RUNNING COMPLETE MULTI-MODEL BENCHMARK COMPARISON         ")
        print("=" * 70)
        
        train_loader, val_loader, test_loader, cfg = get_dataloaders(args.config)
        models_to_test = [
            "cnn",
            "lstm",
            "gru",
            "hybrid_cnn_lstm_gru",
            "hybrid_1d_cnn_lstm_gru",
            "hybrid_cnn_gru",
            "hybrid_1d_cnn_gru"
        ]
        summary_results = []

        for m_name in models_to_test:
            cfg["model"]["name"] = m_name
            print(f"\n\n{'='*30} BENCHMARKING MODEL: {m_name.upper()} {'='*30}")
            model = ModelFactory.create(m_name, **cfg["model"])
            trainer = Trainer(model, cfg)
            ckpt_path = trainer.fit(train_loader, val_loader)

            print(f"\n[*] Evaluating '{m_name}' on Test Set...")
            metrics = evaluate_checkpoint(ckpt_path, test_loader)
            summary_results.append({
                "Model": model.get_model_name(),
                "Parameters": f"{model.get_num_parameters():,}",
                "Accuracy (%)": f"{metrics['accuracy']:.2f}%",
                "Sensitivity (%)": f"{metrics['macro_sensitivity']:.2f}%",
                "Specificity (%)": f"{metrics['macro_specificity']:.2f}%",
                "F1-Score (%)": f"{metrics['macro_f1']:.2f}%",
                "Cohen's Kappa": f"{metrics['cohen_kappa']:.4f}"
            })

        print("\n\n" + "=" * 80)
        print("                     FINAL MULTI-MODEL COMPARISON TABLE                     ")
        print("=" * 80)
        header = f"{'Model Name':<20} | {'Params':<10} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'F1-Score':<10} | {'Kappa':<8}"
        print(header)
        print("-" * 80)
        for r in summary_results:
            print(f"{r['Model']:<20} | {r['Parameters']:<10} | {r['Accuracy (%)']:<10} | {r['Sensitivity (%)']:<12} | {r['Specificity (%)']:<12} | {r['F1-Score (%)']:<10} | {r['Cohen\'s Kappa']:<8}")
        print("=" * 80 + "\n")
        return

    if args.optimize:
        from src.optimization.optuna_pipeline import run_optuna_study
        epochs = args.epochs if args.epochs else 10
        print("\n[*] Initializing Model C Optuna Optimization Pipeline...")
        run_optuna_study(
            config_path=args.config,
            n_trials=args.n_trials,
            epochs_per_trial=epochs
        )
        return

    if args.train:
        train_loader, val_loader, test_loader, cfg = get_dataloaders(args.config)
        model_name = cfg["model"]["name"]
        model = ModelFactory.create(model_name, **cfg["model"])

        trainer = Trainer(model, cfg)
        ckpt_path = trainer.fit(train_loader, val_loader)

        if args.evaluate or True:  # Auto evaluate after training
            print("\n[*] Evaluating best model checkpoint on Test Set...")
            evaluate_checkpoint(ckpt_path, test_loader)

    elif args.evaluate:
        _, _, test_loader, cfg = get_dataloaders(args.config)
        model_name = cfg["model"]["name"]
        ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / f"{model_name.lower()}_best.pth"
        evaluate_checkpoint(str(ckpt_path), test_loader)
    else:
        if not args.download:
            parser.print_help()


if __name__ == "__main__":
    main()
