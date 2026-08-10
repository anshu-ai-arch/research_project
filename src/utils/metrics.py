import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> Dict[str, Any]:
    """
    Computes paper evaluation metrics:
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1-Score
    - Cohen's Kappa Score
    - Confusion Matrix
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Calculate per-class specificity: Specificity = TN / (TN + FP)
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    specificities = np.array(specificities)

    # Macro averages across classes
    macro_precision = np.mean(precision) * 100.0
    macro_sensitivity = np.mean(recall) * 100.0  # Sensitivity is Recall
    macro_specificity = np.mean(specificities) * 100.0
    macro_f1 = np.mean(f1) * 100.0

    metrics = {
        "accuracy": acc,
        "cohen_kappa": kappa,
        "macro_precision": macro_precision,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "per_class_precision": precision * 100.0,
        "per_class_sensitivity": recall * 100.0,
        "per_class_specificity": specificities * 100.0,
        "per_class_f1": f1 * 100.0,
        "confusion_matrix": cm,
    }

    return metrics


def print_metrics_report(metrics: Dict[str, Any], class_names: Dict[int, str]):
    """Prints a nicely formatted evaluation report matching paper tables."""
    print("\n" + "=" * 65)
    print("           ECG ARRHYTHMIA EVALUATION METRICS REPORT          ")
    print("=" * 65)
    print(f" Overall Accuracy:      {metrics['accuracy']:.2f}%")
    print(f" Macro Sensitivity:     {metrics['macro_sensitivity']:.2f}%")
    print(f" Macro Specificity:     {metrics['macro_specificity']:.2f}%")
    print(f" Macro Precision:       {metrics['macro_precision']:.2f}%")
    print(f" Macro F1-Score:        {metrics['macro_f1']:.2f}%")
    print(f" Cohen's Kappa Score:   {metrics['cohen_kappa']:.4f}")
    print("-" * 65)
    print(f"{'Class Name':<35} | {'Sens (%)':<8} | {'Spec (%)':<8} | {'F1 (%)':<8}")
    print("-" * 65)

    for cid, name in class_names.items():
        sens = metrics["per_class_sensitivity"][cid]
        spec = metrics["per_class_specificity"][cid]
        f1 = metrics["per_class_f1"][cid]
        print(f"{name:<35} | {sens:<8.2f} | {spec:<8.2f} | {f1:<8.2f}")

    print("=" * 65 + "\n")
