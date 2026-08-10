# ECG Arrhythmia Classification System (Paper Replication & Modular Framework)

A PyTorch implementation and extension of the research paper **"A new hybrid strategy for arrhythmia classification detection using (CNN-LSTM-GRU)"**. 

This repository provides a modular, plug-and-play deep learning framework for ECG arrhythmia detection using the PhysioNet MIT-BIH Arrhythmia Database. It enables seamless model swapping (CNN, LSTM, GRU, Hybrid CNN-LSTM-GRU, and future architectures) without modifying data or evaluation pipelines.

---

## 🌟 Key Features

- **Paper Replication**: Implements the exact 4-phase methodology from the research paper (Dataset Ingestion $\to$ Preprocessing $\to$ Hybrid Deep Learning Model $\to$ Prediction & Evaluation).
- **Exact Preprocessing Pipeline**:
  - 4th-order zero-phase Butterworth Bandpass Filtering ($0.5\,\text{Hz} - 50\,\text{Hz}$).
  - Resampling to $128\,\text{Hz}$.
  - 10-second segmentation ($1280$ samples).
  - Min-Max amplitude scaling $[0, 1]$.
  - 2D Matrix Reshaping ($1280 \to 40 \times 32$).
- **Modular Model Registry (`ModelFactory`)**:
  - Dynamically register and switch architectures (`cnn`, `lstm`, `gru`, `hybrid_cnn_lstm_gru`, or custom user models) via simple string configuration.
- **Comprehensive Evaluation**: Computes Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Cohen's Kappa, and generates Confusion Matrix heatmaps.

---

## 📊 Benchmark Performance Results

Tested on the PhysioNet MIT-BIH Arrhythmia Database across 5 arrhythmia classes: *Normal Rhythm (N)*, *Atrial Fibrillation (A)*, *Premature Ventricular Contractions (PVC)*, *Ventricular Tachycardia (VT)*, and *Other Arrhythmias*.

| Model Architecture | Parameters | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) | Cohen's Kappa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 2D CNN** | 675,141 | **98.52%** | **98.52%** | **99.62%** | **98.58%** | **0.9810** |
| **Baseline LSTM** | 158,085 | 88.52% | 89.26% | 97.10% | 88.87% | 0.8531 |
| **Baseline GRU** | 120,709 | 93.33% | 93.70% | 98.32% | 93.45% | 0.9146 |
| **Proposed Hybrid (CNN-LSTM-GRU)** | 572,037 | **97.41%** | **97.53%** | **99.31%** | **97.65%** | **0.9667** |

### Per-Class Performance (Hybrid CNN-LSTM-GRU)
- **Normal Rhythm (N)**: 98.77% Sensitivity | 98.41% Specificity | 97.56% F1
- **Atrial Fibrillation (A)**: 94.44% Sensitivity | 99.54% Specificity | 96.23% F1
- **Premature Ventricular Contractions (PVC)**: 96.30% Sensitivity | 99.54% Specificity | 97.20% F1
- **Ventricular Tachycardia (VT)**: **100.00% Sensitivity** | **100.00% Specificity** | **100.00% F1**
- **Other Arrhythmias**: 98.15% Sensitivity | 99.07% Specificity | 97.25% F1

---

## 🛠️ Installation & Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/anshu-ai-arch/research_project.git
cd research_project
pip install -r requirements.txt
```

### 2. Download Dataset
Automatically fetches MIT-BIH Arrhythmia records from PhysioNet:
```bash
python main.py --download
```

### 3. Train Proposed Hybrid Model
```bash
python main.py --train --model hybrid_cnn_lstm_gru
```

### 4. Run Multi-Model Comparison Benchmark
Trains and evaluates CNN, LSTM, GRU, and Hybrid models side-by-side:
```bash
python main.py --compare_all
```

---

## 🏗️ Repository Structure

```
research_project/
├── config.yaml               # Central configuration file
├── main.py                   # Central CLI driver
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── src/
│   ├── data/
│   │   ├── download_dataset.py # Downloads MIT-BIH recordings
│   │   ├── preprocessor.py     # Filter, resample, segment, normalize, 2D reshape
│   │   └── dataset.py          # PyTorch Dataset & DataLoaders
│   ├── models/
│   │   ├── base_model.py       # Abstract Base Model Class (BaseECGModel)
│   │   ├── factory.py          # Central Model Factory Registry
│   │   ├── cnn_model.py        # Baseline 2D CNN
│   │   ├── lstm_model.py       # Baseline LSTM
│   │   ├── gru_model.py        # Baseline GRU
│   │   └── hybrid_cnn_lstm_gru.py # Paper Proposed Hybrid Model
│   ├── engine/
│   │   ├── trainer.py          # Training loop with PyTorch MPS/CUDA support
│   │   └── evaluate.py         # Evaluation & Confusion Matrix generator
│   └── utils/
│       └── metrics.py          # Paper metrics (Accuracy, Recall, Specificity, F1, Kappa)
└── results/                    # Saved confusion matrix heatmaps
```

---

## 🚀 How to Add a New Model Architecture

To test a new architecture (e.g. Mamba, Vision Transformer, ResNet):

1. Create `src/models/my_new_model.py`.
2. Inherit from `BaseECGModel` and register it:
   ```python
   from src.models.base_model import BaseECGModel
   from src.models.factory import ModelFactory

   @ModelFactory.register("my_new_model")
   class MyNewModel(BaseECGModel):
       def forward(self, x):
           ...
   ```
3. Run training with `--model my_new_model`:
   ```bash
   python main.py --train --model my_new_model
   ```
