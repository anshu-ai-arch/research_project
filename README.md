# ECG Arrhythmia Classification System (Paper Replication & Modular Framework)

A PyTorch implementation and extension of the research paper **"A new hybrid strategy for arrhythmia classification detection using (CNN-LSTM-GRU)"**. 

This repository provides a modular, plug-and-play deep learning framework for ECG arrhythmia detection using the PhysioNet MIT-BIH Arrhythmia Database. It supports both **2D Matrix Reshaped Processing (Paper Replication)** and **1D Continuous Waveform Convolutions (Novel Extension)** without modifying data or evaluation pipelines.

---

## 🌟 Key Features

- **Paper Replication**: Implements the 4-phase methodology from the research paper (Dataset Ingestion $\to$ Preprocessing $\to$ Hybrid Deep Learning Model $\to$ Prediction & Evaluation).
- **1D Continuous Waveform Convolutions (Extension)**: Operates directly on raw 1D ECG sequences ($1280$ points) via 1D convolutions (`Conv1d`), preserving uninterrupted wave morphology while reducing parameters by **$88\%$**.
- **Exact Preprocessing Pipeline**:
  - 4th-order zero-phase Butterworth Bandpass Filtering ($0.5\,\text{Hz} - 50\,\text{Hz}$).
  - Resampling to $128\,\text{Hz}$.
  - 10-second segmentation ($1280$ samples).
  - Min-Max amplitude scaling $[0, 1]$.
  - 2D Matrix Reshaping ($1280 \to 40 \times 32$) + 1D Raw Sequence mode.
- **Modular Model Registry (`ModelFactory`)**:
  - Dynamically register and switch architectures (`cnn`, `lstm`, `gru`, `hybrid_cnn_lstm_gru`, `hybrid_1d_cnn_lstm_gru`, or custom user models) via simple string configuration.

---

## 📊 Benchmark Performance Results (5-Model Comparison)

Tested on the PhysioNet MIT-BIH Arrhythmia Database across 5 arrhythmia classes: *Normal Rhythm (N)*, *Atrial Fibrillation (A)*, *Premature Ventricular Contractions (PVC)*, *Ventricular Tachycardia (VT)*, and *Other Arrhythmias*.

| Model Architecture | Input Format | Parameters | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) | Cohen's Kappa |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 2D CNN** | 2D Matrix ($40 \times 32$) | 675,141 | 97.41% | 97.65% | 99.34% | 97.60% | 0.9668 |
| **Baseline LSTM** | 1D Sequence ($1280$) | 158,085 | 88.52% | 88.40% | 97.07% | 88.36% | 0.8526 |
| **Baseline GRU** | 1D Sequence ($1280$) | 120,709 | 94.44% | 94.57% | 98.58% | 94.47% | 0.9287 |
| **Hybrid 2D CNN-LSTM-GRU (Paper)** | 2D Matrix ($40 \times 32$) | 572,037 | 97.78% | 98.15% | 99.42% | 98.08% | 0.9715 |
| **Hybrid 1D CNN-LSTM-GRU (Novel 1D)** | **Raw Continuous 1D ($1280$)** | **68,357** | **99.63%** | **99.63%** | **99.89%** | **99.69%** | **0.9952** |

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

### 3. Train Paper 2D Model or Novel 1D Model
```bash
# Paper 2D Reshaped Model
python main.py --train --model hybrid_cnn_lstm_gru

# Novel 1D Continuous Waveform Model
python main.py --train --model hybrid_1d_cnn_lstm_gru
```

### 4. Run Multi-Model Comparison Benchmark
Trains and evaluates all 5 models side-by-side:
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
│   │   ├── hybrid_cnn_lstm_gru.py    # Paper Proposed 2D Hybrid Model
│   │   └── hybrid_1d_cnn_lstm_gru.py # Novel 1D Continuous Hybrid Model
│   ├── engine/
│   │   ├── trainer.py          # Training loop with PyTorch MPS/CUDA support
│   │   └── evaluate.py         # Evaluation & Confusion Matrix generator
│   └── utils/
│       └── metrics.py          # Paper metrics (Accuracy, Recall, Specificity, F1, Kappa)
└── results/                    # Saved confusion matrix heatmaps
```
