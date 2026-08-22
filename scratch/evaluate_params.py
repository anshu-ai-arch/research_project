import torch
import yaml
from src.models.factory import ModelFactory
import src.models.cnn_model
import src.models.lstm_model
import src.models.gru_model
import src.models.hybrid_cnn_lstm_gru
import src.models.hybrid_1d_cnn_lstm_gru
import src.models.hybrid_cnn_gru
import src.models.hybrid_1d_cnn_gru

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

models = [
    "cnn",
    "lstm",
    "gru",
    "hybrid_cnn_lstm_gru",
    "hybrid_1d_cnn_lstm_gru",
    "hybrid_cnn_gru",
    "hybrid_1d_cnn_gru"
]

print("==================================================================================")
print("                   EXACT MODEL PARAMETER COUNTS (BEAT MODE)                       ")
print("==================================================================================")

for m_name in models:
    model = ModelFactory.create(m_name, **config["model"])
    num_params = model.get_num_parameters()
    print(f"Model: {m_name:<25} | Parameters: {num_params:,}")

print("==================================================================================")
