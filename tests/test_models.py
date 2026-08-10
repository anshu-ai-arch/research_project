import torch
from src.models.factory import ModelFactory
import src.models.cnn_model
import src.models.lstm_model
import src.models.gru_model
import src.models.hybrid_cnn_lstm_gru


def test_registered_models():
    print("[*] Testing registered models in ModelFactory...")
    available = ModelFactory.list_available_models()
    print(f"[*] Available models in registry: {available}")

    dummy_input = torch.randn(8, 1, 40, 32)
    dummy_labels = torch.randint(0, 5, (8,))

    for model_name in available:
        print(f"\n---> Testing model: '{model_name}'")
        model = ModelFactory.create(model_name, num_classes=5)
        print(f" -> Class Name: {model.get_model_name()}")
        print(f" -> Trainable Parameters: {model.get_num_parameters():,}")

        output = model(dummy_input)
        print(f" -> Input Shape: {dummy_input.shape}")
        print(f" -> Output Shape: {output.shape}")

        assert output.shape == (8, 5), f"Expected output shape (8, 5), got {output.shape}"
        print(f" [✓] Model '{model_name}' passed forward pass test!")

    print("\n[✓] ALL MODELS INSTANTIATED AND PASSED SHAPE VERIFICATION!")


if __name__ == "__main__":
    test_registered_models()
