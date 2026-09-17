import os
import sys
import torch
import unittest
from pathlib import Path

# Add project root directory
sys.path.append(os.getcwd())

from src.models.transfer_ecg_model import ECGTransferModel


class TestTransferLearningPipeline(unittest.TestCase):

    def setUp(self):
        self.backbone_path = "checkpoints/backbone_1d_cnn_pretrained.pth"

    def test_transfer_model_frozen_backbone(self):
        model = ECGTransferModel(
            backbone_path=self.backbone_path,
            num_target_classes=5,
            freeze_backbone=True
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        print(f"\n[Test Frozen] Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}")
        
        # Verify backbone parameters are frozen
        self.assertEqual(frozen_params, 91392)
        self.assertLess(trainable_params, total_params)

        # Test forward pass shape
        dummy_input = torch.randn(4, 1, 256)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([4, 5]))

    def test_transfer_model_custom_classes(self):
        # Test transfer learning to a new target dataset with 12 classes (e.g. PTB-XL)
        model = ECGTransferModel(
            backbone_path=self.backbone_path,
            num_target_classes=12,
            freeze_backbone=True
        )

        dummy_input = torch.randn(2, 1, 256)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([2, 12]))
        print(f"[Test Custom Classes] Output shape for 12 classes: {output.shape}")


if __name__ == "__main__":
    unittest.main()
