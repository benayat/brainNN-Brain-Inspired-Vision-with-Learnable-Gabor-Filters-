#!/usr/bin/env python3
"""
Quick test to validate EMNIST dataset integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_universal import make_dataloaders

def test_emnist_datasets():
    """Test all EMNIST splits load correctly."""
    
    emnist_splits = [
        "emnist_letters",
        "emnist_digits", 
        "emnist_balanced",
        "emnist_byclass",
        "emnist_bymerge"
    ]
    
    print("Testing EMNIST Dataset Integration")
    print("=" * 60)
    
    for split in emnist_splits:
        try:
            train_loader, test_loader, in_channels, num_classes = make_dataloaders(
                dataset_name=split,
                image_size=64,
                batch_size=32
            )
            
            # Get first batch
            x_batch, y_batch = next(iter(train_loader))
            
            print(f"✓ {split:20s} | Classes: {num_classes:2d} | "
                  f"Train: {len(train_loader.dataset):6d} | "
                  f"Test: {len(test_loader.dataset):5d} | "
                  f"Shape: {tuple(x_batch.shape)}")
            
        except Exception as e:
            print(f"✗ {split:20s} | ERROR: {str(e)[:40]}")
    
    print("=" * 60)
    print("EMNIST integration test complete!")

if __name__ == "__main__":
    test_emnist_datasets()
