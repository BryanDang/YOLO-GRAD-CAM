#!/usr/bin/env python3
"""Test script to verify gradient computation fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from yolocam import YoloCAMAnalyzer, YoloCAMConfig

# Test configuration
config = YoloCAMConfig(
    device='cpu',
    target_layer_component='backbone',
    # Add target_layer_index if needed
)

# Initialize analyzer
model_path = 'yolov8n-seg.pt'  # Will use dummy if not exists
if not os.path.exists(model_path):
    model_path = '/content/drive/MyDrive/0 Projects/YOLOv8n_Wound_Segmentation/results/train/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Warning: Model not found, using placeholder path")
        model_path = 'dummy_model.pt'

try:
    analyzer = YoloCAMAnalyzer(model_path, config=config)
    print("✅ Analyzer initialized")
except Exception as e:
    print(f"❌ Failed to initialize analyzer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create test image
test_image = torch.rand(3, 640, 640) * 255
test_image_path = 'test_image.jpg'
from PIL import Image
import numpy as np
Image.fromarray(test_image.numpy().transpose(1, 2, 0).astype('uint8')).save(test_image_path)
print(f"✅ Test image created: {test_image_path}")

# This MUST work without errors
try:
    result = analyzer.analyze_single_image(test_image_path)
    print("SUCCESS: analyze_single_image completed")
    print(f"Result keys: {list(result.keys())}")
    
    # Verify outputs
    assert 'cam' in result, "Missing 'cam' in result"
    assert 'overlay' in result, "Missing 'overlay' in result"
    assert result['cam'] is not None, "CAM is None"
    assert result['overlay'] is not None, "Overlay is None"
    
    # Check shapes
    print(f"CAM shape: {result['cam'].shape}")
    print(f"Overlay shape: {result['overlay'].shape}")
    
    print("\n✅ ALL TESTS PASSED")
    
except Exception as e:
    print(f"\n❌ FAILED: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
finally:
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"Cleaned up {test_image_path}")

print("\nTest complete!")