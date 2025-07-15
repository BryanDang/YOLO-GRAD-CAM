#!/usr/bin/env python3
"""Complete test of YOLO-GRAD-CAM library."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing YOLO-GRAD-CAM Library...")

# Test 1: Basic imports
try:
    from yolocam import YoloCAMAnalyzer, YoloCAMConfig
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Create config
try:
    config = YoloCAMConfig(device='cpu', target_layer_component='backbone')
    print("✅ Config created")
except Exception as e:
    print(f"❌ Config failed: {e}")
    exit(1)

# Test 3: Create analyzer (using dummy model path for now)
model_path = "/content/drive/MyDrive/0 Projects/YOLOv8n_Wound_Segmentation/results/train/weights/best.pt"
if not os.path.exists(model_path):
    # For local testing, create a dummy file
    model_path = "test_model.pt"
    print(f"⚠️  Model not found, using dummy path: {model_path}")

try:
    analyzer = YoloCAMAnalyzer(model_path, config=config)
    print("✅ Analyzer created")
except Exception as e:
    print(f"❌ Analyzer creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Check analyze_single_image exists
if hasattr(analyzer, 'analyze_single_image'):
    print("✅ analyze_single_image method exists")
else:
    print("❌ analyze_single_image method missing")
    exit(1)

# Test 5: Create test image
from PIL import Image
import numpy as np

test_image_path = "test_image.jpg"
test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
test_image.save(test_image_path)
print(f"✅ Test image created: {test_image_path}")

# Test 6: Run analyze_single_image
try:
    result = analyzer.analyze_single_image(test_image_path)
    print("✅ analyze_single_image executed successfully")
    print(f"   Result keys: {list(result.keys())}")
    if 'cam_output' in result:
        print(f"   CAM output shape: {result['cam_output'].shape}")
except Exception as e:
    print(f"❌ analyze_single_image failed: {e}")
    import traceback
    traceback.print_exc()
    
# Clean up
if os.path.exists(test_image_path):
    os.remove(test_image_path)
    
print("\nTest complete!")