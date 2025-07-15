#!/usr/bin/env python3
"""Test script to verify the library works correctly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic imports
print("Testing imports...")
try:
    from yolocam import YoloCAMAnalyzer, YoloCAMConfig
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test with a simple model
print("\nTesting initialization...")
try:
    # Create config
    config = YoloCAMConfig(
        device='cpu',  # Use CPU for testing
        target_layer_component='backbone'
    )
    print("✓ Config created")
    
    # Try with a default model name (will download if needed)
    analyzer = YoloCAMAnalyzer('yolov8n.pt', config=config)
    print("✓ Analyzer created")
    
    # Check attributes
    print(f"  - Model version: {analyzer.model_version}")
    print(f"  - Task: {analyzer.task_name}")
    print(f"  - Has model handler: {hasattr(analyzer, 'model_handler')}")
    
    # Test model handler
    if hasattr(analyzer, 'model_handler'):
        handler = analyzer.model_handler
        print(f"  - Model handler type: {type(handler).__name__}")
        print(f"  - Has _yolo_model: {hasattr(handler, '_yolo_model')}")
        
        # Test getting pytorch model
        try:
            pytorch_model = handler.pytorch_model
            print(f"  - PyTorch model type: {type(pytorch_model).__name__}")
        except Exception as e:
            print(f"  - Failed to get pytorch model: {e}")
    
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")