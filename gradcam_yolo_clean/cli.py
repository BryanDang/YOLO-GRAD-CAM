#!/usr/bin/env python
"""CLI for testing Grad-CAM on YOLO."""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_gradcam_on_yolo import test_gradcam_yolo, test_different_layers


def main():
    parser = argparse.ArgumentParser(description="Test Grad-CAM on YOLOv8")
    parser.add_argument('--test-layers', action='store_true', 
                       help='Test different layers (backbone, neck, head)')
    parser.add_argument('--model', default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--device', default='cpu',
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Grad-CAM YOLO Test")
    print("="*50)
    
    # Run main test
    success = test_gradcam_yolo()
    
    # Optionally test different layers
    if success and args.test_layers:
        test_different_layers()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())