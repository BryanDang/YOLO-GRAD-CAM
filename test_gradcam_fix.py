#!/usr/bin/env python3
"""Test script to verify Grad-CAM fixes for YOLO segmentation."""

import sys
import os
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from yolocam import YoloCAMAnalyzer, YoloCAMConfig

def test_single_image():
    """Test with a single image to isolate issues."""
    print("="*60)
    print("TESTING GRAD-CAM FIX WITH SINGLE IMAGE")
    print("="*60)
    
    # Paths - update these to match your setup
    model_path = "/content/drive/MyDrive/0 Projects/YOLOv8n_Wound_Segmentation/results/train/weights/best.pt"
    test_image = "./dataset/images/valid/fusc_1466.png"  # Use the problematic image
    test_mask = "./dataset/masks/valid/fusc_1466.png"
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    if not os.path.exists(test_mask):
        print(f"Test mask not found: {test_mask}")
        return
        
    print(f"Model: {model_path}")
    print(f"Test image: {test_image}")
    print(f"Test mask: {test_mask}")
    
    # Test each layer component
    for layer in ['backbone', 'neck', 'head']:
        print(f"\n{'='*40}")
        print(f"Testing {layer.upper()}")
        print('='*40)
        
        try:
            # Create config
            config = YoloCAMConfig(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                target_layer_component=layer,
                save_visualizations=False,
                log_level='DEBUG'
            )
            
            # Create analyzer
            analyzer = YoloCAMAnalyzer(model_path, config=config)
            print("✓ Analyzer created")
            
            # Test inference model
            print("\nTesting model inference...")
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model(test_image, verbose=False)
            
            print(f"Results type: {type(results)}")
            print(f"Results[0] type: {type(results[0])}")
            
            if hasattr(results[0], 'masks'):
                if results[0].masks is not None:
                    print(f"✓ Masks found: shape = {results[0].masks.data.shape}")
                else:
                    print("✗ No masks detected in image")
            else:
                print("✗ Results object has no masks attribute")
            
            # Test the task handler directly
            print("\nTesting task handler...")
            task_handler = analyzer.task_handler
            
            # Load ground truth
            gt = task_handler.load_ground_truth(test_mask)
            print(f"✓ Ground truth loaded: shape = {gt.shape}")
            
            # Get prediction
            prediction = results[0]
            
            # Test metric computation
            print("\nTesting metric computation...")
            try:
                score = task_handler.compute_performance_metric(prediction, gt)
                print(f"✓ IoU score: {score:.3f}")
            except Exception as e:
                print(f"✗ Metric computation failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n✓ {layer} layer test completed successfully")
            
        except Exception as e:
            print(f"\n✗ {layer} layer test failed: {e}")
            import traceback
            traceback.print_exc()

def test_full_validation():
    """Test with full validation set."""
    print("\n" + "="*60)
    print("TESTING WITH FULL VALIDATION SET")
    print("="*60)
    
    model_path = "/content/drive/MyDrive/0 Projects/YOLOv8n_Wound_Segmentation/results/train/weights/best.pt"
    
    config = YoloCAMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        target_layer_component='backbone',
        num_best_examples=3,
        num_worst_examples=3,
        save_visualizations=True,
        output_dir='./gradcam_test_results'
    )
    
    try:
        analyzer = YoloCAMAnalyzer(model_path, config=config)
        
        # Run analysis
        results = analyzer.analyze_performance(
            "./dataset/images/valid",
            "./dataset/masks/valid"
        )
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"Processed {len(results)} images")
        
        if results:
            scores = [r['score'] for r in results]
            print(f"IoU range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"Average IoU: {sum(scores)/len(scores):.3f}")
            
    except Exception as e:
        print(f"\n✗ Full validation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # First test with single image to debug
    test_single_image()
    
    # If single image works, test full validation
    # test_full_validation()