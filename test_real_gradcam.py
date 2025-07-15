#!/usr/bin/env python3
"""Test Grad-CAM with real validation images."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from yolocam import YoloCAMAnalyzer, YoloCAMConfig
import torch

print("="*60)
print("TESTING GRAD-CAM WITH REAL VALIDATION DATA")
print("="*60)

# Paths
model_path = "/content/drive/MyDrive/0 Projects/YOLOv8n_Wound_Segmentation/results/train/weights/best.pt"
valid_images = "./dataset/images/valid"
valid_masks = "./dataset/masks/valid"

# Find first few validation images
import os
from pathlib import Path

image_files = list(Path(valid_images).glob("*.png"))[:5]
print(f"\nFound {len(image_files)} test images")

# Test with backbone layer
config = YoloCAMConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    target_layer_component='backbone',
    save_visualizations=True,
    output_dir='./gradcam_debug',
    log_level='DEBUG',
    verbose=True
)

try:
    print("\nCreating analyzer...")
    analyzer = YoloCAMAnalyzer(model_path, config=config)
    print("✅ Analyzer created")
    
    # Test single image analysis
    test_image = str(image_files[0])
    test_mask = str(Path(valid_masks) / image_files[0].name)
    
    print(f"\nTesting with image: {test_image}")
    print(f"Mask: {test_mask}")
    
    try:
        result = analyzer.analyze_single_image(test_image, test_mask, visualize=True)
        print("✅ Single image analysis successful!")
        
        if 'score' in result:
            print(f"IoU Score: {result['score']:.3f}")
        if 'visualization_path' in result:
            print(f"Visualization saved to: {result['visualization_path']}")
            
    except Exception as e:
        print(f"❌ Single image analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test batch analysis
    print("\n" + "="*40)
    print("Testing batch analysis...")
    
    try:
        results = analyzer.analyze_performance(valid_images, valid_masks)
        print(f"✅ Batch analysis successful! Processed {len(results)} images")
        
        if results:
            scores = [r['score'] for r in results]
            print(f"IoU range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"Average IoU: {sum(scores)/len(scores):.3f}")
            
            # Visualize best and worst
            best = analyzer.get_best_examples(results, num_examples=3)
            worst = analyzer.get_worst_examples(results, num_examples=3)
            
            print(f"\nVisualizing {len(best)} best and {len(worst)} worst examples...")
            analyzer.visualize_results(best, "best")
            analyzer.visualize_results(worst, "worst")
            
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Failed to create analyzer: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")