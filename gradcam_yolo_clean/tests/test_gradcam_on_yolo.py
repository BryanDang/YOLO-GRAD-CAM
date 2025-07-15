"""Test Grad-CAM on YOLOv8 model."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.yolo_wrapper import YOLOWrapper
from src.gradcam_wrapper import GradCAMWrapper
from src.utils import create_sample_image


def test_gradcam_yolo():
    """Test Grad-CAM on YOLOv8 detection."""
    print("Starting Grad-CAM test on YOLOv8...")
    
    # Create sample image if it doesn't exist
    image_path = "debug/sample.jpg"
    if not os.path.exists(image_path):
        os.makedirs("debug", exist_ok=True)
        create_sample_image(image_path)
        print(f"Created sample image: {image_path}")
    
    try:
        # 1. Initialize YOLO wrapper
        print("\n1. Loading YOLOv8 model...")
        yolo = YOLOWrapper(model_path='yolov8n.pt', device='cpu')  # Use CPU for testing
        print(f"   Model loaded: {yolo.model_path}")
        print(f"   Task: {yolo.task}")
        print(f"   Device: {yolo.device}")
        
        # 2. Run prediction
        print("\n2. Running YOLO prediction...")
        results = yolo.predict(image_path, verbose=False)
        
        # Check if detections were made
        if len(results[0].boxes) > 0:
            num_detections = len(results[0].boxes)
            print(f"   Found {num_detections} objects")
            
            # Print detection details
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                print(f"   Detection {i}: Class {cls}, Confidence {conf:.3f}")
        else:
            print("   No objects detected")
        
        # 3. Prepare input for Grad-CAM
        print("\n3. Preparing input for Grad-CAM...")
        input_tensor = yolo.preprocess_image(image_path)
        print(f"   Input shape: {input_tensor.shape}")
        
        # 4. Get target layer
        print("\n4. Getting target layer...")
        target_layer = yolo.get_target_layer('backbone')
        print(f"   Target layer: {target_layer.__class__.__name__}")
        
        # 5. Initialize Grad-CAM
        print("\n5. Initializing Grad-CAM...")
        gradcam = GradCAMWrapper(yolo.model, target_layer)
        
        # 6. Generate CAM
        print("\n6. Generating Grad-CAM heatmap...")
        
        # Define target function for YOLO
        def yolo_target_fn(output):
            """Target function for YOLO output."""
            if isinstance(output, (list, tuple)):
                # Get the detection output
                pred = output[0]
                # Use maximum objectness score
                if len(pred.shape) == 3:
                    return pred[0, :, 4].max()  # Max confidence
                else:
                    return pred.mean()
            return output.mean()
        
        cam = gradcam.generate_cam(input_tensor, target_fn=yolo_target_fn)
        print(f"   CAM shape: {cam.shape}")
        print(f"   CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
        
        # 7. Load original image and overlay
        print("\n7. Creating overlay visualization...")
        original_img = np.array(Image.open(image_path))
        overlay = gradcam.overlay_cam_on_image(original_img, cam, alpha=0.5)
        
        # 8. Save results
        output_dir = "debug/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CAM
        cam_path = os.path.join(output_dir, "gradcam_heatmap.jpg")
        plt.figure(figsize=(8, 8))
        plt.imshow(cam, cmap='jet')
        plt.colorbar()
        plt.title("Grad-CAM Heatmap")
        plt.savefig(cam_path)
        plt.close()
        print(f"   Saved CAM heatmap: {cam_path}")
        
        # Save overlay
        overlay_path = os.path.join(output_dir, "gradcam_overlay.jpg")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
        print(f"   Saved overlay: {overlay_path}")
        
        # Save detection results
        results_path = os.path.join(output_dir, "yolo_detections.jpg")
        results[0].save(results_path)
        print(f"   Saved YOLO results: {results_path}")
        
        print("\n✅ Test completed successfully!")
        print(f"   Check output in: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_layers():
    """Test Grad-CAM on different layers."""
    print("\n" + "="*50)
    print("Testing different layers...")
    print("="*50)
    
    image_path = "debug/sample.jpg"
    if not os.path.exists(image_path):
        create_sample_image(image_path)
    
    yolo = YOLOWrapper(model_path='yolov8n.pt', device='cpu')
    input_tensor = yolo.preprocess_image(image_path)
    
    for layer_type in ['backbone', 'neck', 'head']:
        try:
            print(f"\nTesting {layer_type} layer...")
            target_layer = yolo.get_target_layer(layer_type)
            print(f"  Layer: {target_layer.__class__.__name__}")
            
            gradcam = GradCAMWrapper(yolo.model, target_layer)
            
            def yolo_target_fn(output):
                if isinstance(output, (list, tuple)):
                    pred = output[0]
                    if len(pred.shape) == 3:
                        return pred[0, :, 4].max()
                    return pred.mean()
                return output.mean()
            
            cam = gradcam.generate_cam(input_tensor, target_fn=yolo_target_fn)
            
            # Save CAM for this layer
            output_path = f"debug/output/gradcam_{layer_type}.jpg"
            plt.figure(figsize=(6, 6))
            plt.imshow(cam, cmap='jet')
            plt.title(f"Grad-CAM: {layer_type}")
            plt.colorbar()
            plt.savefig(output_path)
            plt.close()
            
            print(f"  ✅ Success! Saved to: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    # Run main test
    success = test_gradcam_yolo()
    
    # If successful, test different layers
    if success:
        test_different_layers()