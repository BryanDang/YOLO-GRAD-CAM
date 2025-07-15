"""Notebook-friendly test script for Grad-CAM on YOLO."""

def test_gradcam_in_notebook():
    """Test function designed for notebook environments."""
    
    print("📦 Checking dependencies...")
    
    # Check imports
    try:
        import torch
        print("✅ PyTorch installed")
    except ImportError:
        print("❌ PyTorch not found. Install with: pip install torch")
        return
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics installed")
    except ImportError:
        print("❌ Ultralytics not found. Install with: pip install ultralytics")
        return
        
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        return
    
    print("\n🚀 Running Grad-CAM test...")
    
    # Import our modules
    from src.yolo_wrapper import YOLOWrapper
    from src.gradcam_wrapper import GradCAMWrapper
    from src.utils import create_sample_image
    import numpy as np
    from PIL import Image
    import os
    
    # Create test image
    os.makedirs("debug", exist_ok=True)
    image_path = "debug/test_image.jpg"
    create_sample_image(image_path)
    print(f"✅ Created test image: {image_path}")
    
    # Test YOLO wrapper
    print("\n📋 Testing YOLO wrapper...")
    yolo = YOLOWrapper('yolov8n.pt', device='cpu')
    print(f"  Model: {yolo.model_path}")
    print(f"  Task: {yolo.task}")
    
    # Run prediction
    results = yolo.predict(image_path, verbose=False)
    print(f"  Predictions: {len(results[0].boxes)} objects detected")
    
    # Test Grad-CAM
    print("\n🔥 Testing Grad-CAM...")
    input_tensor = yolo.preprocess_image(image_path)
    print(f"  Input shape: {input_tensor.shape}")
    
    # Get target layer
    target_layer = yolo.get_target_layer('backbone')
    print(f"  Target layer: {target_layer.__class__.__name__}")
    
    # Create Grad-CAM
    gradcam = GradCAMWrapper(yolo.model, target_layer)
    
    # Define YOLO target function
    def yolo_target_fn(output):
        if isinstance(output, (list, tuple)):
            pred = output[0]
            if len(pred.shape) == 3:
                return pred[0, :, 4].max()
            return pred.mean()
        return output.mean()
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor, target_fn=yolo_target_fn)
    print(f"  CAM generated: shape={cam.shape}, range=[{cam.min():.2f}, {cam.max():.2f}]")
    
    # Create overlay
    original_img = np.array(Image.open(image_path))
    overlay = gradcam.overlay_cam_on_image(original_img, cam, alpha=0.5)
    
    # Save results
    os.makedirs("debug/output", exist_ok=True)
    overlay_path = "debug/output/notebook_gradcam.jpg"
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    print(f"  Saved overlay: {overlay_path}")
    
    print("\n✅ Test completed successfully!")
    print("📁 Check debug/output/ for results")
    
    return {
        'success': True,
        'cam': cam,
        'overlay': overlay,
        'results': results
    }


# For easy import in notebook
if __name__ == "__main__":
    test_gradcam_in_notebook()