"""Setup script for Google Colab."""

def setup_colab():
    """Setup everything needed for Colab."""
    
    print("🔧 Setting up Grad-CAM YOLO in Colab...")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    import subprocess
    import sys
    
    packages = [
        'ultralytics',
        'opencv-python',
        'matplotlib'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    print("✅ Dependencies installed")
    
    # Create directory structure
    import os
    dirs = ['src', 'tests', 'debug', 'debug/output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create __init__.py
    open('src/__init__.py', 'a').close()
    
    print("✅ Directory structure created")
    
    # Copy code files
    print("\n📄 Creating source files...")
    
    # Create all the necessary files inline
    create_source_files()
    
    print("✅ Source files created")
    print("\n🎉 Setup complete! You can now run the test.")
    print("\nUsage:")
    print("```python")
    print("from notebook_test import test_gradcam_in_notebook")
    print("results = test_gradcam_in_notebook()")
    print("```")


def create_source_files():
    """Create all source files inline for Colab."""
    
    # yolo_wrapper.py
    yolo_wrapper_code = '''"""YOLOv8 wrapper that exposes PyTorch layers for Grad-CAM."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image


class YOLOWrapper:
    """Wrapper for YOLOv8 that provides access to PyTorch layers."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'auto'):
        """Initialize YOLO wrapper."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.task = self._detect_task()
        
    def _detect_task(self) -> str:
        """Detect the task type from model."""
        model_name = str(self.model_path).lower()
        if 'seg' in model_name:
            return 'segmentation'
        elif 'cls' in model_name:
            return 'classification'
        elif 'pose' in model_name:
            return 'pose'
        else:
            return 'detection'
    
    def predict(self, image: Any, **kwargs) -> Any:
        """Run YOLO prediction on image."""
        return self.yolo(image, **kwargs)
    
    def get_target_layer(self, layer_type: str = 'backbone') -> nn.Module:
        """Get target layer for Grad-CAM."""
        sequential_model = self.model.model
        
        if layer_type == 'backbone':
            return sequential_model[9]
        elif layer_type == 'neck':
            return sequential_model[12]
        elif layer_type == 'head':
            if self.task == 'segmentation':
                return sequential_model[21]
            else:
                return sequential_model[-1]
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def preprocess_image(self, image_path: str, size: int = 640) -> torch.Tensor:
        """Preprocess image for model input."""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((size, size))
        image_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
'''
    
    with open('src/yolo_wrapper.py', 'w') as f:
        f.write(yolo_wrapper_code)
    
    # gradcam_wrapper.py
    gradcam_wrapper_code = '''"""Grad-CAM implementation using PyTorch hooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, List
import cv2


class GradCAMWrapper:
    """Grad-CAM implementation with forward/backward hooks."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Initialize Grad-CAM wrapper."""
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, 
                     target_class: Optional[int] = None,
                     target_fn: Optional[Callable] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_fn is not None:
            target = target_fn(output)
        elif isinstance(output, (list, tuple)):
            if len(output) > 0 and hasattr(output[0], 'shape'):
                pred_tensor = output[0] if torch.is_tensor(output[0]) else output[0].pred
                if len(pred_tensor.shape) == 3:
                    confidences = pred_tensor[0, :, 4]
                    target = confidences.max()
                else:
                    target = pred_tensor.mean()
            else:
                target = output[0].mean()
        else:
            if target_class is not None:
                target = output[0, target_class]
            else:
                target = output.max()
        
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients")
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.cpu().numpy()
        
        input_size = input_tensor.shape[-2:]
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        
        return cam
    
    def overlay_cam_on_image(self, image: np.ndarray, cam: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        """Overlay CAM heatmap on image."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        overlay = (1 - alpha) * image + alpha * heatmap
        return overlay
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()
'''
    
    with open('src/gradcam_wrapper.py', 'w') as f:
        f.write(gradcam_wrapper_code)
    
    # utils.py
    utils_code = '''"""Utility functions for creating test images."""

import numpy as np
from PIL import Image
import os


def create_sample_image(output_path: str, size: tuple = (640, 640)):
    """Create a simple test image with shapes."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    img[100:250, 100:300] = [255, 50, 50]
    img[300:450, 350:500] = [50, 255, 50]
    img[200:350, 400:550] = [50, 50, 255]
    Image.fromarray(img).save(output_path)
    return output_path
'''
    
    with open('src/utils.py', 'w') as f:
        f.write(utils_code)
    
    # Copy notebook_test.py content
    notebook_test_code = '''"""Notebook-friendly test script for Grad-CAM on YOLO."""

def test_gradcam_in_notebook():
    """Test function designed for notebook environments."""
    
    print("📦 Checking dependencies...")
    
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
    
    print("\\n🚀 Running Grad-CAM test...")
    
    from src.yolo_wrapper import YOLOWrapper
    from src.gradcam_wrapper import GradCAMWrapper
    from src.utils import create_sample_image
    import numpy as np
    from PIL import Image
    import os
    
    os.makedirs("debug", exist_ok=True)
    image_path = "debug/test_image.jpg"
    create_sample_image(image_path)
    print(f"✅ Created test image: {image_path}")
    
    print("\\n📋 Testing YOLO wrapper...")
    yolo = YOLOWrapper('yolov8n.pt', device='cpu')
    print(f"  Model: {yolo.model_path}")
    print(f"  Task: {yolo.task}")
    
    results = yolo.predict(image_path, verbose=False)
    print(f"  Predictions: {len(results[0].boxes)} objects detected")
    
    print("\\n🔥 Testing Grad-CAM...")
    input_tensor = yolo.preprocess_image(image_path)
    print(f"  Input shape: {input_tensor.shape}")
    
    target_layer = yolo.get_target_layer('backbone')
    print(f"  Target layer: {target_layer.__class__.__name__}")
    
    gradcam = GradCAMWrapper(yolo.model, target_layer)
    
    def yolo_target_fn(output):
        if isinstance(output, (list, tuple)):
            pred = output[0]
            if len(pred.shape) == 3:
                return pred[0, :, 4].max()
            return pred.mean()
        return output.mean()
    
    cam = gradcam.generate_cam(input_tensor, target_fn=yolo_target_fn)
    print(f"  CAM generated: shape={cam.shape}, range=[{cam.min():.2f}, {cam.max():.2f}]")
    
    original_img = np.array(Image.open(image_path))
    overlay = gradcam.overlay_cam_on_image(original_img, cam, alpha=0.5)
    
    os.makedirs("debug/output", exist_ok=True)
    overlay_path = "debug/output/notebook_gradcam.jpg"
    Image.fromarray((overlay * 255).astype(np.uint8)).save(overlay_path)
    print(f"  Saved overlay: {overlay_path}")
    
    print("\\n✅ Test completed successfully!")
    print("📁 Check debug/output/ for results")
    
    return {
        'success': True,
        'cam': cam,
        'overlay': overlay,
        'results': results
    }
'''
    
    with open('notebook_test.py', 'w') as f:
        f.write(notebook_test_code)


if __name__ == "__main__":
    setup_colab()