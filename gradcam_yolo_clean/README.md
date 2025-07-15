# Grad-CAM for YOLOv8

Clean, modular implementation of Grad-CAM for YOLOv8 models.

## Features

- ✅ Clean wrapper for YOLOv8 models (Ultralytics)
- ✅ Grad-CAM implementation using PyTorch hooks
- ✅ Support for different layers (backbone, neck, head)
- ✅ No pip reinstall loops - just edit and run
- ✅ Works in Colab and local environments

## Structure

```
gradcam_yolo_clean/
├── src/
│   ├── __init__.py
│   ├── yolo_wrapper.py      # YOLOv8 wrapper
│   ├── gradcam_wrapper.py   # Grad-CAM implementation
│   └── utils.py             # Utility functions
├── tests/
│   └── test_gradcam_on_yolo.py  # Main test script
├── debug/
│   └── output/              # Generated visualizations
├── cli.py                   # Command line interface
├── requirements.txt
└── README.md
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the test:
```bash
python tests/test_gradcam_on_yolo.py
```

3. Or use the CLI:
```bash
python cli.py --test-layers
```

## Usage in Code

```python
from src.yolo_wrapper import YOLOWrapper
from src.gradcam_wrapper import GradCAMWrapper

# Load YOLO model
yolo = YOLOWrapper('yolov8n.pt')

# Get target layer
target_layer = yolo.get_target_layer('backbone')

# Initialize Grad-CAM
gradcam = GradCAMWrapper(yolo.model, target_layer)

# Generate CAM
input_tensor = yolo.preprocess_image('image.jpg')
cam = gradcam.generate_cam(input_tensor)

# Overlay on image
overlay = gradcam.overlay_cam_on_image(original_img, cam)
```

## For Colab

```python
# No need to restart kernel!
import sys
sys.path.append('/path/to/gradcam_yolo_clean')

from src.yolo_wrapper import YOLOWrapper
from src.gradcam_wrapper import GradCAMWrapper

# Use as above
```

## Outputs

Check `debug/output/` for:
- `gradcam_heatmap.jpg` - Raw Grad-CAM heatmap
- `gradcam_overlay.jpg` - Heatmap overlaid on image
- `yolo_detections.jpg` - YOLO detection results
- `gradcam_backbone.jpg` - Backbone layer CAM
- `gradcam_neck.jpg` - Neck layer CAM
- `gradcam_head.jpg` - Head layer CAM