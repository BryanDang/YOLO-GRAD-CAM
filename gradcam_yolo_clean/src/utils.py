"""Utility functions for creating test images."""

import numpy as np
from PIL import Image
import os


def create_sample_image(output_path: str, size: tuple = (640, 640)):
    """Create a simple test image with shapes."""
    # Create a blank image
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles
    # Red rectangle
    img[100:250, 100:300] = [255, 50, 50]
    
    # Green rectangle  
    img[300:450, 350:500] = [50, 255, 50]
    
    # Blue circle (approximated with square)
    img[200:350, 400:550] = [50, 50, 255]
    
    # Save image
    Image.fromarray(img).save(output_path)
    return output_path