"""Pytest configuration and shared fixtures."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from yolocam.core.config import YoloCAMConfig
from yolocam.core.registry import YOLOModelRegistry, TaskRegistry


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config() -> YoloCAMConfig:
    """Create a sample configuration for testing."""
    return YoloCAMConfig(
        device='cpu',
        model_input_size=320,  # Smaller for faster testing
        cam_method='gradcam',
        num_best_examples=2,
        num_worst_examples=2,
        save_visualizations=False,
        save_intermediate_results=False,
        verbose=False,
    )


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    # Create a simple test image
    image = Image.new('RGB', (640, 480), color='red')
    
    # Add some simple patterns to make it interesting
    pixels = np.array(image)
    pixels[100:200, 100:200] = [0, 255, 0]  # Green square
    pixels[300:400, 300:400] = [0, 0, 255]  # Blue square
    
    image = Image.fromarray(pixels)
    image_path = temp_dir / "test_image.jpg"
    image.save(image_path)
    
    return image_path


@pytest.fixture
def sample_mask(temp_dir: Path) -> Path:
    """Create a sample segmentation mask for testing."""
    # Create a binary mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 255  # Mask for green square
    
    mask_image = Image.fromarray(mask, mode='L')
    mask_path = temp_dir / "test_mask.png"
    mask_image.save(mask_path)
    
    return mask_path


@pytest.fixture
def sample_dataset(temp_dir: Path) -> Dict[str, Path]:
    """Create a sample dataset with multiple images and masks."""
    images_dir = temp_dir / "images"
    masks_dir = temp_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    
    # Create multiple test images and masks
    for i in range(5):
        # Create image with random colors
        color = tuple(np.random.randint(0, 255, 3))
        image = Image.new('RGB', (320, 240), color=color)
        
        # Add a random square
        pixels = np.array(image)
        x, y = np.random.randint(50, 200, 2)
        square_color = tuple(np.random.randint(0, 255, 3))
        pixels[y:y+50, x:x+50] = square_color
        
        image = Image.fromarray(pixels)
        image_path = images_dir / f"image_{i:03d}.jpg"
        image.save(image_path)
        
        # Create corresponding mask
        mask = np.zeros((240, 320), dtype=np.uint8)
        mask[y:y+50, x:x+50] = 255
        
        mask_image = Image.fromarray(mask, mode='L')
        mask_path = masks_dir / f"image_{i:03d}.png"
        mask_image.save(mask_path)
    
    return {
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'num_samples': 5,
    }


class MockYOLOModel(nn.Module):
    """Mock YOLO model for testing."""
    
    def __init__(self, task_type: str = 'segmentation'):
        super().__init__()
        self.task_type = task_type
        
        # Simple CNN-like architecture for testing
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((20, 20)),
        )
        
        self.neck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        if task_type == 'segmentation':
            self.head = nn.Conv2d(128, 1, 1)  # Single class segmentation
        elif task_type == 'detection':
            self.head = nn.Conv2d(128, 85, 1)  # 80 classes + 5 (x,y,w,h,conf)
        else:  # classification
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 1000),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    model = MockYOLOModel()
    
    # Create a mock checkpoint with metadata
    checkpoint = {
        'model': model,
        'version': 'yolov8n',
        'task': 'segmentation',
        'epoch': 100,
        'date': '2024-01-01',
    }
    
    checkpoint_path = temp_dir / "mock_model.pt"
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


class MockInferenceResult:
    """Mock inference result for testing."""
    
    def __init__(self, task_type: str = 'segmentation'):
        self.task_type = task_type
        
        if task_type == 'segmentation':
            # Mock segmentation masks
            self.masks = self._create_mock_masks()
        elif task_type == 'detection':
            # Mock detection boxes
            self.boxes = self._create_mock_boxes()
        else:
            # Mock classification results
            self.probs = self._create_mock_probs()
    
    def _create_mock_masks(self):
        """Create mock segmentation masks."""
        class MockMasks:
            def __init__(self):
                # Create a simple mask tensor
                self.data = torch.rand(1, 240, 320) > 0.5
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self.data.numpy()
        
        return MockMasks()
    
    def _create_mock_boxes(self):
        """Create mock detection boxes."""
        class MockBoxes:
            def __init__(self):
                # Create mock bounding boxes [x1, y1, x2, y2, conf, class]
                self.data = torch.tensor([
                    [100, 100, 200, 200, 0.8, 0],
                    [250, 150, 350, 250, 0.6, 1],
                ])
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self.data.numpy()
        
        return MockBoxes()
    
    def _create_mock_probs(self):
        """Create mock classification probabilities."""
        class MockProbs:
            def __init__(self):
                self.data = torch.softmax(torch.randn(1000), dim=0)
            
            def cpu(self):
                return self
            
            def numpy(self):
                return self.data.numpy()
        
        return MockProbs()


@pytest.fixture
def mock_inference_model():
    """Create a mock inference model."""
    def mock_inference(image_path: str, **kwargs):
        return [MockInferenceResult()]
    
    return mock_inference


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before each test to avoid contamination."""
    # Store original registries
    original_models = YOLOModelRegistry._models.copy()
    original_tasks = TaskRegistry._tasks.copy()
    original_aliases = TaskRegistry._task_aliases.copy()
    
    yield
    
    # Restore original registries
    YOLOModelRegistry._models = original_models
    TaskRegistry._tasks = original_tasks
    TaskRegistry._task_aliases = original_aliases


@pytest.fixture
def sample_performance_results() -> list:
    """Create sample performance analysis results."""
    results = []
    
    for i in range(10):
        score = np.random.random()  # Random IoU score
        results.append({
            'filename': f'image_{i:03d}.jpg',
            'image_path': f'/fake/path/image_{i:03d}.jpg',
            'ground_truth_path': f'/fake/path/mask_{i:03d}.png',
            'score': score,
            'prediction': MockInferenceResult(),
            'ground_truth': np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255,
        })
    
    # Sort by score (worst to best)
    results.sort(key=lambda x: x['score'])
    return results


# Pytest markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (requires actual models/data)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take a long time
        if any(keyword in item.name.lower() for keyword in ['slow', 'real', 'full']):
            item.add_marker(pytest.mark.slow)