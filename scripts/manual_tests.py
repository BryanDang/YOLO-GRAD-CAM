#!/usr/bin/env python3
"""
Manual testing scenarios for YoloCAM library.

These tests require manual inspection and validation.
Run these after the automated tests pass.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image


def setup_test_environment():
    """Setup test environment with mock data."""
    # Add src to path
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Create temporary directory for test files
    temp_dir = Path(tempfile.mkdtemp(prefix='yolocam_manual_test_'))
    
    # Create mock images and masks
    images_dir = temp_dir / 'images'
    masks_dir = temp_dir / 'masks'
    images_dir.mkdir()
    masks_dir.mkdir()
    
    # Create 5 test images and masks
    for i in range(5):
        # Create test image
        image = Image.new('RGB', (640, 480), color=(i*50, 100, 200))
        # Add some patterns
        pixels = np.array(image)
        pixels[100:200, 100:200] = [255, 255, 255]  # White square
        pixels[300:400, 300:400] = [255, 0, 0]      # Red square
        image = Image.fromarray(pixels)
        image.save(images_dir / f'test_image_{i:03d}.jpg')
        
        # Create corresponding mask
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # Mask for white square
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(masks_dir / f'test_image_{i:03d}.png')
    
    return temp_dir, images_dir, masks_dir


def test_basic_configuration():
    """Test 1: Basic configuration creation and validation."""
    print("[Section] Test 1: Basic Configuration")
    print("-" * 30)
    
    try:
        from yolocam import YoloCAMConfig
        
        # Test default config
        config = YoloCAMConfig()
        print(f"PASSED Default config created")
        print(f"   Device: {config.device}")
        print(f"   CAM method: {config.cam_method}")
        print(f"   Input size: {config.model_input_size}")
        
        # Test custom config
        custom_config = YoloCAMConfig(
            device='cpu',
            cam_method='gradcam',
            model_input_size=320,
            num_best_examples=3
        )
        print(f"PASSED Custom config created")
        print(f"   Device: {custom_config.device}")
        print(f"   Best examples: {custom_config.num_best_examples}")
        
        # Test config to dict
        config_dict = custom_config.to_dict()
        print(f"PASSED Config serialization: {len(config_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"FAILED Configuration test failed: {e}")
        return False


def test_registry_system():
    """Test 2: Registry system with mock models and tasks."""
    print("\n[Section] Test 2: Registry System")
    print("-" * 30)
    
    try:
        from yolocam.core.registry import (
            YOLOModelRegistry, TaskRegistry,
            register_yolo_model, register_yolo_task
        )
        from yolocam.models.base_model import BaseYOLOModel
        from yolocam.tasks.base_task import BaseTask
        from yolocam.core.config import YoloCAMConfig
        
        # Register a test model
        @register_yolo_model('manual_test_model')
        class ManualTestModel(BaseYOLOModel):
            def load_model(self, model_path):
                print(f"   Mock loading model: {model_path}")
                return None
            
            def get_target_layers(self, component='backbone'):
                print(f"   Mock getting target layers for: {component}")
                return []
            
            def get_inference_model(self):
                print(f"   Mock getting inference model")
                return lambda x: []
            
            def preprocess_input(self, image_path):
                print(f"   Mock preprocessing: {image_path}")
                import torch
                return torch.zeros(1, 3, 320, 320)
            
            def detect_task(self):
                return 'manual_test'
            
            @property
            def supported_tasks(self):
                return ['manual_test']
            
            @property
            def version(self):
                return 'manual_test_model'
        
        # Register a test task
        @register_yolo_task('manual_test')
        class ManualTestTask(BaseTask):
            def compute_performance_metric(self, prediction, ground_truth, **kwargs):
                print(f"   Mock computing performance metric")
                return 0.75
            
            def create_cam_target_function(self, **kwargs):
                print(f"   Mock creating CAM target function")
                return lambda x: x.sum()
            
            def load_ground_truth(self, path):
                print(f"   Mock loading ground truth: {path}")
                return np.zeros((100, 100))
            
            def visualize_results(self, image, prediction, ground_truth, cam_output, **kwargs):
                print(f"   Mock visualizing results")
                return np.zeros((480, 640, 3), dtype=np.uint8)
            
            @property
            def task_name(self):
                return 'manual_test'
            
            @property
            def supported_metrics(self):
                return ['mock_metric']
            
            @property
            def required_ground_truth_format(self):
                return 'mock_format'
        
        # Test registry
        models = YOLOModelRegistry.list_models()
        tasks = TaskRegistry.list_tasks()
        
        print(f"PASSED Registered models: {len(models)} ({', '.join(models)})")
        print(f"PASSED Registered tasks: {len(tasks)} ({', '.join(tasks)})")
        
        # Test getting registered components
        model_class = YOLOModelRegistry.get_model('manual_test_model')
        task_class = TaskRegistry.get_task('manual_test')
        
        print(f"PASSED Retrieved model class: {model_class}")
        print(f"PASSED Retrieved task class: {task_class}")
        
        return True
        
    except Exception as e:
        print(f"FAILED Registry test failed: {e}")
        return False


def test_cli_interface():
    """Test 3: CLI interface without actual execution."""
    print("\n[Section] Test 3: CLI Interface")
    print("-" * 30)
    
    try:
        from yolocam.cli.commands import create_parser, create_config_template
        from yolocam.core.config import YoloCAMConfig
        
        # Test parser creation
        parser = create_parser()
        print(f"PASSED CLI parser created")
        
        # Test help generation (this will raise SystemExit, which is expected)
        try:
            # Capture help output
            import io
            import contextlib
            
            help_output = io.StringIO()
            with contextlib.redirect_stdout(help_output):
                with contextlib.redirect_stderr(help_output):
                    try:
                        parser.parse_args(['--help'])
                    except SystemExit:
                        pass
            
            help_text = help_output.getvalue()
            print(f"PASSED Help system working (captured {len(help_text)} characters)")
            
        except Exception as e:
            print(f"WARNING Help test skipped: {e}")
        
        # Test config template generation
        config = YoloCAMConfig()
        template = create_config_template(config)
        
        print(f"PASSED Config template generated ({len(template)} characters)")
        
        # Check template content
        required_sections = ['Model settings', 'CAM settings', 'device:', 'cam_method:']
        missing_sections = [section for section in required_sections if section not in template]
        
        if missing_sections:
            print(f"WARNING Missing template sections: {missing_sections}")
        else:
            print(f"PASSED All required template sections present")
        
        return True
        
    except Exception as e:
        print(f"FAILED CLI test failed: {e}")
        return False


def test_mock_analyzer():
    """Test 4: Mock analyzer functionality."""
    print("\n[Section] Test 4: Mock Analyzer")
    print("-" * 30)
    
    try:
        temp_dir, images_dir, masks_dir = setup_test_environment()
        print(f"PASSED Test environment created: {temp_dir}")
        
        # Test imports
        from yolocam.core.config import YoloCAMConfig
        from yolocam.utils.logging import setup_logging, get_logger
        
        # Setup logging
        setup_logging()
        logger = get_logger(__name__)
        print(f"PASSED Logging setup complete")
        
        # Test configuration
        config = YoloCAMConfig(
            device='cpu',
            model_input_size=320,
            num_best_examples=2,
            num_worst_examples=3,
            save_visualizations=False,  # Don't save during test
            output_dir=str(temp_dir / 'results')
        )
        print(f"PASSED Test configuration created")
        
        # Test logging with context
        logger.info("Test analyzer functionality", test_phase="mock_analyzer")
        print(f"PASSED Contextual logging working")
        
        # Test file operations
        from yolocam.utils.io import find_files, get_file_info
        
        image_files = find_files(images_dir, pattern="*.jpg")
        mask_files = find_files(masks_dir, pattern="*.png")
        
        print(f"PASSED Found {len(image_files)} images and {len(mask_files)} masks")
        
        if image_files:
            file_info = get_file_info(image_files[0])
            print(f"PASSED File info extracted: {file_info['name']} ({file_info['size_bytes']} bytes)")
        
        # Test validation
        from yolocam.utils.validation import validate_directory, validate_image_path
        
        validate_directory(images_dir)
        validate_directory(masks_dir)
        if image_files:
            validate_image_path(image_files[0])
        print(f"PASSED Path validation working")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"PASSED Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"FAILED Mock analyzer test failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False


def test_package_imports():
    """Test 5: Package imports and structure."""
    print("\n[Section] Test 5: Package Imports")
    print("-" * 30)
    
    try:
        # Test main package import
        import yolocam
        print(f"PASSED Main package imported: yolocam v{yolocam.__version__}")
        
        # Test main classes
        from yolocam import YoloCAMConfig, YoloCAMAnalyzer
        print(f"PASSED Main classes available: {YoloCAMConfig}, {YoloCAMAnalyzer}")
        
        # Test subpackages
        subpackages = [
            'yolocam.core',
            'yolocam.models',
            'yolocam.tasks',
            'yolocam.utils',
            'yolocam.cli',
            'yolocam.visualization',
            'yolocam.cam'
        ]
        
        imported_count = 0
        for package in subpackages:
            try:
                __import__(package)
                imported_count += 1
                print(f"   PASSED {package}")
            except ImportError as e:
                print(f"   FAILED {package}: {e}")
        
        print(f"PASSED Successfully imported {imported_count}/{len(subpackages)} subpackages")
        
        # Test specific utilities
        from yolocam.utils.logging import get_logger
        from yolocam.utils.validation import validate_config_value
        from yolocam.core.registry import get_registry_info
        
        logger = get_logger(__name__)
        registry_info = get_registry_info()
        
        print(f"PASSED Utility functions working")
        print(f"   Logger: {type(logger)}")
        print(f"   Registry: {registry_info['models']['count']} models, {registry_info['tasks']['count']} tasks")
        
        return True
        
    except Exception as e:
        print(f"FAILED Package import test failed: {e}")
        return False


def run_manual_tests():
    """Run all manual tests."""
    print("[TEST] YoloCAM Manual Testing Suite")
    print("=" * 50)
    print("These tests require manual inspection of output.")
    print("Please review each test result carefully.")
    print()
    
    tests = [
        test_basic_configuration,
        test_registry_system,
        test_cli_interface,
        test_mock_analyzer,
        test_package_imports,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"FAILED Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("[SUMMARY] MANUAL TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS ALL MANUAL TESTS PASSED!")
        print("PASSED Your library appears to be working correctly!")
    else:
        print(f"FAILED {total - passed} tests failed or had issues")
        print("Please review the output above and fix any problems")
    
    print("\n[INFO] What to check manually:")
    print("1. All imports work without errors")
    print("2. Configuration system accepts valid values and rejects invalid ones")
    print("3. Registry system can register and retrieve components")
    print("4. CLI help system displays proper information")
    print("5. Logging system produces readable output")
    print("6. File operations work with test data")
    
    return passed == total


if __name__ == '__main__':
    success = run_manual_tests()
    sys.exit(0 if success else 1)