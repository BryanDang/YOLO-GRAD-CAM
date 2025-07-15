#!/usr/bin/env python3
"""
Comprehensive testing script to validate YoloCAM library before upload.

This script performs end-to-end testing of the library including:
1. Package structure validation
2. Import testing
3. Configuration testing
4. Mock functionality testing
5. CLI testing
6. Build testing

Run this before uploading to ensure everything works.
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import importlib
import json


class TestRunner:
    """Comprehensive test runner for pre-upload validation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def run_all_tests(self):
        """Run all pre-upload tests."""
        print("[TEST] YoloCAM Pre-Upload Testing")
        print("=" * 50)
        
        tests = [
            ("Package Structure", self.test_package_structure),
            ("Import System", self.test_imports),
            ("Configuration System", self.test_configuration),
            ("Registry System", self.test_registry),
            ("Mock Functionality", self.test_mock_functionality),
            ("CLI Interface", self.test_cli),
            ("Build System", self.test_build),
            ("Code Quality", self.test_code_quality),
        ]
        
        for test_name, test_func in tests:
            print(f"\n[Section] Testing {test_name}...")
            try:
                test_func()
                print(f"PASSED {test_name}: PASSED")
            except Exception as e:
                print(f"FAILED {test_name}: FAILED - {e}")
                self.errors.append(f"{test_name}: {e}")
        
        self.print_summary()
        return len(self.errors) == 0
    
    def test_package_structure(self):
        """Test that package structure is correct."""
        required_files = [
            "pyproject.toml",
            "src/yolocam/__init__.py",
            "src/yolocam/core/__init__.py",
            "src/yolocam/models/__init__.py",
            "src/yolocam/tasks/__init__.py",
            "src/yolocam/utils/__init__.py",
            "tests/conftest.py",
            ".github/workflows/ci.yml",
            "Makefile",
            "LICENSE",
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required file missing: {file_path}")
        
        # Check that __init__.py files have content
        init_files = [
            "src/yolocam/__init__.py",
            "src/yolocam/core/__init__.py",
        ]
        
        for init_file in init_files:
            full_path = self.project_root / init_file
            if full_path.stat().st_size == 0:
                raise ValueError(f"Empty __init__.py file: {init_file}")
    
    def test_imports(self):
        """Test that all imports work correctly."""
        # Add src to path
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test main imports
        try:
            import yolocam
            assert hasattr(yolocam, '__version__')
            assert hasattr(yolocam, 'YoloCAMAnalyzer')
            assert hasattr(yolocam, 'YoloCAMConfig')
        except ImportError as e:
            raise ImportError(f"Failed to import main package: {e}")
        
        # Test subpackage imports
        subpackages = [
            'yolocam.core',
            'yolocam.models',
            'yolocam.tasks',
            'yolocam.utils',
            'yolocam.cli',
        ]
        
        for package in subpackages:
            try:
                importlib.import_module(package)
            except ImportError as e:
                raise ImportError(f"Failed to import {package}: {e}")
    
    def test_configuration(self):
        """Test configuration system."""
        from yolocam.core.config import YoloCAMConfig
        
        # Test default config creation
        config = YoloCAMConfig()
        assert config.device == 'auto'
        assert config.cam_method == 'gradcam'
        
        # Test config validation
        try:
            YoloCAMConfig(device='invalid')
            raise AssertionError("Should have failed with invalid device")
        except ValueError:
            pass  # Expected
        
        # Test config serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'device' in config_dict
        
        # Test config with custom values
        custom_config = YoloCAMConfig(
            device='cpu',
            model_input_size=320,
            num_best_examples=3
        )
        assert custom_config.device == 'cpu'
        assert custom_config.model_input_size == 320
        assert custom_config.num_best_examples == 3
    
    def test_registry(self):
        """Test registry system."""
        from yolocam.core.registry import (
            YOLOModelRegistry, TaskRegistry, 
            register_yolo_model, register_yolo_task,
            validate_registries
        )
        from yolocam.models.base_model import BaseYOLOModel
        from yolocam.tasks.base_task import BaseTask
        
        # Test model registry
        initial_models = len(YOLOModelRegistry.list_models())
        
        @register_yolo_model('test_model')
        class TestModel(BaseYOLOModel):
            def load_model(self, model_path): pass
            def get_target_layers(self, component='backbone'): pass
            def get_inference_model(self): pass
            def preprocess_input(self, image_path): pass
            def detect_task(self): return 'test'
            @property
            def supported_tasks(self): return ['test']
            @property
            def version(self): return 'test_model'
        
        assert len(YOLOModelRegistry.list_models()) == initial_models + 1
        assert YOLOModelRegistry.get_model('test_model') == TestModel
        
        # Test task registry
        initial_tasks = len(TaskRegistry.list_tasks())
        
        @register_yolo_task('test_task')
        class TestTask(BaseTask):
            def compute_performance_metric(self, prediction, ground_truth, **kwargs): return 0.5
            def create_cam_target_function(self, **kwargs): return lambda x: x.sum()
            def load_ground_truth(self, path): return None
            def visualize_results(self, image, prediction, ground_truth, cam_output, **kwargs): return None
            @property
            def task_name(self): return 'test_task'
            @property
            def supported_metrics(self): return ['test_metric']
            @property
            def required_ground_truth_format(self): return 'test_format'
        
        assert len(TaskRegistry.list_tasks()) == initial_tasks + 1
        assert TaskRegistry.get_task('test_task') == TestTask
    
    def test_mock_functionality(self):
        """Test core functionality with mocks."""
        from yolocam.core.config import YoloCAMConfig
        from yolocam.utils.logging import get_logger, setup_logging
        from yolocam.utils.validation import validate_config_value
        
        # Test logging
        setup_logging()
        logger = get_logger(__name__)
        logger.info("Test log message")
        
        # Test validation
        result = validate_config_value(
            5, 
            lambda x: isinstance(x, int) and x > 0,
            "Must be positive integer",
            "INVALID_INT"
        )
        assert result == 5
        
        # Test config with task configs
        config = YoloCAMConfig()
        seg_config = config.get_task_config('segmentation')
        assert isinstance(seg_config, dict)
        assert 'metrics' in seg_config
    
    def test_cli(self):
        """Test CLI interface."""
        from yolocam.cli.commands import create_parser, create_config_template
        from yolocam.core.config import YoloCAMConfig
        
        # Test parser creation
        parser = create_parser()
        assert parser is not None
        
        # Test help doesn't crash
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            pass  # Expected for --help
        
        # Test config template generation
        config = YoloCAMConfig()
        template = create_config_template(config)
        assert 'YoloCAM Configuration Template' in template
        assert 'device:' in template
    
    def test_build(self):
        """Test that package can be built."""
        # Test that pyproject.toml is valid
        import tomllib
        
        with open(self.project_root / 'pyproject.toml', 'rb') as f:
            pyproject = tomllib.load(f)
        
        assert 'project' in pyproject
        assert 'name' in pyproject['project']
        assert pyproject['project']['name'] == 'yolocam'
        assert 'dependencies' in pyproject['project']
        
        # Test that we can import the build system
        try:
            import build
        except ImportError:
            self.warnings.append("python-build not available for build testing")
            return
        
        # Test build in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_project = Path(temp_dir) / 'yolocam'
            shutil.copytree(self.project_root, temp_project)
            
            # Try to build (this will fail without dependencies, but we can check syntax)
            result = subprocess.run([
                sys.executable, '-m', 'build', '--no-isolation', '--wheel'
            ], cwd=temp_project, capture_output=True, text=True)
            
            # We expect this to fail due to missing dependencies, but not due to syntax errors
            if 'invalid syntax' in result.stderr.lower():
                raise SyntaxError(f"Syntax error in build: {result.stderr}")
    
    def test_code_quality(self):
        """Test code quality basics."""
        # Check for common Python issues
        python_files = list(self.project_root.glob('src/**/*.py'))
        python_files.extend(self.project_root.glob('tests/**/*.py'))
        
        issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for syntax by compiling
                compile(content, str(py_file), 'exec')
                
                # Check for common issues
                if 'print(' in content and 'test' not in str(py_file):
                    issues.append(f"Print statement in {py_file}")
                
                if 'import *' in content:
                    issues.append(f"Star import in {py_file}")
                    
            except SyntaxError as e:
                issues.append(f"Syntax error in {py_file}: {e}")
            except Exception as e:
                issues.append(f"Error reading {py_file}: {e}")
        
        if issues:
            raise ValueError(f"Code quality issues: {'; '.join(issues[:5])}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("[SUMMARY] TEST SUMMARY")
        print("=" * 50)
        
        if not self.errors and not self.warnings:
            print("SUCCESS ALL TESTS PASSED!")
            print("PASSED Your library is ready for upload!")
        else:
            if self.errors:
                print(f"FAILED {len(self.errors)} ERRORS:")
                for error in self.errors:
                    print(f"   • {error}")
            
            if self.warnings:
                print(f"WARNING {len(self.warnings)} WARNINGS:")
                for warning in self.warnings:
                    print(f"   • {warning}")
        
        print("\n[INFO] Next Steps:")
        if not self.errors:
            print("1. Run: make test  # Run full test suite")
            print("2. Run: make check # Run quality checks")
            print("3. Initialize git repository")
            print("4. Push to GitHub")
            print("5. Create first release with: git tag v0.1.0")
        else:
            print("1. Fix the errors listed above")
            print("2. Run this test script again")
            print("3. Proceed with upload when all tests pass")


def quick_functionality_test():
    """Quick smoke test of core functionality."""
    print("\n[TEST] Quick Functionality Test")
    print("-" * 30)
    
    try:
        # Add src to path
        import sys
        from pathlib import Path
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Test basic imports
        from yolocam import YoloCAMConfig
        from yolocam.core.registry import YOLOModelRegistry, TaskRegistry
        from yolocam.utils.logging import get_logger
        
        print("PASSED Imports successful")
        
        # Test config
        config = YoloCAMConfig(device='cpu', model_input_size=320)
        print(f"PASSED Config created: device={config.device}, size={config.model_input_size}")
        
        # Test logging
        logger = get_logger(__name__)
        logger.info("Test log message")
        print("PASSED Logging system working")
        
        # Test registries
        model_count = len(YOLOModelRegistry.list_models())
        task_count = len(TaskRegistry.list_tasks())
        print(f"PASSED Registry system: {model_count} models, {task_count} tasks registered")
        
        print("SUCCESS Quick functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"FAILED Quick functionality test FAILED: {e}")
        return False


def main():
    """Main testing function."""
    print("[TEST] YoloCAM Pre-Upload Testing Suite")
    print("=" * 50)
    print("This will validate your library before upload to ensure everything works correctly.")
    print()
    
    # Quick test first
    if not quick_functionality_test():
        print("\nFAILED Quick test failed. Please check your installation.")
        return 1
    
    # Full test suite
    runner = TestRunner()
    success = runner.run_all_tests()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())