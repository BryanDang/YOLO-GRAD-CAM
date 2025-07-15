# YoloCAM Pre-Upload Testing Checklist

Use this checklist to thoroughly test your library before uploading to GitHub/PyPI.

## Quick Start Testing

### 1. Environment Setup
```bash
# Navigate to project directory
cd /path/to/YOLO-GRAD-CAM

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### 2. Run Automated Test Suite
```bash
# Run the comprehensive test script
python scripts/test_before_upload.py

# If that passes, run manual tests
python scripts/manual_tests.py
```

## Detailed Testing Steps

### Step 1: Package Structure Validation
- [ ] All required files exist (pyproject.toml, src/, tests/, etc.)
- [ ] Package imports work correctly
- [ ] __init__.py files have proper content
- [ ] No syntax errors in any Python files

```bash
# Quick structure check
ls -la pyproject.toml src/yolocam/__init__.py tests/conftest.py
```

### Step 2: Import System Testing
- [ ] Main package imports: `import yolocam`
- [ ] Core classes available: `YoloCAMAnalyzer`, `YoloCAMConfig`
- [ ] All subpackages import without errors
- [ ] No circular import issues

```python
# Test in Python REPL
import yolocam
from yolocam import YoloCAMAnalyzer, YoloCAMConfig
print(f"YoloCAM version: {yolocam.__version__}")
```

### Step 3: Configuration System
- [ ] Default configuration creates successfully
- [ ] Custom configurations work
- [ ] Invalid configurations raise appropriate errors
- [ ] Configuration serialization (to_dict, from_file) works
- [ ] Environment variable override works

```python
# Test configuration
from yolocam import YoloCAMConfig

# Default config
config = YoloCAMConfig()
print(f"Device: {config.device}, CAM: {config.cam_method}")

# Custom config
custom = YoloCAMConfig(device='cpu', model_input_size=320)
print(f"Custom device: {custom.device}, size: {custom.model_input_size}")

# Test validation (should fail)
try:
    YoloCAMConfig(device='invalid_device')
    print("ERROR: Should have failed!")
except ValueError:
    print("SUCCESS: Validation working")
```

### Step 4: Registry System
- [ ] Model registry accepts new models
- [ ] Task registry accepts new tasks
- [ ] Auto-detection patterns work
- [ ] Registry validation catches issues

```python
# Test registry
from yolocam.core.registry import YOLOModelRegistry, TaskRegistry, get_registry_info

info = get_registry_info()
print(f"Models: {info['models']['count']}, Tasks: {info['tasks']['count']}")
```

### Step 5: Logging System
- [ ] Logger creation works
- [ ] Different log levels work
- [ ] Structured logging produces valid output
- [ ] Performance logging works
- [ ] Context management works

```python
# Test logging
from yolocam.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)
logger.info("Test message")
logger.debug("Debug message")
logger.error("Error message")
```

### Step 6: CLI Interface
- [ ] CLI parser creates without errors
- [ ] Help system works (`yolocam --help`)
- [ ] Subcommands are available
- [ ] Config template generation works

```bash
# Test CLI (should show help)
python -m yolocam.cli.commands --help

# Test config template
python -c "
from yolocam.cli.commands import create_config_template
from yolocam import YoloCAMConfig
print(create_config_template(YoloCAMConfig()))
"
```

### Step 7: Error Handling
- [ ] Custom exceptions work properly
- [ ] Input validation catches invalid inputs
- [ ] Error messages are helpful
- [ ] Stack traces are preserved

```python
# Test error handling
from yolocam.utils.validation import validate_model_path, ValidationError

try:
    validate_model_path("nonexistent_file.pt")
    print("ERROR: Should have failed!")
except ValidationError as e:
    print(f"SUCCESS: Validation error caught: {e.error_code}")
```

### Step 8: Mock Functionality
- [ ] Mock models can be registered
- [ ] Mock tasks can be registered
- [ ] Basic workflow runs without real YOLO models
- [ ] File I/O operations work

```python
# Create mock test (see scripts/manual_tests.py for full example)
from yolocam.core.registry import register_yolo_model
from yolocam.models.base_model import BaseYOLOModel

@register_yolo_model('test_model')
class TestModel(BaseYOLOModel):
    # Minimal implementation for testing
    pass
```

### Step 9: Build System
- [ ] pyproject.toml is valid
- [ ] Package can be built with `python -m build`
- [ ] Dependencies are correctly specified
- [ ] Entry points work

```bash
# Test build (install build first: pip install build)
python -m build --no-isolation --wheel

# Check if wheel was created
ls dist/*.whl
```

### Step 10: Documentation
- [ ] README.md renders correctly
- [ ] Examples run without errors
- [ ] Docstrings are present and helpful
- [ ] Type hints are correct

```bash
# Quick docstring check
python -c "from yolocam import YoloCAMAnalyzer; help(YoloCAMAnalyzer)"
```

## Testing Scenarios

### Scenario 1: Complete Mock Workflow
```python
# This should run without real YOLO models
from yolocam import YoloCAMConfig
from yolocam.utils.logging import setup_logging
import tempfile
from pathlib import Path

# Setup
setup_logging()
config = YoloCAMConfig(device='cpu', save_visualizations=False)

# Create mock data directory
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    
    # Mock workflow would go here
    print(f"Mock test directory: {temp_path}")
    print("SUCCESS: Mock workflow setup successful")
```

### Scenario 2: CLI Command Simulation
```python
# Test CLI argument parsing
from yolocam.cli.commands import create_parser

parser = create_parser()

# Test valid arguments
try:
    args = parser.parse_args([
        'analyze', 
        '--model', 'test.pt',
        '--images', 'images/',
        '--masks', 'masks/',
        '--device', 'cpu'
    ])
    print("SUCCESS: CLI parsing successful")
    print(f"Model: {args.model}, Device: {args.device}")
except Exception as e:
    print(f"ERROR: CLI parsing failed: {e}")
```

### Scenario 3: Configuration Round-trip
```python
# Test configuration save/load
from yolocam import YoloCAMConfig
import tempfile
import os

config = YoloCAMConfig(device='cpu', model_input_size=320)

with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    config.save_to_file(f.name)
    
    # Load it back
    loaded_config = YoloCAMConfig.from_file(f.name)
    
    assert loaded_config.device == config.device
    assert loaded_config.model_input_size == config.model_input_size
    print("SUCCESS: Configuration round-trip successful")
    
    os.unlink(f.name)
```

## Common Issues to Check

### Import Issues
- [ ] Missing `__init__.py` files
- [ ] Circular imports
- [ ] Missing dependencies in pyproject.toml
- [ ] Incorrect package structure

### Configuration Issues
- [ ] Invalid default values
- [ ] Missing validation
- [ ] Serialization problems
- [ ] Type mismatches

### CLI Issues
- [ ] Missing required arguments
- [ ] Incorrect argument types
- [ ] Help text problems
- [ ] Command conflicts

### Build Issues
- [ ] Invalid pyproject.toml syntax
- [ ] Missing build requirements
- [ ] Incorrect package data
- [ ] Version conflicts

## Success Criteria

Your library is ready for upload when:

1. **All automated tests pass** (scripts/test_before_upload.py)
2. **All manual tests pass** (scripts/manual_tests.py)
3. **Package builds successfully** (`python -m build`)
4. **CLI help works** (`python -m yolocam.cli.commands --help`)
5. **Basic imports work** (`import yolocam; yolocam.__version__`)
6. **No syntax errors** in any Python files
7. **Mock workflows run** without real YOLO models

## Next Steps After Testing

1. **Initialize Git Repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial YoloCAM library implementation"
   ```

2. **Create GitHub Repository** and push

3. **Setup GitHub Secrets** (for PyPI publishing):
   - PYPI_API_TOKEN (or setup trusted publishing)

4. **Create First Release**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. **Monitor GitHub Actions** for CI/CD pipeline

## Troubleshooting

### If Tests Fail
1. Check error messages carefully
2. Verify all files are in correct locations
3. Check for typos in file names
4. Ensure Python path includes src/
5. Install missing dependencies

### If Imports Fail
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install in development mode
pip install -e .
```

### If Build Fails
1. Check pyproject.toml syntax
2. Install build dependencies: `pip install build`
3. Check for missing files
4. Verify package structure

---

**Remember**: Testing thoroughly now saves hours of debugging later!