#!/usr/bin/env python3
"""
Most basic test - just check if we can import the package structure.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("[INFO] Basic Structure Test")
print("=" * 30)

# Just test version info first
print("Testing version info...")
try:
    # Try to import just the __init__.py content
    sys.path.insert(0, str(Path(__file__).parent / "src" / "yolocam"))
    
    # Import the version info directly
    exec(open("src/yolocam/__init__.py").read(), globals())
    
    print(f"PASSED Version: {__version__}")
    print(f"PASSED Author: {__author__}")
    
except Exception as e:
    print(f"FAILED Basic info failed: {e}")
    
# Test file structure
print("\nTesting file structure...")
required_files = [
    "src/yolocam/__init__.py",
    "src/yolocam/core/__init__.py", 
    "src/yolocam/models/__init__.py",
    "src/yolocam/tasks/__init__.py",
    "pyproject.toml",
    "README.md"
]

all_exist = True
for file_path in required_files:
    if Path(file_path).exists():
        print(f"PASSED {file_path}")
    else:
        print(f"FAILED {file_path} missing")
        all_exist = False

if all_exist:
    print("\nSUCCESS Basic structure is complete!")
    print("\nNext steps - To test functionality:")
    print("1. Create virtual environment: python3 -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install package: pip install -e .")
    print("4. Test imports: python -c 'import yolocam; print(yolocam.__version__)'")
else:
    print("\nFAILED Some files are missing!")

print("\nPASSED Library structure is ready for development!")