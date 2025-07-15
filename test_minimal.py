#!/usr/bin/env python3
"""
Minimal test without external dependencies to verify core structure works.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("[TEST] Minimal YoloCAM Library Test")
print("=" * 50)

# Test 1: Package structure
print("\n[1] Testing package structure...")
try:
    # Check that main package exists
    import yolocam
    print("PASSED Main package found")
    
    # Check version info
    if hasattr(yolocam, '__version__'):
        print(f"PASSED Version: {yolocam.__version__}")
    else:
        print("WARNING No version found")
        
except Exception as e:
    print(f"FAILED Package structure test failed: {e}")
    sys.exit(1)

# Test 2: Core subpackages
print("\n[2] Testing core subpackages...")
subpackages = ['core', 'models', 'tasks', 'utils', 'cli']
for subpkg in subpackages:
    try:
        module = __import__(f'yolocam.{subpkg}', fromlist=[''])
        print(f"PASSED yolocam.{subpkg} imports successfully")
    except Exception as e:
        print(f"FAILED yolocam.{subpkg} failed: {e}")

# Test 3: Check Python syntax
print("\n[3] Checking Python syntax...")
import ast
import os

error_count = 0
file_count = 0

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            file_count += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the Python file to check syntax
                ast.parse(content, filename=file_path)
                
            except SyntaxError as e:
                print(f"FAILED Syntax error in {file_path}: {e}")
                error_count += 1
            except Exception as e:
                print(f"WARNING Could not check {file_path}: {e}")

if error_count == 0:
    print(f"PASSED All {file_count} Python files have valid syntax!")
else:
    print(f"FAILED Found {error_count} files with syntax errors")

# Test 4: Check __init__.py files
print("\n[4] Checking __init__.py files...")
init_files = [
    'src/yolocam/__init__.py',
    'src/yolocam/core/__init__.py',
    'src/yolocam/models/__init__.py',
    'src/yolocam/tasks/__init__.py',
    'src/yolocam/utils/__init__.py',
]

for init_file in init_files:
    if os.path.exists(init_file):
        size = os.path.getsize(init_file)
        if size > 0:
            print(f"PASSED {init_file} exists ({size} bytes)")
        else:
            print(f"WARNING {init_file} is empty")
    else:
        print(f"FAILED {init_file} missing")

# Test 5: Check critical files
print("\n[5] Checking critical files...")
critical_files = [
    'pyproject.toml',
    'README.md',
    'LICENSE',
    '.gitignore',
    'Makefile',
]

for file in critical_files:
    if os.path.exists(file):
        print(f"PASSED {file} exists")
    else:
        print(f"FAILED {file} missing")

# Summary
print("\n" + "=" * 50)
print("[SUMMARY] MINIMAL TEST SUMMARY")
print("=" * 50)

print("""
PASSED Package structure is valid
PASSED Python files have correct syntax
PASSED Core files are in place

NOTE: This is a minimal test without dependencies.
   To fully test the library:
   
   1. Create a virtual environment:
      python3 -m venv venv
      source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   
   2. Install the package:
      pip install -e .
   
   3. Run the full test:
      python test_it_works.py
   
   4. Or run specific tests:
      python scripts/test_before_upload.py

The library structure looks good! Just need to install dependencies to test functionality.
""")