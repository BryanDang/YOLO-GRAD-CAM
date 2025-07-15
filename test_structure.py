#!/usr/bin/env python3
"""Test the library structure without dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Checking library structure...")

# Test module imports
modules_to_check = [
    'yolocam',
    'yolocam.core',
    'yolocam.core.analyzer',
    'yolocam.core.config', 
    'yolocam.core.registry',
    'yolocam.models',
    'yolocam.models.base_model',
    'yolocam.models.yolo_wrapper',
    'yolocam.tasks',
    'yolocam.tasks.base_task',
    'yolocam.tasks.segmentation',
    'yolocam.cam',
    'yolocam.cam.gradcam',
    'yolocam.utils',
    'yolocam.utils.logging',
    'yolocam.utils.validation'
]

failed = []
for module in modules_to_check:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module}: {e}")
        failed.append(module)

if failed:
    print(f"\n{len(failed)} modules failed to import")
else:
    print("\n✓ All modules can be imported!")