#!/usr/bin/env python3
"""Minimal test to check if tensor cloning fixes the issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn

# Create a minimal test
print("Testing tensor cloning and gradient computation...")

# Simulate the issue
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

# Test inference tensor issue
model = DummyModel()
model.eval()

# Create input tensor (simulating inference mode)
with torch.inference_mode():
    input_tensor = torch.randn(1, 3, 32, 32)

print(f"Original tensor requires_grad: {input_tensor.requires_grad}")
print(f"Original tensor is_inference: {input_tensor.is_inference()}")

# Try without cloning (should fail)
try:
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    target = output.mean()
    target.backward()
    print("❌ ERROR: This should have failed!")
except RuntimeError as e:
    print(f"✅ Expected error without cloning: {type(e).__name__}")

# Try with cloning (should work)
try:
    # Clone and detach to escape inference mode
    cloned_tensor = input_tensor.clone().detach().requires_grad_(True)
    print(f"\nCloned tensor requires_grad: {cloned_tensor.requires_grad}")
    print(f"Cloned tensor is_inference: {cloned_tensor.is_inference()}")
    
    # Now gradient computation should work
    with torch.set_grad_enabled(True):
        output = model(cloned_tensor)
        target = output.mean()
        target.backward()
    
    print("✅ SUCCESS: Gradient computation worked with cloning!")
    
except Exception as e:
    print(f"❌ FAILED with cloning: {e}")

# Test our actual implementation
print("\n" + "="*50)
print("Testing actual ManualGradCAM implementation...")

from yolocam.cam.gradcam import ManualGradCAM

# Create a simple model with a known layer
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(3, 16, 3, padding=1)
        self.layer2 = nn.Conv2d(16, 32, 3, padding=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

test_model = TestModel()
target_layers = [test_model.layer1]

# Initialize ManualGradCAM
cam = ManualGradCAM(test_model, target_layers, device='cpu')

# Test with inference tensor
with torch.inference_mode():
    test_input = torch.randn(1, 3, 64, 64)

try:
    # This should work because ManualGradCAM clones the tensor
    cam_output = cam.generate_cam(test_input)
    print(f"✅ ManualGradCAM worked! Output shape: {cam_output.shape}")
except Exception as e:
    print(f"❌ ManualGradCAM failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")