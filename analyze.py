# ===================================================================
# FILE: analyze.py
#
# A reusable tool for running Grad-CAM analysis on a trained
# YOLOv8 segmentation model.
# ===================================================================

# --- 0. INSTALL DEPENDENCIES (for notebook use) ---
# !pip install ultralytics grad-cam -q

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import sys

class YOLOv8Analyzer:
    """
    Encapsulates the entire analysis workflow: loading a model,
    evaluating performance, and generating Grad-CAM visualizations.
    """
    def __init__(self, model_path: str):
        """
        Initializes the analyzer by loading the model.
        """
        print("--- Initializing Analyzer ---")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pytorch_model = self._load_pure_model(model_path)
        self.model_for_inference = YOLO(model_path)
        self.cam = self._initialize_cam()
        print("Analyzer initialized successfully.")

    def _load_pure_model(self, model_path: str) -> nn.Module:
        """
        Loads the YOLOv8 model into a "pure" PyTorch module suitable for CAM.
        This is the critical step to bypass inference-time optimizations.
        """
        print("Loading model into a pure, differentiable module...")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = ckpt['model'].yaml
        model = SegmentationModel(cfg)
        model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
        model.eval()
        if self.device == 'cuda':
            model = model.cuda().half()
        return model

    def _initialize_cam(self) -> GradCAM:
        """
        Initializes and returns the GradCAM object.
        """
        # This wrapper simplifies the model's output for the CAM library.
        class YOLOv8SegWithCAM(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0]

        # This target function tells CAM how to calculate a "loss" from the model's output.
        def yolo_v8_seg_target(model_output):
            # We want to maximize the class score (at index 4) for all potential detections.
            return model_output[:, 4].sum()

        cam_model = YOLOv8SegWithCAM(self.pytorch_model)
        target_layer = [self.pytorch_model.model[9].cv2]
        return GradCAM(model=cam_model, target_layers=target_layer)

    def analyze_performance(self, image_dir: str, mask_dir: str) -> list:
        """
        Calculates the IoU for every image in the validation set.
        """
        print(f"\n--- Analyzing performance on images in {image_dir} ---")
        image_performance = []
        valid_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in valid_image_files:
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + '.png')
            if not os.path.exists(mask_path): continue

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None: continue
            _, gt_binary_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

            results = self.model_for_inference(image_path, verbose=False, imgsz=640)
            pred_binary_mask = np.zeros_like(gt_binary_mask)
            if results[0].masks is not None:
                pred_mask_raw = results[0].masks.data[0].cpu().numpy()
                pred_mask_resized = cv2.resize(pred_mask_raw, (gt_binary_mask.shape[1], gt_binary_mask.shape[0]))
                _, pred_binary_mask = cv2.threshold(pred_mask_resized, 0.5, 255, cv2.THRESH_BINARY)
                pred_binary_mask = pred_binary_mask.astype(np.uint8)

            intersection = np.logical_and(gt_binary_mask, pred_binary_mask).sum()
            union = np.logical_or(gt_binary_mask, pred_binary_mask).sum()
            iou = intersection / union if union > 0 else 0
            image_performance.append({'filename': filename, 'iou': iou, 'image_path': image_path, 'mask_path': mask_path})

        image_performance.sort(key=lambda x: x['iou'])
        print(f"Analysis complete. Found {len(image_performance)} image-mask pairs.")
        return image_performance

    def visualize_results(self, image_list: list, analysis_type: str):
        """
        Generates and displays the 1x4 analysis plot for a given list of images.
        """
        print(f"\n--- Visualizing Top {len(image_list)} {analysis_type.upper()} Performing Images ---")
        if not image_list:
            print(f"No images to visualize for category: {analysis_type}")
            return

        for i, item in enumerate(image_list):
            image_path, mask_path, iou_score = item['image_path'], item['mask_path'], item['iou']

            img = cv2.imread(image_path)
            img_resized = cv2.resize(img, (640, 640))
            rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(np.float32(rgb_img) / 255).permute(2, 0, 1).unsqueeze(0)
            if self.device == 'cuda':
                input_tensor = input_tensor.cuda().half()

            grayscale_cam = self.cam(input_tensor=input_tensor, targets=[yolo_v8_seg_target])[0, :]
            cam_image = show_cam_on_image(np.float32(rgb_img) / 255, grayscale_cam, use_rgb=True, image_weight=0.6)

            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(f'{analysis_type.upper()} Example #{i+1}: {item["filename"]}', fontsize=16)

            original_img_for_display = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            axs[0].imshow(original_img_for_display); axs[0].set_title('Original Image'); axs[0].axis('off')
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            axs[1].imshow(gt_mask, cmap='gray'); axs[1].set_title('Ground Truth Mask'); axs[1].axis('off')
            results = self.model_for_inference(image_path, verbose=False, imgsz=640)
            pred_mask_display = np.zeros_like(cv2.resize(gt_mask, (gt_mask.shape[1], gt_mask.shape[0])))
            if results[0].masks is not None:
                pred_mask_raw = results[0].masks.data[0].cpu().numpy()
                pred_mask_display = cv2.resize(pred_mask_raw, (gt_mask.shape[1], gt_mask.shape[0]))
            axs[2].imshow(pred_mask_display, cmap='gray'); axs[2].set_title(f'Predicted Mask\nIoU: {iou_score:.3f}'); axs[2].axis('off')
            axs[3].imshow(cam_image); axs[3].set_title('Grad-CAM Heatmap'); axs[3].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

def main():
    """
    The main entry point for running the analysis from the command line.
    """
    parser = argparse.ArgumentParser(description="Run Grad-CAM analysis on a YOLOv8 segmentation model.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the best.pt model checkpoint.')
    parser.add_argument('--image-dir', type=str, required=True, help='Path to the directory of validation images.')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to the directory of validation masks.')
    parser.add_argument('--num-best', type=int, default=5, help='Number of best-performing images to visualize.')
    parser.add_argument('--num-worst', type=int, default=10, help='Number of worst-performing images to visualize.')
    
    # This little trick checks if we are in a notebook or a real CLI
    # If sys.argv has more than one element, we're in a CLI. Otherwise, use notebook defaults.
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # In a notebook, we can't use parser.parse_args(), so we create a mock args object.
        # This allows the same `main` function to work in both environments.
        args = parser.parse_args(args=[
            '--model-path', '/content/drive/MyDrive/YOLOv8n_Wound_Segmentation (1)/results/train/weights/best.pt',
            '--image-dir', './dataset/images/valid',
            '--mask-dir', './dataset/masks/valid',
            '--num-best', '5',
            '--num-worst', '10'
        ])

    analyzer = YOLOv8Analyzer(model_path=args.model_path)
    performance_data = analyzer.analyze_performance(image_dir=args.image_dir, mask_dir=args.mask_dir)

    # Visualize the best and worst cases
    best_images = performance_data[-args.num_best:]
    best_images.reverse()
    worst_images = performance_data[:args.num_worst]

    analyzer.visualize_results(best_images, "Best")
    analyzer.visualize_results(worst_images, "Worst")

# This is the standard Python entry point.
if __name__ == '__main__':
    main()