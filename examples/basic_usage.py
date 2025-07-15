#!/usr/bin/env python3
"""
Basic usage example for YoloCAM library.

This example demonstrates how to:
1. Initialize the analyzer with auto-detection
2. Analyze model performance on a validation set
3. Generate Grad-CAM visualizations
4. Save results and visualizations

Run this example:
    python examples/basic_usage.py
"""

import os
from pathlib import Path
from yolocam import YoloCAMAnalyzer, YoloCAMConfig
from yolocam.utils.logging import get_logger, setup_logging


def main():
    """Main example function."""
    
    # Setup logging for the example
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting YoloCAM basic usage example")
    
    # Configuration for the analysis
    config = YoloCAMConfig(
        device='cpu',  # Use 'cuda' if you have a GPU
        model_input_size=640,
        cam_method='gradcam',
        target_layer_component='neck',  # Can be 'backbone', 'neck', or 'head'
        num_best_examples=3,
        num_worst_examples=5,
        save_visualizations=True,
        output_dir='./example_results',
        log_level='INFO'
    )
    
    # Example paths (replace with your actual paths)
    model_path = "path/to/your/yolo_model.pt"
    images_dir = "path/to/validation/images"
    masks_dir = "path/to/validation/masks"  # For segmentation task
    
    # Check if example files exist
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        logger.info("Please update model_path, images_dir, and masks_dir with your actual paths")
        
        # Create mock example for demonstration
        logger.info("Creating mock example...")
        create_mock_example(config)
        return
    
    try:
        # Initialize the analyzer
        # YoloCAM will automatically detect the model version and task
        logger.info(f"Initializing analyzer with model: {model_path}")
        analyzer = YoloCAMAnalyzer(
            model_path=model_path,
            task='auto',  # Auto-detect task, or specify: 'detection', 'segmentation', etc.
            config=config
        )
        
        # Get information about the analyzer
        info = analyzer.get_info()
        logger.info(f"Analyzer initialized:")
        logger.info(f"  Model version: {info['model']['version']}")
        logger.info(f"  Task: {info['task']['task_name']}")
        logger.info(f"  CAM method: {info['cam_method']}")
        logger.info(f"  Target component: {info['target_component']}")
        
        # Analyze performance on validation set
        logger.info("Starting performance analysis...")
        results = analyzer.analyze_performance(
            image_dir=images_dir,
            ground_truth_dir=masks_dir
        )
        
        logger.info(f"Analysis complete! Processed {len(results)} images")
        
        # Print summary statistics
        scores = [r['score'] for r in results]
        logger.info(f"Performance summary:")
        logger.info(f"  Mean score: {sum(scores) / len(scores):.3f}")
        logger.info(f"  Best score: {max(scores):.3f}")
        logger.info(f"  Worst score: {min(scores):.3f}")
        
        # Get best and worst examples
        best_examples = analyzer.get_best_examples(results)
        worst_examples = analyzer.get_worst_examples(results)
        
        logger.info(f"Best {len(best_examples)} examples:")
        for i, example in enumerate(best_examples):
            logger.info(f"  {i+1}. {example['filename']}: {example['score']:.3f}")
        
        logger.info(f"Worst {len(worst_examples)} examples:")
        for i, example in enumerate(worst_examples):
            logger.info(f"  {i+1}. {example['filename']}: {example['score']:.3f}")
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # Visualize best examples
        best_viz_paths = analyzer.visualize_results(best_examples, "best")
        logger.info(f"Created {len(best_viz_paths)} best example visualizations")
        
        # Visualize worst examples
        worst_viz_paths = analyzer.visualize_results(worst_examples, "worst")
        logger.info(f"Created {len(worst_viz_paths)} worst example visualizations")
        
        # Generate individual CAM for a specific image
        if results:
            example_image = results[0]['image_path']
            logger.info(f"Generating CAM for example image: {example_image}")
            
            cam_output = analyzer.generate_cam(example_image)
            logger.info(f"CAM shape: {cam_output.shape}")
        
        logger.info(f"Results saved to: {analyzer.output_dir}")
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.error("Please check your file paths and model compatibility")
        raise


def create_mock_example(config: YoloCAMConfig):
    """Create a mock example when real files are not available."""
    logger = get_logger(__name__)
    
    logger.info("=" * 50)
    logger.info("MOCK EXAMPLE - How to use YoloCAM")
    logger.info("=" * 50)
    
    print("""
YoloCAM Usage Example:

1. Install YoloCAM:
   pip install yolocam

2. Prepare your data:
   - YOLO model file (.pt format)
   - Validation images directory
   - Ground truth masks/annotations directory

3. Basic usage:
   
   from yolocam import YoloCAMAnalyzer
   
   # Initialize analyzer (auto-detects model type and task)
   analyzer = YoloCAMAnalyzer("your_model.pt")
   
   # Analyze performance
   results = analyzer.analyze_performance("images/", "masks/")
   
   # Get best and worst examples
   best = analyzer.get_best_examples(results, num_examples=5)
   worst = analyzer.get_worst_examples(results, num_examples=10)
   
   # Generate visualizations
   analyzer.visualize_results(best, "best")
   analyzer.visualize_results(worst, "worst")

4. Advanced configuration:
   
   from yolocam import YoloCAMConfig
   
   config = YoloCAMConfig(
       device='cuda',
       cam_method='gradcam',
       target_layer_component='neck',
       num_best_examples=10,
       save_visualizations=True
   )
   
   analyzer = YoloCAMAnalyzer("model.pt", config=config)

5. Command line usage:
   
   yolocam analyze --model model.pt --images images/ --masks masks/

6. Update this example:
   - Replace 'path/to/your/yolo_model.pt' with your model path
   - Replace 'path/to/validation/images' with your images directory
   - Replace 'path/to/validation/masks' with your ground truth directory
   - Run the example again

Features:
✓ Automatic YOLO version detection (v8, v9, v10+)
✓ Multi-task support (detection, segmentation, classification, pose)
✓ Plugin architecture for easy extension
✓ Comprehensive testing and CI/CD
✓ Professional logging and error handling
✓ Zero-maintenance operation with automation

For more examples and documentation:
- GitHub: https://github.com/yourusername/yolocam
- Documentation: https://yolocam.readthedocs.io
""")


if __name__ == "__main__":
    main()