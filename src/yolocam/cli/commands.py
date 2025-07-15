"""Command-line interface for YoloCAM library."""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
import json

from ..core.analyzer import YoloCAMAnalyzer
from ..core.config import YoloCAMConfig, load_config_from_env
from ..utils.logging import setup_logging, get_logger
from ..utils.validation import validate_model_path, validate_directory


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='yolocam',
        description='YoloCAM: YOLO Model Analysis with Grad-CAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  yolocam analyze --model model.pt --images images/ --masks masks/
  
  # Advanced analysis with custom config
  yolocam analyze --model model.pt --images images/ --masks masks/ \\
                  --config config.yaml --output results/ --device cuda
  
  # Quick analysis with specific settings
  yolocam analyze --model model.pt --images images/ --masks masks/ \\
                  --num-best 3 --num-worst 5 --cam-method eigencam
  
  # Generate configuration template
  yolocam config --template > config.yaml
  
  # Validate configuration
  yolocam config --validate config.yaml
  
  # Show model information
  yolocam info --model model.pt

For more information: https://github.com/yourusername/yolocam
        """
    )
    
    # Global options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Analyze command
    add_analyze_parser(subparsers)
    
    # Config command
    add_config_parser(subparsers)
    
    # Info command
    add_info_parser(subparsers)
    
    return parser


def add_analyze_parser(subparsers):
    """Add the analyze subcommand parser."""
    parser = subparsers.add_parser(
        'analyze',
        help='Analyze YOLO model performance using Grad-CAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Analyze YOLO model performance and generate Grad-CAM visualizations.

This command will:
1. Auto-detect your YOLO model version and task
2. Analyze performance on your validation dataset
3. Identify best and worst performing examples
4. Generate comprehensive visualizations with Grad-CAM heatmaps
5. Save results and reports to the output directory
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to YOLO model file (.pt, .yaml)'
    )
    
    parser.add_argument(
        '--images', '-i',
        type=str,
        required=True,
        help='Directory containing validation images'
    )
    
    parser.add_argument(
        '--masks', '--ground-truth', '-g',
        type=str,
        required=True,
        help='Directory containing ground truth masks/annotations'
    )
    
    # Optional configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML/JSON)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    # Model settings
    parser.add_argument(
        '--task', '-t',
        type=str,
        choices=['auto', 'detection', 'segmentation', 'classification', 'pose'],
        default='auto',
        help='Task type (default: auto-detect)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        default=640,
        help='Model input size (default: 640)'
    )
    
    # CAM settings
    parser.add_argument(
        '--cam-method',
        type=str,
        choices=['gradcam', 'eigencam'],
        default='gradcam',
        help='CAM method to use (default: gradcam)'
    )
    
    parser.add_argument(
        '--target-component',
        type=str,
        choices=['backbone', 'neck', 'head', 'auto'],
        default='auto',
        help='Model component for CAM target layers (default: auto)'
    )
    
    # Analysis settings
    parser.add_argument(
        '--num-best',
        type=int,
        default=5,
        help='Number of best examples to visualize (default: 5)'
    )
    
    parser.add_argument(
        '--num-worst',
        type=int,
        default=10,
        help='Number of worst examples to visualize (default: 10)'
    )
    
    # Output settings
    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        default=True,
        help='Save visualization images (default: True)'
    )
    
    parser.add_argument(
        '--no-save-visualizations',
        action='store_false',
        dest='save_visualizations',
        help='Do not save visualization images'
    )
    
    parser.add_argument(
        '--save-raw-cams',
        action='store_true',
        help='Save raw CAM arrays as .npy files'
    )
    
    parser.add_argument(
        '--image-format',
        type=str,
        choices=['png', 'jpg', 'pdf'],
        default='png',
        help='Output image format (default: png)'
    )


def add_config_parser(subparsers):
    """Add the config subcommand parser."""
    parser = subparsers.add_parser(
        'config',
        help='Configuration management utilities',
        description='Generate, validate, and manage YoloCAM configurations.'
    )
    
    parser.add_argument(
        '--template',
        action='store_true',
        help='Generate a configuration template'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        metavar='CONFIG_FILE',
        help='Validate a configuration file'
    )
    
    parser.add_argument(
        '--show-defaults',
        action='store_true',
        help='Show default configuration values'
    )


def add_info_parser(subparsers):
    """Add the info subcommand parser."""
    parser = subparsers.add_parser(
        'info',
        help='Show information about models and system',
        description='Display information about YOLO models, system capabilities, and YoloCAM status.'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Show information about a specific model'
    )
    
    parser.add_argument(
        '--system',
        action='store_true',
        help='Show system information and capabilities'
    )
    
    parser.add_argument(
        '--registry',
        action='store_true',
        help='Show registered models and tasks'
    )


def cmd_analyze(args, logger) -> int:
    """Execute the analyze command."""
    try:
        # Load configuration
        config = load_configuration(args, logger)
        
        # Validate inputs
        logger.info("Validating inputs...")
        validate_model_path(args.model)
        validate_directory(args.images)
        validate_directory(args.masks)
        
        # Initialize analyzer
        logger.info(f"Initializing analyzer with model: {args.model}")
        analyzer = YoloCAMAnalyzer(
            model_path=args.model,
            task=args.task,
            config=config
        )
        
        # Show analyzer info
        info = analyzer.get_info()
        logger.info(f"Analyzer initialized:")
        logger.info(f"  Model version: {info['model']['version']}")
        logger.info(f"  Task: {info['task']['task_name']}")
        logger.info(f"  Device: {info['model']['device']}")
        logger.info(f"  CAM method: {info['cam_method']}")
        
        # Run analysis
        logger.info("Starting performance analysis...")
        results = analyzer.analyze_performance(args.images, args.masks)
        
        if not results:
            logger.error("No valid image-ground truth pairs found!")
            return 1
        
        logger.info(f"Analysis complete! Processed {len(results)} images")
        
        # Print summary
        scores = [r['score'] for r in results]
        logger.info(f"Performance summary:")
        logger.info(f"  Mean score: {sum(scores) / len(scores):.3f}")
        logger.info(f"  Best score: {max(scores):.3f}")
        logger.info(f"  Worst score: {min(scores):.3f}")
        
        # Get examples
        best_examples = analyzer.get_best_examples(results, args.num_best)
        worst_examples = analyzer.get_worst_examples(results, args.num_worst)
        
        # Generate visualizations
        if config.save_visualizations:
            logger.info("Generating visualizations...")
            best_paths = analyzer.visualize_results(best_examples, "best")
            worst_paths = analyzer.visualize_results(worst_examples, "worst")
            
            logger.info(f"Created {len(best_paths)} best example visualizations")
            logger.info(f"Created {len(worst_paths)} worst example visualizations")
        
        logger.info(f"Results saved to: {analyzer.output_dir}")
        logger.info("Analysis completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


def cmd_config(args, logger) -> int:
    """Execute the config command."""
    try:
        if args.template:
            # Generate configuration template
            config = YoloCAMConfig()
            template = create_config_template(config)
            print(template)
            return 0
        
        elif args.validate:
            # Validate configuration file
            logger.info(f"Validating configuration: {args.validate}")
            try:
                config = YoloCAMConfig.from_file(args.validate)
                logger.info("PASSED Configuration is valid")
                return 0
            except Exception as e:
                logger.error(f"FAILED Configuration validation failed: {e}")
                return 1
        
        elif args.show_defaults:
            # Show default configuration
            config = YoloCAMConfig()
            print(json.dumps(config.to_dict(), indent=2))
            return 0
        
        else:
            logger.error("No config action specified. Use --help for options.")
            return 1
    
    except Exception as e:
        logger.error(f"Config command failed: {e}")
        return 1


def cmd_info(args, logger) -> int:
    """Execute the info command."""
    try:
        if args.model:
            # Show model information
            logger.info(f"Analyzing model: {args.model}")
            validate_model_path(args.model)
            
            # Create minimal config for model loading
            config = YoloCAMConfig(device='cpu', verbose=False)
            analyzer = YoloCAMAnalyzer(args.model, config=config)
            
            info = analyzer.get_info()
            print_model_info(info)
            return 0
        
        elif args.system:
            # Show system information
            print_system_info()
            return 0
        
        elif args.registry:
            # Show registry information
            print_registry_info()
            return 0
        
        else:
            # Show general information
            print_general_info()
            return 0
    
    except Exception as e:
        logger.error(f"Info command failed: {e}")
        return 1


def load_configuration(args, logger) -> YoloCAMConfig:
    """Load configuration from file and command line arguments."""
    # Start with defaults
    config = YoloCAMConfig()
    
    # Load from file if specified
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        file_config = YoloCAMConfig.from_file(args.config)
        config = config.merge(file_config)
    
    # Load from environment variables
    env_config = load_config_from_env()
    config = config.merge(env_config)
    
    # Override with command line arguments
    cli_config = {}
    
    if hasattr(args, 'device') and args.device != 'auto':
        cli_config['device'] = args.device
    if hasattr(args, 'input_size'):
        cli_config['model_input_size'] = args.input_size
    if hasattr(args, 'cam_method'):
        cli_config['cam_method'] = args.cam_method
    if hasattr(args, 'target_component'):
        cli_config['target_layer_component'] = args.target_component
    if hasattr(args, 'num_best'):
        cli_config['num_best_examples'] = args.num_best
    if hasattr(args, 'num_worst'):
        cli_config['num_worst_examples'] = args.num_worst
    if hasattr(args, 'output'):
        cli_config['output_dir'] = args.output
    if hasattr(args, 'save_visualizations'):
        cli_config['save_visualizations'] = args.save_visualizations
    if hasattr(args, 'save_raw_cams'):
        cli_config['save_raw_cams'] = args.save_raw_cams
    if hasattr(args, 'image_format'):
        cli_config['image_save_format'] = args.image_format
    
    # Set logging level based on verbosity
    if args.verbose:
        cli_config['log_level'] = 'DEBUG'
        cli_config['verbose'] = True
    elif args.quiet:
        cli_config['log_level'] = 'ERROR'
        cli_config['verbose'] = False
    
    config = config.merge(cli_config)
    return config


def create_config_template(config: YoloCAMConfig) -> str:
    """Create a configuration template with comments."""
    template = f"""# YoloCAM Configuration Template
# 
# This file contains all available configuration options for YoloCAM.
# Uncomment and modify the values you want to change.

# Model settings
device: {config.device}                    # Device: 'auto', 'cpu', 'cuda', 'mps'
model_input_size: {config.model_input_size}               # Model input size in pixels
confidence_threshold: {config.confidence_threshold}            # Confidence threshold for predictions
iou_threshold: {config.iou_threshold}                # IoU threshold for NMS

# CAM settings
cam_method: {config.cam_method}                 # CAM method: 'gradcam', 'eigencam'
target_layer_component: {config.target_layer_component}        # Target component: 'backbone', 'neck', 'head', 'auto'
cam_alpha: {config.cam_alpha}                     # CAM overlay transparency (0-1)

# Analysis settings
num_best_examples: {config.num_best_examples}                # Number of best examples to show
num_worst_examples: {config.num_worst_examples}               # Number of worst examples to show
save_visualizations: {config.save_visualizations}             # Save visualization images
save_raw_cams: {config.save_raw_cams}                 # Save raw CAM arrays
output_dir: {config.output_dir}               # Output directory for results

# Visualization settings
figure_size: [{config.figure_size[0]}, {config.figure_size[1]}]                  # Figure size in inches
image_save_format: {config.image_save_format}              # Image format: 'png', 'jpg', 'pdf'
colormap: {config.colormap}                       # Colormap for CAM visualization

# Logging settings
log_level: {config.log_level}                    # Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
verbose: {config.verbose}                      # Enable verbose output

# Task-specific settings
task_configs:
  segmentation:
    metrics: {config.task_configs.get('segmentation', {}).get('metrics', ['iou'])}
    mask_threshold: {config.task_configs.get('segmentation', {}).get('mask_threshold', 0.5)}
  
  detection:
    metrics: {config.task_configs.get('detection', {}).get('metrics', ['mAP'])}
    
  classification:
    metrics: {config.task_configs.get('classification', {}).get('metrics', ['accuracy'])}
"""
    return template


def print_model_info(info: dict):
    """Print formatted model information."""
    print("[INFO] Model Information")
    print("=" * 50)
    print(f"Version: {info['model']['version']}")
    print(f"Task: {info['task']['task_name']}")
    print(f"Supported Tasks: {', '.join(info['model']['supported_tasks'])}")
    print(f"Device: {info['model']['device']}")
    print(f"Input Size: {info['model']['input_size']}")
    print(f"Model Path: {info['model']['model_path']}")
    print()
    print("[INFO] CAM Configuration")
    print("-" * 30)
    print(f"Method: {info['cam_method']}")
    print(f"Target Component: {info['target_component']}")
    print()
    print("[INFO] Supported Metrics")
    print("-" * 30)
    print(f"{', '.join(info['task']['supported_metrics'])}")


def print_system_info():
    """Print system information."""
    import platform
    import torch
    
    print("[INFO] System Information")
    print("=" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def print_registry_info():
    """Print registry information."""
    from ..core.registry import get_registry_info
    
    info = get_registry_info()
    
    print("[INFO] Registry Information")
    print("=" * 50)
    print(f"Registered Models: {info['models']['count']}")
    for model in info['models']['registered']:
        print(f"  - {model}")
    
    print(f"\nRegistered Tasks: {info['tasks']['count']}")
    for task in info['tasks']['registered']:
        print(f"  - {task}")
    
    if info['tasks']['aliases']:
        print(f"\nTask Aliases:")
        for alias, task in info['tasks']['aliases'].items():
            print(f"  - {alias} -> {task}")


def print_general_info():
    """Print general YoloCAM information."""
    print("[INFO] YoloCAM - YOLO Model Analysis with Grad-CAM")
    print("=" * 50)
    print("Version: 0.1.0")
    print("Homepage: https://github.com/yourusername/yolocam")
    print("Documentation: https://yolocam.readthedocs.io")
    print()
    print("[Features]:")
    print("  - Multi-YOLO support (v8, v9, v10+)")
    print("  - Automatic model and task detection")
    print("  - Grad-CAM visualizations")
    print("  - Performance analysis")
    print("  - Plugin architecture")
    print("  - Comprehensive automation")
    print()
    print("Use 'yolocam --help' for available commands.")


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'ERROR' if args.quiet else 'INFO'
    setup_logging()
    logger = get_logger(__name__)
    
    # Execute command
    if args.command == 'analyze':
        return cmd_analyze(args, logger)
    elif args.command == 'config':
        return cmd_config(args, logger)
    elif args.command == 'info':
        return cmd_info(args, logger)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())