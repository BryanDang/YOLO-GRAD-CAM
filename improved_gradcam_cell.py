# Grad-CAM Analysis on Wound Segmentation Model
from yolocam import YoloCAMAnalyzer, YoloCAMConfig
import torch
import os

print("Performing Grad-CAM analysis on wound segmentation model...")

# Get the best model path
best_model_path = cfg.get_best_model_path(use_balanced=False)

if not best_model_path:
    raise FileNotFoundError(
        f"\n{'='*60}\n"
        f" NO MODEL FOUND FOR GRAD-CAM ANALYSIS\n"
        f"{'='*60}\n"
        f"Checked paths:\n"
        f"  - Balanced: {cfg.BALANCED_BEST_MODEL}\n"
        f"  - Initial: {cfg.INITIAL_BEST_MODEL}\n\n"
        f"Please ensure you have trained models or update paths.\n"
        f"{'='*60}"
    )

# Validate validation dataset exists
if not os.path.exists(cfg.valid_images_dir) or not os.path.exists(cfg.valid_masks_dir):
    raise FileNotFoundError(
        f"\n{'='*60}\n"
        f" VALIDATION DATASET NOT FOUND\n"
        f"{'='*60}\n"
        f"Cannot run Grad-CAM without validation data.\n"
        f"Please run data preparation cells first.\n"
        f"{'='*60}"
    )

print(f"Using model: {best_model_path}")

# Configure analyzer to test different layers
layers_to_test = ['backbone', 'neck', 'head']
successful_analyses = []

for layer_component in layers_to_test:
    print(f"\n{'='*60}")
    print(f"Analyzing {layer_component.upper()} features")
    print('='*60)
    
    config = YoloCAMConfig(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        target_layer_component=layer_component,
        num_best_examples=3,
        num_worst_examples=3,
        save_visualizations=True,
        output_dir=f'./gradcam_results_{layer_component}',
        # Add more detailed logging for debugging
        log_level='INFO',
        verbose=True
    )
    
    try:
        # Create analyzer
        analyzer = YoloCAMAnalyzer(best_model_path, config=config)
        
        # Analyze on validation set
        print(f"Running performance analysis...")
        results = analyzer.analyze_performance(
            cfg.valid_images_dir,
            cfg.valid_masks_dir
        )
        
        if not results:
            print(f"  Warning: No results found for {layer_component}")
            continue
            
        # Get examples
        best = analyzer.get_best_examples(results, num_examples=3)
        worst = analyzer.get_worst_examples(results, num_examples=3)
        
        print(f"\nFound {len(results)} total examples")
        print(f"Best examples: {len(best)}, Worst examples: {len(worst)}")
        
        # Print performance statistics
        scores = [r['score'] for r in results]
        print(f"Performance range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"Average IoU: {sum(scores)/len(scores):.3f}")
        
        # Visualize
        print("Creating visualizations...")
        best_paths = analyzer.visualize_results(best, f"best_{layer_component}")
        worst_paths = analyzer.visualize_results(worst, f"worst_{layer_component}")
        
        print(f"Visualizations saved to: {config.output_dir}")
        
        successful_analyses.append({
            'layer': layer_component,
            'results': results,
            'best': best,
            'worst': worst,
            'analyzer': analyzer
        })
        
        # Layer-specific insights
        if layer_component == 'backbone':
            print("\n📊 Backbone Analysis Insights:")
            print("- Focuses on low-level features (edges, textures)")
            print("- Good for understanding basic shape detection")
            print("- May show 'doughnut effect' around boundaries")
            
        elif layer_component == 'neck':
            print("\n📊 Neck Analysis Insights:")
            print("- Combines multi-scale features")
            print("- Better for contextual understanding")
            print("- Shows feature fusion patterns")
            
        elif layer_component == 'head':
            print("\n📊 Head Analysis Insights:")
            print("- Task-specific feature representation")
            print("- Most relevant for segmentation decisions")
            print("- Shows final prediction reasoning")
            
    except Exception as e:
        print(f"\n❌ Error analyzing {layer_component}: {e}")
        print("This may occur if the model architecture doesn't support this layer.")
        print("Continuing with next layer...")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*60}")
print(f"GRAD-CAM ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Successfully analyzed {len(successful_analyses)} out of {len(layers_to_test)} layers")

if successful_analyses:
    print("\n✅ Successful analyses:")
    for analysis in successful_analyses:
        layer = analysis['layer']
        num_results = len(analysis['results'])
        avg_score = sum(r['score'] for r in analysis['results']) / num_results
        print(f"  - {layer.upper()}: {num_results} examples, avg IoU: {avg_score:.3f}")
        
    print(f"\nNext steps:")
    print("1. Review visualization images in gradcam_results_* folders")
    print("2. Compare attention patterns across different layers")
    print("3. Use insights to improve model architecture or training")
else:
    print("\n❌ No successful analyses completed")
    print("Check model compatibility and validation data")