"""Integration tests for the main analyzer functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch

from yolocam.core.analyzer import YoloCAMAnalyzer
from yolocam.core.config import YoloCAMConfig
from yolocam.core.registry import YOLOModelRegistry, TaskRegistry, register_yolo_model, register_yolo_task
from yolocam.models.base_model import BaseYOLOModel
from yolocam.tasks.base_task import BaseTask


class IntegrationTestModel(BaseYOLOModel):
    """Test model for integration testing."""
    
    def __init__(self, model_path: str, config: YoloCAMConfig):
        super().__init__(model_path, config)
        self._model = Mock()
    
    def load_model(self, model_path: str):
        return self._model
    
    def get_target_layers(self, component: str = 'backbone'):
        return [Mock()]
    
    def get_inference_model(self):
        def mock_inference(image_path, **kwargs):
            # Return mock results with masks
            result = Mock()
            result.masks = Mock()
            result.masks.data = torch.rand(1, 240, 320) > 0.5
            result.masks.data.cpu = lambda: result.masks.data
            result.masks.data.numpy = lambda: result.masks.data.numpy()
            return [result]
        return mock_inference
    
    def preprocess_input(self, image_path: str):
        return torch.rand(1, 3, self.config.model_input_size, self.config.model_input_size)
    
    def detect_task(self) -> str:
        return 'segmentation'
    
    @property
    def supported_tasks(self):
        return ['segmentation']
    
    @property
    def version(self):
        return 'integration_test'


class IntegrationTestTask(BaseTask):
    """Test task for integration testing."""
    
    def compute_performance_metric(self, prediction, ground_truth, **kwargs):
        # Mock IoU calculation
        return np.random.random()
    
    def create_cam_target_function(self, **kwargs):
        def target_function(model_output):
            return model_output.sum()
        return target_function
    
    def load_ground_truth(self, path: str):
        return np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255
    
    def visualize_results(self, image, prediction, ground_truth, cam_output, **kwargs):
        return np.random.randint(0, 255, (480, 1920, 3), dtype=np.uint8)  # 1x4 layout
    
    @property
    def task_name(self):
        return 'integration_test_task'
    
    @property
    def supported_metrics(self):
        return ['iou']
    
    @property
    def required_ground_truth_format(self):
        return 'mask_png'


@pytest.fixture(autouse=True)
def setup_test_registry():
    """Setup test registry for integration tests."""
    # Register test components
    register_yolo_model('integration_test', pattern=r'integration.*')(IntegrationTestModel)
    register_yolo_task('integration_test_task')(IntegrationTestTask)
    
    yield
    
    # Cleanup is handled by clear_registries fixture


class TestAnalyzerIntegration:
    """Integration tests for YoloCAMAnalyzer."""
    
    def test_analyzer_initialization(self, mock_model_checkpoint, sample_config):
        """Test analyzer initialization with mocked components."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        assert analyzer.model_version == 'integration_test'
        assert analyzer.task_name == 'integration_test_task'
        assert analyzer.config == sample_config
        assert analyzer.output_dir.exists()
    
    def test_analyzer_auto_task_detection(self, mock_model_checkpoint, sample_config):
        """Test analyzer with automatic task detection."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='auto',
                config=sample_config
            )
        
        # Should detect 'segmentation' from model, which maps to our registered task
        assert analyzer.task_name == 'segmentation'
    
    def test_analyzer_model_task_compatibility_validation(self, mock_model_checkpoint, sample_config):
        """Test that analyzer validates model-task compatibility."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            # This should work - our test model supports segmentation
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
            assert analyzer is not None
    
    def test_analyzer_unsupported_model_version(self, mock_model_checkpoint, sample_config):
        """Test analyzer with unsupported model version."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='unsupported_version'):
            with pytest.raises(ValueError, match="Unsupported model version"):
                YoloCAMAnalyzer(
                    str(mock_model_checkpoint),
                    task='integration_test_task',
                    config=sample_config
                )
    
    def test_analyzer_unsupported_task(self, mock_model_checkpoint, sample_config):
        """Test analyzer with unsupported task."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            with pytest.raises(ValueError, match="Unsupported task"):
                YoloCAMAnalyzer(
                    str(mock_model_checkpoint),
                    task='unsupported_task',
                    config=sample_config
                )
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_analyze_performance_integration(self, mock_cam_wrapper, mock_model_checkpoint, sample_config, sample_dataset):
        """Test performance analysis integration."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        results = analyzer.analyze_performance(
            str(sample_dataset['images_dir']),
            str(sample_dataset['masks_dir'])
        )
        
        assert isinstance(results, list)
        assert len(results) == sample_dataset['num_samples']
        
        # Check that results are sorted (worst to best)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores)
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_generate_cam_integration(self, mock_cam_wrapper, mock_model_checkpoint, sample_config, sample_image):
        """Test CAM generation integration."""
        # Mock CAM wrapper to return expected output
        mock_cam_instance = Mock()
        mock_cam_instance.generate_cam.return_value = np.random.rand(240, 320)
        mock_cam_wrapper.return_value = mock_cam_instance
        
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        cam_output = analyzer.generate_cam(str(sample_image))
        
        assert isinstance(cam_output, np.ndarray)
        assert cam_output.shape == (240, 320)
        mock_cam_instance.generate_cam.assert_called_once()
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_visualize_results_integration(self, mock_cam_wrapper, mock_model_checkpoint, sample_config, sample_performance_results):
        """Test visualization integration."""
        # Mock CAM wrapper
        mock_cam_instance = Mock()
        mock_cam_instance.generate_cam.return_value = np.random.rand(240, 320)
        mock_cam_wrapper.return_value = mock_cam_instance
        
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        # Mock file existence checks
        with patch('os.path.exists', return_value=True):
            visualization_paths = analyzer.visualize_results(
                sample_performance_results[:3],
                analysis_type="test"
            )
        
        assert isinstance(visualization_paths, list)
        # Should be empty since save_visualizations is False in sample_config
        assert len(visualization_paths) == 0
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_get_best_worst_examples(self, mock_cam_wrapper, mock_model_checkpoint, sample_config, sample_performance_results):
        """Test getting best and worst examples."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        best_examples = analyzer.get_best_examples(sample_performance_results)
        worst_examples = analyzer.get_worst_examples(sample_performance_results)
        
        assert len(best_examples) == sample_config.num_best_examples
        assert len(worst_examples) == sample_config.num_worst_examples
        
        # Best examples should have higher scores than worst
        assert all(b['score'] >= w['score'] for b in best_examples for w in worst_examples)
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_analyzer_info(self, mock_cam_wrapper, mock_model_checkpoint, sample_config):
        """Test getting analyzer information."""
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='integration_test_task',
                config=sample_config
            )
        
        info = analyzer.get_info()
        
        assert isinstance(info, dict)
        assert 'model' in info
        assert 'task' in info
        assert 'cam_method' in info
        assert 'config' in info
        
        assert info['model']['version'] == 'integration_test'
        assert info['task']['task_name'] == 'integration_test_task'
        assert info['cam_method'] == sample_config.cam_method


class TestFullWorkflow:
    """Test complete workflow scenarios."""
    
    @patch('yolocam.cam.gradcam.GradCAMWrapper')
    def test_complete_analysis_workflow(self, mock_cam_wrapper, mock_model_checkpoint, sample_dataset, temp_dir):
        """Test complete analysis workflow from start to finish."""
        # Setup config with saving enabled
        config = YoloCAMConfig(
            device='cpu',
            model_input_size=320,
            num_best_examples=2,
            num_worst_examples=2,
            save_visualizations=True,
            save_intermediate_results=True,
            output_dir=str(temp_dir / "output")
        )
        
        # Mock CAM wrapper
        mock_cam_instance = Mock()
        mock_cam_instance.generate_cam.return_value = np.random.rand(240, 320)
        mock_cam_wrapper.return_value = mock_cam_instance
        
        with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
            # Initialize analyzer
            analyzer = YoloCAMAnalyzer(
                str(mock_model_checkpoint),
                task='auto',
                config=config
            )
            
            # Run performance analysis
            results = analyzer.analyze_performance(
                str(sample_dataset['images_dir']),
                str(sample_dataset['masks_dir'])
            )
            
            assert len(results) == sample_dataset['num_samples']
            
            # Get best and worst examples
            best_examples = analyzer.get_best_examples(results)
            worst_examples = analyzer.get_worst_examples(results)
            
            assert len(best_examples) == 2
            assert len(worst_examples) == 2
            
            # Mock file existence for visualization
            with patch('os.path.exists', return_value=True):
                # Visualize results
                best_viz_paths = analyzer.visualize_results(best_examples, "best")
                worst_viz_paths = analyzer.visualize_results(worst_examples, "worst")
                
                # Check that visualizations were created (paths returned)
                assert isinstance(best_viz_paths, list)
                assert isinstance(worst_viz_paths, list)
            
            # Check that output directory was created
            assert analyzer.output_dir.exists()
            
            # Get analyzer info
            info = analyzer.get_info()
            assert info['model']['version'] == 'integration_test'
    
    def test_configuration_driven_behavior(self, mock_model_checkpoint, sample_dataset):
        """Test that configuration properly drives analyzer behavior."""
        # Test with different configurations
        configs = [
            YoloCAMConfig(cam_method='gradcam', target_layer_component='backbone'),
            YoloCAMConfig(cam_method='eigencam', target_layer_component='neck'),
        ]
        
        for config in configs:
            with patch('yolocam.core.registry.YOLOModelRegistry.auto_detect', return_value='integration_test'):
                with patch('yolocam.cam.gradcam.GradCAMWrapper') as mock_cam:
                    analyzer = YoloCAMAnalyzer(
                        str(mock_model_checkpoint),
                        task='integration_test_task',
                        config=config
                    )
                    
                    # Verify that configuration was applied
                    assert analyzer.config.cam_method == config.cam_method
                    assert analyzer.config.target_layer_component == config.target_layer_component
                    
                    # Verify that CAM wrapper was initialized with correct config
                    mock_cam.assert_called_once()
                    call_args = mock_cam.call_args
                    assert call_args[1]['config'] == config