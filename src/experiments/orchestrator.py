"""Experiment orchestrator for evaluating synthetic data quality."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
import mlflow
import torch
from ..privacy.pipeline import PrivacyPipeline, PrivacyConfig
from ..privacy.reducer_comparison import UMAPReducerStandalone as UMAPReducer, TSNEReducerStandalone as TSNEReducer
from ..privacy.estimator import KNNPrivacyEstimator, KDEPrivacyEstimator, ClassifierPrivacyEstimator
# PrivacyVisualizer removed - use functions from src.visualization.privacy directly
from ..fidelity.pipeline import FidelityPipeline, FidelityConfig
from ..fidelity.metrics import FIDMetric, LPIPSMetric, PSNRMetric, ISMetric, SSIMMetric
from ..utility.pipeline import UtilityPipeline, UtilityConfig
from ..utility.metrics import ClassificationUtilityMetric, GradCAMUtilityMetric
from ..data.processor import ImageProcessor


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""
    experiment_name: str
    run_name: str
    privacy_config: Union[PrivacyConfig, Dict[str, Any]]
    fidelity_config: Union[FidelityConfig, Dict[str, Any]]
    utility_config: Union[UtilityConfig, Dict[str, Any]]
    data_config: Optional[Dict[str, Any]] = None
    processor: Optional[ImageProcessor] = None

    def __post_init__(self):
        """Convert dictionary configs to proper config objects if needed."""
        # Convert privacy config if it's a dictionary
        if isinstance(self.privacy_config, dict):
            # Create estimator instance if specified
            if 'estimator' in self.privacy_config:
                estimator_config = self.privacy_config['estimator']
                estimator_type = estimator_config.get('type')
                
                # Remove 'type' key from config before passing to constructor
                estimator_params = {k: v for k, v in estimator_config.items() if k != 'type'}
                
                if estimator_type == 'knn':
                    self.privacy_config['estimator'] = KNNPrivacyEstimator(**estimator_params)
                elif estimator_type == 'kde':
                    self.privacy_config['estimator'] = KDEPrivacyEstimator(**estimator_params)
                elif estimator_type == 'classifier':
                    self.privacy_config['estimator'] = ClassifierPrivacyEstimator(**estimator_params)
                else:
                    raise ValueError(f"Unknown estimator type: {estimator_type}")
            
            # Create reducer instance if specified
            if 'reducer' in self.privacy_config:
                reducer_config = self.privacy_config['reducer']
                reducer_type = reducer_config.get('type')
                
                # Remove 'type' key from config before passing to constructor
                reducer_params = {k: v for k, v in reducer_config.items() if k != 'type'}
                
                if reducer_type == 'umap':
                    self.privacy_config['reducer'] = UMAPReducer(**reducer_params)
                elif reducer_type == 'tsne':
                    self.privacy_config['reducer'] = TSNEReducer(**reducer_params)
            
            # Create visualizer instance if specified
            if 'visualizer' in self.privacy_config:
                visualizer_config = self.privacy_config['visualizer']
                if isinstance(visualizer_config, dict):
                    visualizer_type = visualizer_config.get('type')
                    if visualizer_type == 'privacy':
                        self.privacy_config['visualizer'] = PrivacyVisualizer()
            
            # Create config object
            self.privacy_config = PrivacyConfig(**self.privacy_config)
        
        # Convert fidelity config if it's a dictionary
        if isinstance(self.fidelity_config, dict):
            # Create metric instances if specified
            if 'metrics' in self.fidelity_config:
                metrics_config = self.fidelity_config['metrics']
                metric_instances = {}
                
                for metric_name, metric_config in metrics_config.items():
                    if metric_name == 'fid':
                        metric_instances[metric_name] = FIDMetric()
                    elif metric_name == 'lpips':
                        metric_instances[metric_name] = LPIPSMetric()
                    elif metric_name == 'psnr':
                        metric_instances[metric_name] = PSNRMetric()
                    elif metric_name == 'is':
                        metric_instances[metric_name] = ISMetric()
                    elif metric_name == 'ssim':
                        metric_instances[metric_name] = SSIMMetric()
                    else:
                        raise ValueError(f"Unknown fidelity metric: {metric_name}")
                
                self.fidelity_config['metrics'] = metric_instances
            
            self.fidelity_config = FidelityConfig(**self.fidelity_config)
        
        # Convert utility config if it's a dictionary
        if isinstance(self.utility_config, dict):
            # Create metric instances if specified
            if 'metrics' in self.utility_config:
                metrics_config = self.utility_config['metrics']
                metric_instances = {}
                
                for metric_name, metric_config in metrics_config.items():
                    if metric_name == 'classification':
                        # Extract configuration parameters
                        model = metric_config.get('model')
                        optimizer_cls = metric_config.get('optimizer', torch.optim.Adam)
                        criterion = metric_config.get('criterion')
                        epochs = metric_config.get('epochs', 5)
                        device = metric_config.get('device')
                        
                        # Create optimizer instance if model is provided
                        optimizer = None
                        if model is not None:
                            optimizer = optimizer_cls(model.parameters())
                        
                        metric_instances[metric_name] = ClassificationUtilityMetric(
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            epochs=epochs,
                            device=device
                        )
                    elif metric_name == 'gradcam':
                        model = metric_config.get('model')
                        target_layer = metric_config.get('target_layer')
                        device = metric_config.get('device')
                        
                        metric_instances[metric_name] = GradCAMUtilityMetric(
                            model=model,
                            target_layer=target_layer,
                            device=device
                        )
                    else:
                        raise ValueError(f"Unknown utility metric: {metric_name}")
                
                self.utility_config['metrics'] = metric_instances
            
            self.utility_config = UtilityConfig(**self.utility_config)
        
        # Create processor if data_config is provided
        if self.data_config is not None and self.processor is None:
            self.processor = ImageProcessor(**self.data_config)


class ExperimentOrchestrator:
    """Orchestrates the evaluation of synthetic data quality."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize experiment orchestrator.
        
        Args:
            config: Experiment configuration
            device: Device to use for computation
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize pipelines
        self.privacy_pipeline = PrivacyPipeline(config.privacy_config)
        if config.processor:
            self.privacy_pipeline.processor = config.processor
            
        self.fidelity_pipeline = FidelityPipeline(config.fidelity_config)
        self.utility_pipeline = UtilityPipeline(config.utility_config)
        
        # Set up MLflow
        mlflow.set_experiment(config.experiment_name)
    
    def run_experiment(
        self,
        query_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor,
        task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run experiment to evaluate synthetic data quality.
        
        Args:
            query_data: Query data samples
            synthetic_data: Synthetic data samples
            real_data: Real data samples
            task_info: Task information for utility evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        with mlflow.start_run(run_name=self.config.run_name):
            # Evaluate privacy
            privacy_results = self.privacy_pipeline.evaluate(
                query_data=query_data,
                synthetic_data=synthetic_data,
                real_data=real_data
            )
            
            # Evaluate fidelity
            fidelity_results = self.fidelity_pipeline.evaluate(
                real_data=real_data,
                synthetic_data=synthetic_data
            )
            
            # Evaluate utility
            utility_results = self.utility_pipeline.evaluate(
                synthetic_data=synthetic_data,
                task_info=task_info
            )
            
            # Log results
            results = {
                'privacy': privacy_results,
                'fidelity': fidelity_results,
                'utility': utility_results
            }
            
            for metric_type, metrics in results.items():
                for metric_name, value in metrics.items():
                    # Only log scalar numeric values, skip tensors and other objects
                    if isinstance(value, (int, float)) or (hasattr(value, 'item') and value.numel() == 1):
                        if hasattr(value, 'item'):
                            value = value.item()
                        mlflow.log_metric(f"{metric_type}/{metric_name}", value)
            
            return results


def create_experiment(
    privacy_config: Union[PrivacyConfig, Dict[str, Any]],
    fidelity_config: Union[FidelityConfig, Dict[str, Any]],
    utility_config: Union[UtilityConfig, Dict[str, Any]],
    experiment_name: str,
    run_name: str,
    data_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None
) -> ExperimentOrchestrator:
    """Create an experiment orchestrator.
    
    Args:
        privacy_config: Privacy evaluation configuration
        fidelity_config: Fidelity evaluation configuration
        utility_config: Utility evaluation configuration
        experiment_name: Name of the MLflow experiment
        run_name: Name of the MLflow run
        data_config: Optional data processing configuration
        device: Device to use for computation
        
    Returns:
        Configured experiment orchestrator
    """
    config = ExperimentConfig(
        privacy_config=privacy_config,
        fidelity_config=fidelity_config,
        utility_config=utility_config,
        experiment_name=experiment_name,
        run_name=run_name,
        data_config=data_config
    )
    return ExperimentOrchestrator(config, device) 