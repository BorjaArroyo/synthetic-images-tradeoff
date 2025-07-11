"""Utility evaluation pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set
import torch

from ..core.interfaces import UtilityMetric


@dataclass
class UtilityConfig:
    """Configuration for utility evaluation."""
    metrics: Dict[str, UtilityMetric]
    selected_metrics: Set[str]


class UtilityPipeline:
    """Pipeline for utility evaluation."""
    
    def __init__(self, config: UtilityConfig):
        """Initialize utility pipeline.
        
        Args:
            config: Utility configuration
        """
        self.metrics = config.metrics
        self.selected_metrics = config.selected_metrics
    
    def evaluate(
        self,
        synthetic_data: torch.Tensor,
        task_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate utility metrics.
        
        Args:
            synthetic_data: Synthetic data samples
            task_info: Task information for utility evaluation
            
        Returns:
            Dictionary containing metric values
        """
        results = {}
        
        for metric_name in self.selected_metrics:
            if metric_name not in self.metrics:
                raise ValueError(f"Metric {metric_name} not found in available metrics")
            
            metric = self.metrics[metric_name]
            results[metric_name] = metric.compute(synthetic_data, task_info)
        
        return results 