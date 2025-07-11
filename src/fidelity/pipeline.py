"""Fidelity evaluation pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Any, Dict, Set, Optional
import torch

from ..core.interfaces import FidelityMetric
from .metrics import FidelityPipeline as FidelityEvaluator


@dataclass
class FidelityConfig:
    """Configuration for fidelity evaluation."""
    metrics: Dict[str, Dict[str, Any]]
    selected_metrics: Set[str]


class FidelityPipeline:
    """Fidelity evaluation pipeline."""
    
    def __init__(self, config: FidelityConfig):
        """Initialize fidelity pipeline.
        
        Args:
            config: Fidelity configuration
        """
        self.config = config
        self.evaluator = FidelityEvaluator()
    
    def evaluate(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate fidelity metrics.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Dictionary containing metric values
        """
        return self.evaluator.compute_metrics(
            real_data=real_data,
            synthetic_data=synthetic_data,
            selected_metrics=self.config.selected_metrics
        ) 