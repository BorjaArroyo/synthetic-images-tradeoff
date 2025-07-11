"""Privacy evaluation pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.interfaces import PrivacyEstimator, DimensionalityReducer, Visualizer
from ..data.processor import ImageProcessor


@dataclass
class PrivacyConfig:
    """Configuration for privacy evaluation."""
    estimator: PrivacyEstimator
    reducer: Optional[DimensionalityReducer] = None
    visualizer: Optional[Visualizer] = None
    processor: Optional[ImageProcessor] = None


class PrivacyPipeline:
    """Privacy evaluation pipeline."""
    
    def __init__(self, config: PrivacyConfig):
        """Initialize privacy pipeline.
        
        Args:
            config: Privacy configuration
        """
        self.estimator = config.estimator
        self.reducer = config.reducer
        self.visualizer = config.visualizer
        self.processor = config.processor
        self._reducer_fitted = False
    
    def _transform_data(self, reducer, data: torch.Tensor):
        """Transform data using reducer, handling different interfaces.
        
        This uses the same flexible pattern as the working reducer_comparison.py
        """
        if hasattr(reducer, 'transform'):
            return torch.tensor(reducer.transform(data))
        elif hasattr(reducer, 'encode'):
            return reducer.encode(data)
        else:
            raise ValueError(f"Reducer {type(reducer)} has no transform or encode method")
    
    def fit_reducer(self, data: torch.Tensor) -> None:
        """Fit the reducer once on the reference data.
        
        Args:
            data: Reference data to fit the reducer on
        """
        # NOTE: Reducers should already be fitted by ReducerManager on real training data
        # This method is kept for interface compatibility but should not refit
        if self.reducer is not None and not self._reducer_fitted:
            print("⚠️  Warning: Reducer should already be fitted by ReducerManager")
            self._reducer_fitted = True
    
    def evaluate(
        self,
        query_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Evaluate privacy metrics.
        
        Args:
            query_data: Query data samples
            synthetic_data: Synthetic data samples
            real_data: Real data samples
            
        Returns:
            Dictionary containing metric values and visualization data
        """
        # Process data if processor is available
        if self.processor is not None:
            query_data = self.processor.process(query_data)
            synthetic_data = self.processor.process(synthetic_data)
            real_data = self.processor.process(real_data)
        
        # Reduce dimensionality if reducer is available
        if self.reducer is not None:
            # Reducer should already be fitted by ReducerManager on real training data
            # Just mark as fitted and transform the data using flexible interface
            self._reducer_fitted = True
            
            # Transform each dataset separately using the fitted reducer
            query_reduced = self._transform_data(self.reducer, query_data)
            synthetic_reduced = self._transform_data(self.reducer, synthetic_data)
            real_reduced = self._transform_data(self.reducer, real_data)
        else:
            query_reduced = query_data
            synthetic_reduced = synthetic_data
            real_reduced = real_data
        
        # Estimate privacy
        epsilon = self.estimator.estimate(
            query_data=query_reduced,
            numerator_data=synthetic_reduced,
            denominator_data=real_reduced
        )
        
        # Visualize if visualizer is available
        if self.visualizer is not None:
            self.visualizer.visualize(
                query_data=query_reduced,
                synthetic_data=synthetic_reduced,
                real_data=real_reduced
            )
        
        # Return results
        results = {
            'epsilon': epsilon,
            'query_data': query_reduced,
            'synthetic_data': synthetic_reduced,
            'real_data': real_reduced
        }
        
        return results 