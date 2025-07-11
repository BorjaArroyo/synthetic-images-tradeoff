"""Core interfaces for the tradeoff evaluation framework."""
from typing import Any, Dict, Protocol, runtime_checkable, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    """Interface for generative models."""
    
    @abstractmethod
    def generate(self, num_samples: int, **kwargs) -> torch.Tensor:
        """Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated samples tensor
        """
        pass


class PrivacyMetric(ABC):
    """Interface for privacy metrics."""
    
    @abstractmethod
    def compute(
        self,
        query_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor
    ) -> float:
        """Compute privacy metric.
        
        Args:
            query_data: Query data samples
            synthetic_data: Synthetic data samples
            real_data: Real data samples
            
        Returns:
            Privacy metric value
        """
        pass


class PrivacyAnalyzer(ABC):
    """Interface for privacy analyzers."""
    
    @abstractmethod
    def analyze(
        self,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze privacy properties.
        
        Args:
            synthetic_data: Synthetic data samples
            real_data: Real data samples
            
        Returns:
            Privacy analysis results
        """
        pass


class DataPipeline(ABC):
    """Interface for data processing pipelines."""
    
    @abstractmethod
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """Process input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        pass


class ModelPipeline(ABC):
    """Interface for model training pipelines."""
    
    @abstractmethod
    def train(self, data: torch.Tensor) -> torch.nn.Module:
        """Train a model.
        
        Args:
            data: Training data
            
        Returns:
            Trained model
        """
        pass


class EvaluationPipeline(ABC):
    """Interface for evaluation pipelines."""
    
    @abstractmethod
    def evaluate(
        self,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor
    ) -> Dict[str, Any]:
        """Evaluate synthetic data.
        
        Args:
            synthetic_data: Synthetic data samples
            real_data: Real data samples
            
        Returns:
            Evaluation results
        """
        pass


class PrivacyEstimator(ABC):
    """Interface for privacy estimators."""
    
    @abstractmethod
    def estimate(
        self,
        query_data: torch.Tensor,
        numerator_data: torch.Tensor,
        denominator_data: torch.Tensor
    ) -> float:
        """Estimate privacy metric.
        
        Args:
            query_data: Query data samples
            numerator_data: Synthetic data samples
            denominator_data: Real data samples
            
        Returns:
            Estimated privacy metric value
        """
        pass


class Visualizer(ABC):
    """Interface for visualizers."""
    
    @abstractmethod
    def visualize(
        self,
        query_data: torch.Tensor,
        synthetic_data: torch.Tensor,
        real_data: torch.Tensor
    ) -> None:
        """Visualize data.
        
        Args:
            query_data: Query data samples
            synthetic_data: Synthetic data samples
            real_data: Real data samples
        """
        pass


class FidelityMetric(ABC):
    """Interface for fidelity metrics."""
    
    @abstractmethod
    def compute(
        self,
        real_data: torch.Tensor,
        synthetic_data: torch.Tensor
    ) -> float:
        """Compute fidelity metric.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            
        Returns:
            Fidelity metric value
        """
        pass


class UtilityMetric(ABC):
    """Interface for utility metrics."""
    
    @abstractmethod
    def compute(
        self,
        synthetic_data: torch.Tensor,
        task_info: Dict[str, Any]
    ) -> float:
        """Compute utility metric.
        
        Args:
            synthetic_data: Synthetic data samples
            task_info: Task information for utility evaluation
            
        Returns:
            Utility metric value
        """
        pass


@runtime_checkable
class DataProcessor(Protocol):
    """Interface for data processing pipelines."""
    def process(self, data: torch.Tensor) -> torch.Tensor:
        """Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        ...


@runtime_checkable
class DimensionalityReducer(Protocol):
    """Interface for dimensionality reduction methods.
    
    This is the standard interface used throughout the codebase.
    Supports flexible implementation via duck typing:
    - transform() method for most reducers
    - encode() method for neural network-based reducers
    """
    def fit(self, data: torch.Tensor) -> None:
        """Fit the reducer to the data.
        
        Args:
            data: Data to fit the reducer on
        """
        ...
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data to lower dimensions.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data in lower dimensions
        """
        ... 