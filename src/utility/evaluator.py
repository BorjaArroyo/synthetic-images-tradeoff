"""Utility evaluation pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Any, Dict, Set, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.interfaces import UtilityMetric


class ClassificationUtilityMetric(UtilityMetric):
    """Utility metric based on classification performance."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        epochs: int = 100
    ):
        """Initialize the classification utility metric.
        
        Args:
            model: Classification model
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to use for computation
            epochs: Number of training epochs
        """
        self.model = model
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
    
    def compute(self, synthetic_data: torch.Tensor, task: Any) -> float:
        """Compute classification utility.
        
        Args:
            synthetic_data: Synthetic data samples
            task: Task information containing validation data
            
        Returns:
            Classification accuracy on validation data
        """
        # Move model to device
        self.model.to(self.device)
        
        # Create data loader for synthetic data
        if isinstance(synthetic_data, tuple):
            data, labels = synthetic_data
        else:
            data = synthetic_data
            labels = task.labels
        
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Train model
        self.model.train()
        for epoch in range(self.epochs):
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
        
        # Evaluate on validation data
        self.model.eval()
        val_loader = task.val_loader
        correct = 0
        total = 0
        
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(self.device), val_labels.to(self.device)
                outputs = self.model(val_data)
                _, predicted = torch.max(outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
        
        return correct / total


# GradCAMUtilityMetric removed - use the implementation in src.utility.metrics instead


class UtilityPipeline:
    """Pipeline for computing utility metrics."""
    
    def __init__(self):
        """Initialize with available metrics."""
        self.metrics = {}
    
    def add_metric(self, name: str, metric: UtilityMetric) -> None:
        """Add a utility metric to the pipeline.
        
        Args:
            name: Name of the metric
            metric: Metric implementation
        """
        self.metrics[name] = metric
    
    def compute_metrics(
        self,
        synthetic_data: torch.Tensor,
        task: Any,
        selected_metrics: Set[str]
    ) -> Dict[str, float]:
        """Compute selected utility metrics.
        
        Args:
            synthetic_data: Synthetic data samples
            task: Task information
            selected_metrics: Set of metric names to compute
            
        Returns:
            Dictionary of metric names and their values
        """
        results = {}
        for metric_name in selected_metrics:
            if metric_name in self.metrics:
                results[metric_name] = self.metrics[metric_name].compute(
                    synthetic_data,
                    task
                )
        return results 