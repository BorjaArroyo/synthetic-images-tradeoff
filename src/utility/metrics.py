"""Utility metrics for evaluating synthetic data quality."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..core.interfaces import UtilityMetric


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
            Metric value
        """
        pass


class ClassificationUtilityMetric(UtilityMetric):
    """Classification-based utility metric."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        epochs: int = 5,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        num_channels: int = 1,  # Default to MNIST (grayscale)
        num_classes: int = 10
    ):
        """Initialize classification utility metric.
        
        Args:
            model: Classification model (if None, uses default CNN)
            optimizer: Optimizer (if None, uses Adam)
            criterion: Loss criterion (if None, uses CrossEntropyLoss)
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use for computation
            num_channels: Number of input channels
            num_classes: Number of output classes
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Initialize model if not provided
        if model is None:
            # Use the same architecture as the original implementation
            self.model = nn.Sequential(
                nn.Conv2d(num_channels, 64, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.GroupNorm(32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.GroupNorm(32, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.GroupNorm(32, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.AdaptiveAvgPool2d((1, 1)),  # This handles any input size!
                nn.Flatten(),
                nn.Linear(512, num_classes)
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            self.optimizer = optimizer
        
        # Initialize criterion if not provided
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
    
    def _train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _evaluate(
        self,
        test_loader: DataLoader
    ) -> float:
        """Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Accuracy on test data
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def compute(
        self,
        synthetic_data: torch.Tensor,
        task_info: Dict[str, Any]
    ) -> float:
        """Compute classification accuracy on synthetic data.
        
        Args:
            synthetic_data: Synthetic data samples
            task_info: Dictionary containing task information:
                - task_type: Type of task (e.g., 'classification')
                - num_classes: Number of classes
                OR can contain explicit training/test data:
                - train_data: Training data tensor
                - train_labels: Training labels tensor
                - test_data: Test data tensor
                - test_labels: Test labels tensor
            
        Returns:
            Classification accuracy
        """
        # Check if explicit training/test data is provided
        if 'train_data' in task_info and 'test_data' in task_info:
            # Use provided data
            train_dataset = TensorDataset(
                task_info['train_data'],
                task_info['train_labels']
            )
            test_dataset = TensorDataset(
                task_info['test_data'],
                task_info['test_labels']
            )
        else:
            # Generate synthetic labels and use MNIST test data for evaluation
            from torchvision import datasets, transforms
            
            # Generate labels for synthetic data (assuming uniform distribution)
            num_classes = task_info.get('num_classes', 10)
            num_samples = len(synthetic_data)
            synthetic_labels = torch.randint(0, num_classes, (num_samples,))
            
            # Use synthetic data for training
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)
            
            # Load MNIST test data for evaluation
            transform = transforms.Compose([transforms.ToTensor()])
            test_mnist = datasets.MNIST(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )
            
            # Use all available test data - much simpler approach
            test_data = torch.stack([test_mnist[i][0] for i in range(len(test_mnist))])
            test_labels = torch.tensor([test_mnist[i][1] for i in range(len(test_mnist))])
            test_dataset = TensorDataset(test_data, test_labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Train model
        for epoch in range(self.epochs):
            self._train_epoch(train_loader)
        
        # Evaluate model
        accuracy = self._evaluate(test_loader)
        
        return accuracy


class GradCAMUtilityMetric(UtilityMetric):
    """GradCAM-based utility metric."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize GradCAM utility metric.
        
        Args:
            model: Classification model (if None, uses default CNN)
            target_layer: Target layer for GradCAM (if None, uses last conv layer)
            device: Device to use for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model if not provided
        if model is None:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 10)
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Set target layer if not provided
        if target_layer is None:
            self.target_layer = self.model[3]  # Last conv layer
        else:
            self.target_layer = target_layer
        
        # Register hooks for GradCAM
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def _compute_gradcam(
        self,
        image: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """Compute GradCAM for a single image.
        
        Args:
            image: Input image tensor
            target_class: Target class index
            
        Returns:
            GradCAM heatmap
        """
        # Forward pass
        output = self.model(image.unsqueeze(0))
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute GradCAM
        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.zeros_like(self.activations[0])
        
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        
        cam = torch.relu(cam)
        cam = nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        return cam.squeeze()
    
    def compute(
        self,
        synthetic_data: torch.Tensor,
        task_info: Dict[str, Any]
    ) -> float:
        """Compute GradCAM similarity between synthetic and real data.
        
        Args:
            synthetic_data: Synthetic data samples
            task_info: Dictionary containing task information:
                - task_type: Type of task (e.g., 'classification')
                - num_classes: Number of classes
                - real_data: Real data tensor
                - real_labels: Real data labels tensor
            
        Returns:
            Average GradCAM similarity score
        """
        real_data = task_info['real_data']
        real_labels = task_info['real_labels']
        
        # Compute GradCAM for real and synthetic data
        real_cams = []
        synth_cams = []
        
        for i in range(len(real_data)):
            real_cam = self._compute_gradcam(real_data[i], real_labels[i])
            synth_cam = self._compute_gradcam(synthetic_data[i], real_labels[i])
            
            real_cams.append(real_cam)
            synth_cams.append(synth_cam)
        
        # Stack CAMs
        real_cams = torch.stack(real_cams)
        synth_cams = torch.stack(synth_cams)
        
        # Compute similarity (cosine similarity)
        similarity = nn.functional.cosine_similarity(
            real_cams.view(len(real_cams), -1),
            synth_cams.view(len(synth_cams), -1)
        )
        
        return similarity.mean().item() 