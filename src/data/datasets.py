"""Dataset loading utilities for multiple datasets."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple
import medmnist
from medmnist import INFO


def load_mnist_data(
    batch_size: int, 
    num_workers: int = 2, 
    subset_fn: Optional[Callable] = None, 
    shuffle: bool = True, 
    train: bool = True
) -> Tuple[DataLoader, datasets.MNIST]:
    """Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        subset_fn: Optional function to create subset
        shuffle: Whether to shuffle data
        train: Whether to load training or test set
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = datasets.MNIST(
        root="./data", 
        train=train, 
        download=True, 
        transform=transform
    )
    
    if subset_fn is not None:
        dataset = subset_fn(dataset)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    return dataloader, dataset


def _load_medmnist_data(
    dataset_name: str,
    batch_size: int,
    num_workers: int = 2,
    subset_fn: Optional[Callable] = None,
    shuffle: bool = True,
    train: bool = True
) -> Tuple[DataLoader, medmnist.dataset.MedMNIST]:
    """Generic MedMNIST dataset loader using proper medmnist interface.
    
    Args:
        dataset_name: Name of the medmnist dataset (e.g., 'pathmnist', 'octmnist')
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        subset_fn: Optional function to create subset
        shuffle: Whether to shuffle data
        train: Whether to load training or test set
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    # Use proper medmnist interface
    info = INFO[dataset_name.lower()]
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Dynamic class loading following medmnist best practices
    DataClass = getattr(medmnist, info["python_class"])
    split = 'train' if train else 'test'
    
    dataset = DataClass(
        split=split,
        transform=transform,
        download=True,
        root="./data"
    )
    
    if subset_fn is not None:
        dataset = subset_fn(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return dataloader, dataset


def load_pathmnist_data(
    batch_size: int,
    num_workers: int = 2,
    subset_fn: Optional[Callable] = None,
    shuffle: bool = True,
    train: bool = True
) -> Tuple[DataLoader, medmnist.dataset.MedMNIST]:
    """Load PathMNIST dataset using proper medmnist interface.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        subset_fn: Optional function to create subset
        shuffle: Whether to shuffle data
        train: Whether to load training or test set
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    return _load_medmnist_data(
        dataset_name="pathmnist",
        batch_size=batch_size,
        num_workers=num_workers,
        subset_fn=subset_fn,
        shuffle=shuffle,
        train=train
    )


def load_octmnist_data(
    batch_size: int,
    num_workers: int = 2,
    subset_fn: Optional[Callable] = None,
    shuffle: bool = True,
    train: bool = True
) -> Tuple[DataLoader, medmnist.dataset.MedMNIST]:
    """Load OCTMNIST dataset using proper medmnist interface.
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        subset_fn: Optional function to create subset
        shuffle: Whether to shuffle data
        train: Whether to load training or test set
        
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    return _load_medmnist_data(
        dataset_name="octmnist",
        batch_size=batch_size,
        num_workers=num_workers,
        subset_fn=subset_fn,
        shuffle=shuffle,
        train=train
    )


def get_subset(
    dataset, 
    n_per_class: int, 
    exclude_class: Optional[int] = None, 
    n_classes: int = 10, 
    add_query_element: bool = False
) -> Subset:
    """Get a subset of the dataset with n_per_class samples per class.
    
    Args:
        dataset: The original dataset
        n_per_class: The number of samples per class
        exclude_class: The class to exclude from the subset
        n_classes: Total number of classes
        add_query_element: Whether to add a query element from excluded class
        
    Returns:
        A subset of the dataset with n_per_class samples per class
    """
    indices = []
    
    # Get targets - handle different dataset types
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
        if isinstance(targets, torch.Tensor) and targets.dim() > 1:
            targets = targets.squeeze()
    else:
        # Extract targets manually
        targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    
    for i in range(n_classes):
        if i == exclude_class:
            continue
        class_indices = [j for j, label in enumerate(targets) if label == i]
        indices.extend(class_indices[:n_per_class])
    
    # Add query element if requested
    if add_query_element and exclude_class is not None:
        for i, label in enumerate(targets):
            if label == exclude_class:
                indices.append(i)
                break
    
    return Subset(dataset, indices)


def filter_query(dataset, query_label: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter and return the first sample with the given label.
    
    Args:
        dataset: Dataset to search
        query_label: Label to search for
        
    Returns:
        Tuple of (features, label)
    """
    for features, label in dataset:
        if label == query_label:
            return features, label
    
    raise ValueError(f"No sample found with label {query_label}")


def create_medmnist_loader(dataset_name: str):
    """Create a loader function for any MedMNIST dataset.
    
    Args:
        dataset_name: Name of the medmnist dataset
        
    Returns:
        Loading function compatible with the framework
    """
    def loader_func(batch_size: int, num_workers: int = 2, subset_fn: Optional[Callable] = None,
                   shuffle: bool = True, train: bool = True):
        return _load_medmnist_data(dataset_name, batch_size, num_workers, subset_fn, shuffle, train)
    
    return loader_func


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset configuration
    """
    if dataset_name == 'mnist':
        return {
            'num_channels': 1,
            'num_classes': 10,
            'load_fn': load_mnist_data,
            'img_shape': (1, 28, 28)
        }
    elif dataset_name in INFO:
        # Use medmnist INFO for accurate configuration
        info = INFO[dataset_name]
        return {
            'num_channels': info['n_channels'],
            'num_classes': len(info['label']),
            'load_fn': create_medmnist_loader(dataset_name),
            'img_shape': (info['n_channels'], 28, 28),  # MedMNIST datasets are 28x28
            'task': info['task'],
            'description': info['description']
        }
    else:
        # Fallback for legacy configurations
        legacy_configs = {
            'pathmnist': {
                'num_channels': 3,
                'num_classes': 9,
                'load_fn': load_pathmnist_data,
                'img_shape': (3, 28, 28)
            },
            'octmnist': {
                'num_channels': 1,
                'num_classes': 4,
                'load_fn': load_octmnist_data,
                'img_shape': (1, 28, 28)
            }
        }
        
        if dataset_name in legacy_configs:
            return legacy_configs[dataset_name]
    
    raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(INFO.keys()) + ['mnist']}") 