"""Data processing pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ..core.interfaces import DataProcessor


@dataclass
class ProcessingStep:
    """A single processing step in the pipeline."""
    name: str
    processor: Callable[[torch.Tensor], torch.Tensor]
    description: Optional[str] = None


class PipelineProcessor(DataProcessor):
    """A pipeline of data processing steps."""
    
    def __init__(self, steps: Optional[List[ProcessingStep]] = None):
        """Initialize the processor with optional steps.
        
        Args:
            steps: List of processing steps to apply
        """
        self.steps = steps or []
    
    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline.
        
        Args:
            step: Processing step to add
        """
        self.steps.append(step)
    
    def process(self, data: Union[torch.Tensor, DataLoader, Dataset]) -> torch.Tensor:
        """Process the data through all steps in the pipeline.
        
        Args:
            data: Input data to process (Tensor, DataLoader, or Dataset)
            
        Returns:
            Processed data as tensor
        """
        # Convert DataLoader or Dataset to tensor
        if isinstance(data, DataLoader):
            data = next(iter(data))[0] if isinstance(next(iter(data)), (tuple, list)) else next(iter(data))
        elif isinstance(data, Dataset):
            data = data[0][0] if isinstance(data[0], (tuple, list)) else data[0]
        
        # Process through steps
        processed_data = data
        for step in self.steps:
            processed_data = step.processor(processed_data)
        return processed_data


class ImageProcessor:
    """Processes images for evaluation."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (299, 299),
        normalize: bool = True,
        device: Optional[torch.device] = None
    ):
        """Initialize image processor.
        
        Args:
            target_size: Target size for resizing images
            normalize: Whether to normalize images
            device: Device to use for computation
        """
        self.target_size = target_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
    
    def process(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Process images.
        
        Args:
            images: Input images tensor
            
        Returns:
            Processed images tensor
        """
        # Move to device
        images = images.to(self.device)
        
        # Resize if needed
        if images.shape[-2:] != self.target_size:
            images = F.resize(
                images,
                self.target_size,
                interpolation=F.InterpolationMode.BILINEAR
            )
        
        # Normalize if needed
        if self.normalize:
            # Convert to RGB if grayscale
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Normalize using ImageNet stats
            images = F.normalize(
                images,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        return images 