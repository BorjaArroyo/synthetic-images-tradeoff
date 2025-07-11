"""Visualization utilities for utility analysis."""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import cv2

# Import visualization module for automatic font configuration
import src.visualization as viz


class GradCAMVisualizer:
    """Gradient-weighted Class Activation Mapping (GradCAM) visualizer."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: Optional[torch.device] = None):
        """Initialize GradCAM visualizer.
        
        Args:
            model: The trained model
            target_layer: The layer to compute GradCAM for
            device: Device to use for computation
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_backward_hook(self._backward_hook)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _forward_hook(self, module, input, output):
        """Forward hook to capture activations."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Compute GradCAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], device=self.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.cpu().numpy()
    
    def visualize(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """Generate GradCAM visualization overlaid on input image.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            alpha: Transparency for overlay
            colormap: Colormap for heatmap
            
        Returns:
            Visualization as numpy array
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Convert input to numpy and normalize
        input_np = input_tensor.squeeze().cpu().numpy()
        if input_np.ndim == 3:  # RGB
            input_np = np.transpose(input_np, (1, 2, 0))
        
        # Normalize image to [0, 1]
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        # Resize CAM to match input dimensions
        h, w = input_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = plt.get_cmap(colormap)(cam_resized)[:, :, :3]  # Remove alpha channel
        
        # Ensure input is RGB
        if input_np.ndim == 2:  # Grayscale
            input_rgb = np.stack([input_np] * 3, axis=2)
        else:
            input_rgb = input_np
        
        # Overlay heatmap on image
        visualization = alpha * heatmap + (1 - alpha) * input_rgb
        
        return np.clip(visualization, 0, 1)
    
    def plot_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """Plot GradCAM visualization.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
            save_path: Path to save the plot
            title: Title for the plot
        """
        # Font configuration handled automatically by visualization module
        
        # Generate visualization
        visualization = self.visualize(input_tensor, target_class)
        
        # Get original image
        input_np = input_tensor.squeeze().cpu().numpy()
        if input_np.ndim == 3:
            input_np = np.transpose(input_np, (1, 2, 0))
        
        # Normalize
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
        
        # Ensure RGB
        if input_np.ndim == 2:
            input_rgb = np.stack([input_np] * 3, axis=2)
        else:
            input_rgb = input_np
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(input_rgb)
        axes[0].axis('off')
        
        # GradCAM overlay
        axes[1].imshow(visualization)
        axes[1].axis('off')
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def cleanup(self):
        """Remove hooks to free memory."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def log_gradcam_images(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    save_dir: str = "figs",
    prefix: str = "gradcam"
) -> List[str]:
    """Generate and save GradCAM visualizations for a batch of images.
    
    Args:
        model: The trained model
        target_layer: Layer to compute GradCAM for
        images: Batch of images
        labels: True labels
        device: Device to use
        save_dir: Directory to save images
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    gradcam = GradCAMVisualizer(model, target_layer, device)
    saved_paths = []
    
    for idx, (image, label) in enumerate(zip(images, labels)):
        # Add batch dimension
        image_batch = image.unsqueeze(0)
        
        # Generate visualization
        save_path = os.path.join(save_dir, f"{prefix}_{idx}_label_{label.item()}.png")
        gradcam.plot_gradcam(
            image_batch,
            target_class=label.item(),
            save_path=save_path,
            title=f"GradCAM - Label: {label.item()}"
        )
        saved_paths.append(save_path)
    
    gradcam.cleanup()
    return saved_paths 