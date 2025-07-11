"""Sampling utilities for generative models."""
import torch
from typing import Union
from .vae import VAE
from .gan import Generator
from .diffusion import DiffusionModel


def sample(
    model: Union[VAE, Generator, DiffusionModel],
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    return_labels: bool = False
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Sample images from either a VAE or conditional GAN model.

    Args:
        model: The model to sample from (VAE or Generator)
        labels: The labels for the samples
        device: The device to use for sampling
        batch_size: Batch size for generation
        return_labels: Whether to return labels along with samples

    Returns:
        Generated images, optionally with labels
    """
    num_samples = len(labels)
    counter = 0
    model.eval()
    
    with torch.no_grad():
        samples = []
        while counter < num_samples:
            this_batch_size = min(batch_size, num_samples - counter)
            y = labels[counter:counter + this_batch_size].to(device)

            # VAE vs GAN vs Diffusion logic matching original implementation
            if isinstance(model, VAE):
                z = torch.randn(this_batch_size, model.latent_dim).to(device)
                this_samples = model.decoder(z, y)
            elif isinstance(model, Generator):
                z = torch.randn(this_batch_size, model.latent_dim).to(device)
                this_samples = model(z, y)
                # No range conversion needed - Generator uses Sigmoid output [0,1]
            elif isinstance(model, DiffusionModel):
                # Use TeaPearce's full sampling process for best quality
                # With classifier-free guidance (guide_w=2.0 for good balance)
                this_samples = model.sample(this_batch_size, y, guide_w=2.0)
                # Ensure samples are in [0,1] range for consistency
                this_samples = torch.clamp(this_samples, 0.0, 1.0)
            else:
                raise ValueError("Unsupported model type for sampling.")

            samples.append(this_samples.cpu())
            counter += this_batch_size

    samples = torch.cat(samples, dim=0)
    
    if return_labels:
        return samples, labels
    return samples


def generate_data(
    model: Union[VAE, Generator, DiffusionModel],
    n_classes: int,
    samples_per_class: int,
    device: torch.device,
    exclude_class: int = None,
    return_labels: bool = False
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic data with specified distribution across classes.
    
    Args:
        model: The generative model
        n_classes: Total number of classes
        samples_per_class: Number of samples per class
        device: Device to generate on
        exclude_class: Class to exclude from generation
        return_labels: Whether to return labels
        
    Returns:
        Generated data, optionally with labels
    """
    classes = [i for i in range(n_classes) if i != exclude_class]
    labels = torch.tensor(classes).repeat_interleave(samples_per_class)
    
    return sample(model, labels, device, return_labels=return_labels) 