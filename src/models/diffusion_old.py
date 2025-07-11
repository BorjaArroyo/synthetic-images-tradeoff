"""
Diffusion model implementation for the tradeoff evaluation framework.

This implements a simplified DDPM (Denoising Diffusion Probabilistic Model) 
inspired by the DPDM paper: https://github.com/nv-tlabs/DPDM

The model is designed to work with the existing training and evaluation pipeline,
supporting both conditional and unconditional generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def create_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Create linear beta schedule for diffusion process."""
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract coefficients from a based on timestep t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for diffusion models with time conditioning.
    
    This implementation is compatible with the existing dataset shapes and
    works with the DPSGD training pipeline.
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10, 
                 model_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        # Class embedding for conditional generation
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, model_channels * 4)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList([
            # Level 1: 28x28 -> 14x14
            nn.Sequential(
                ResBlock(model_channels, model_channels * 2, model_channels * 4, dropout),
                nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1),
            ),
            # Level 2: 14x14 -> 7x7  
            nn.Sequential(
                ResBlock(model_channels * 2, model_channels * 4, model_channels * 4, dropout),
                nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1),
            ),
        ])
        
        # Middle block
        self.middle_block = ResBlock(model_channels * 4, model_channels * 4, model_channels * 4, dropout)
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList([
            # Level 2: 7x7 -> 14x14
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResBlock(model_channels * 8, model_channels * 2, model_channels * 4, dropout),
            ),
            # Level 1: 14x14 -> 28x28
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ResBlock(model_channels * 4, model_channels, model_channels * 4, dropout),
            ),
        ])
        
        # Final layer to handle concatenated skip connection
        self.final_resblock = ResBlock(model_channels * 2, model_channels, model_channels * 4, dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, model_channels), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )
    
    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of U-Net.
        
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
            y: Class labels [B] (optional for conditional generation)
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # Add class embedding if provided
        if y is not None and hasattr(self, 'class_embed'):
            c_emb = self.class_embed(y)
            t_emb = t_emb + c_emb
        
        # Input projection
        h = self.input_proj(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for block in self.encoder:
            h = block[0](h, t_emb)  # ResBlock
            skip_connections.append(h)
            h = block[1](h)  # Downsample
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Decoder
        for block in self.decoder:
            h = block[0](h)  # Upsample
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block[1](h, t_emb)  # ResBlock
        
        # Add final skip connection and process
        h = torch.cat([h, skip_connections.pop()], dim=1)
        h = self.final_resblock(h, t_emb)
        
        # Output projection
        return self.output_proj(h)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.shortcut(x)


class DiffusionModel(nn.Module):
    """
    Complete diffusion model implementation compatible with the tradeoff framework.
    
    This implements DDPM with support for:
    - Conditional generation (class labels)
    - DPSGD training compatibility
    - Integration with existing evaluation metrics
    """
    
    def __init__(self, num_channels: int = 1, num_classes: int = 10, 
                 timesteps: int = 1000, model_channels: int = 64,
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.timesteps = timesteps
        
        # U-Net model
        self.model = DiffusionUNet(
            in_channels=num_channels,
            num_classes=num_classes,
            model_channels=model_channels
        )
        
        # Precompute diffusion parameters
        self.register_buffer('betas', create_beta_schedule(timesteps, beta_start, beta_end))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', torch.cat([torch.ones(1), self.alphas_cumprod[:-1]]))
        
        # Precompute values for sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # For sampling
        self.register_buffer('posterior_variance', 
                           self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass during training."""
        return self.model(x, t, y)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: sample x_t from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_start: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion loss for training.
        
        Args:
            x_start: Original images [B, C, H, W]
            y: Class labels [B] (optional)
            
        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, y)
        
        # Compute MSE loss
        return F.mse_loss(predicted_noise, noise)
    
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) - single reverse diffusion step.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
        
        # Predict noise
        predicted_noise = self.model(x, t, y)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample(self, batch_size: int, y: Optional[torch.Tensor] = None, 
               num_inference_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate samples using the reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            y: Class labels [batch_size] (optional)
            num_inference_steps: Number of denoising steps (defaults to self.timesteps)
            
        Returns:
            Generated samples [batch_size, channels, height, width]
        """
        device = next(self.parameters()).device
        
        if num_inference_steps is None:
            num_inference_steps = self.timesteps
        
        # Start from pure noise
        x = torch.randn(batch_size, self.num_channels, 28, 28, device=device)
        
        # Create proper sampling schedule with integer timesteps
        if num_inference_steps == self.timesteps:
            # Use all timesteps in reverse order
            timesteps = torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
        else:
            # Uniformly subsample timesteps, ensuring they're integers
            step_size = self.timesteps // num_inference_steps
            timesteps = torch.arange(self.timesteps - 1, -1, -step_size, device=device, dtype=torch.long)
            # Ensure we end at timestep 0
            if timesteps[-1] != 0:
                timesteps = torch.cat([timesteps, torch.tensor([0], device=device, dtype=torch.long)])
        
        self.eval()
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                x = self.p_sample(x, t_batch, y)
                
                # Add stability checks to prevent NaN/exploding values
                x = torch.clamp(x, -5.0, 5.0)  # Prevent extreme values
                
                # Optional: print progress for debugging
                if i % (len(timesteps) // 4) == 0:
                    print(f"   Sampling step {i+1}/{len(timesteps)}, t={t}, range=[{x.min():.3f}, {x.max():.3f}]")
        
        # Final output normalization - ensure samples are in valid image range
        # No additional scaling needed since training data should be in [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        
        return x 