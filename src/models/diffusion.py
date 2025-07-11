"""
Diffusion model implementation based on TeaPearce's working Conditional_Diffusion_MNIST.

This implements the proven architecture from:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Key features:
- Simple U-Net architecture with ResidualConvBlocks
- Proper classifier-free guidance with context masking
- Element-wise multiplication for conditioning (crucial for guidance)
- Linear beta schedule
- Simplified embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ResidualConvBlock(nn.Module):
    """Standard ResNet style convolutional block"""
    
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        
        # Use GroupNorm instead of BatchNorm for differential privacy compatibility
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # Add correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414  # Scale factor from TeaPearce
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """Process and downscale the image feature maps"""
    
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    """Process and upscale the image feature maps"""
    
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    """Generic one layer FC NN for embedding things"""
    
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """U-Net with proper context conditioning for classifier-free guidance"""
    
    def __init__(self, in_channels=1, n_feat=128, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # Time and context embeddings
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        """
        Forward pass with proper conditioning.
        
        Args:
            x: Noisy images [B, C, H, W]
            c: Context labels [B] 
            t: Timesteps [B, 1, 1, 1] (normalized)
            context_mask: Which samples to mask context [B]
        """
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # Convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # Mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1*(1-context_mask))  # Flip 0 <-> 1
        c = c * context_mask
        
        # Embed context and time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        # KEY: Use element-wise multiplication for conditioning (crucial for guidance)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """Returns pre-computed schedules for DDPM sampling, training process."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DiffusionModel(nn.Module):
    """
    Complete DDPM implementation with classifier-free guidance.
    
    Based on TeaPearce's working implementation with proper guidance sampling.
    """
    
    def __init__(self, num_channels: int = 1, num_classes: int = 10, 
                 timesteps: int = 400, n_feat: int = 128, drop_prob: float = 0.1):
        super(DiffusionModel, self).__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.drop_prob = drop_prob
        
        # Create the U-Net model
        self.nn_model = ContextUnet(
            in_channels=num_channels, 
            n_feat=n_feat, 
            n_classes=num_classes
        )
        
        # Register DDPM schedules as buffers
        # Use TeaPearce's proven hyperparameters
        schedules = ddpm_schedules(1e-4, 0.02, timesteps)
        for k, v in schedules.items():
            self.register_buffer(k, v)
            
        self.loss_mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training"""
        return self.nn_model(x, y, t, torch.zeros_like(y))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample q(x_t|x_0) - forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Ensure t is the right shape for indexing
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if len(t.shape) == 1:
            t = t.long()
            
        sqrt_alphas_cumprod_t = self.sqrtab[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrtmab[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute training loss with proper context dropout"""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample random timesteps
        t = torch.randint(1, self.timesteps + 1, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise)
        
        # Context dropout for classifier-free guidance
        context_mask = torch.bernoulli(
            torch.zeros_like(y, dtype=torch.float) + self.drop_prob
        ).to(device)
        
        # Normalize timesteps as in TeaPearce
        t_normalized = (t.float() / self.timesteps).view(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.nn_model(x_t, y, t_normalized, context_mask)
        
        return self.loss_mse(noise, predicted_noise)

    def sample(self, batch_size: int, y: Optional[torch.Tensor] = None, 
               num_inference_steps: Optional[int] = None, guide_w: float = 0.0) -> torch.Tensor:
        """
        Sample from the model with classifier-free guidance.
        
        Args:
            batch_size: Number of samples
            y: Class labels [batch_size]  
            num_inference_steps: Number of denoising steps (uses self.timesteps if None)
            guide_w: Guidance weight (0.0 = no guidance, higher = more guidance)
        """
        device = next(self.parameters()).device
        
        if num_inference_steps is None:
            num_inference_steps = self.timesteps
            
        # Handle class labels
        if y is None:
            # Cycle through classes like TeaPearce
            y = torch.arange(0, 10, device=device)
            y = y.repeat(int(batch_size / y.shape[0]) + 1)[:batch_size]
        
        # Start from pure noise
        x = torch.randn(batch_size, self.num_channels, 28, 28, device=device)
        
        # Don't drop context at test time
        context_mask = torch.zeros_like(y, dtype=torch.float, device=device)
        
        # Double the batch for classifier-free guidance
        y_doubled = y.repeat(2)
        context_mask_doubled = context_mask.repeat(2)
        context_mask_doubled[batch_size:] = 1.0  # Second half is context-free
        
        # Sampling loop
        for i in range(self.timesteps, 0, -1):
            # Prepare timestep
            t_normalized = torch.tensor([i / self.timesteps], device=device)
            t_normalized = t_normalized.repeat(batch_size, 1, 1, 1)
            
            # Double batch
            x_doubled = x.repeat(2, 1, 1, 1)
            t_doubled = t_normalized.repeat(2, 1, 1, 1)
            
            # Sample noise for next step
            z = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            
            # Predict noise with both conditional and unconditional
            with torch.no_grad():
                eps = self.nn_model(x_doubled, y_doubled, t_doubled, context_mask_doubled)
                eps_conditional = eps[:batch_size]
                eps_unconditional = eps[batch_size:]
                
                # Apply classifier-free guidance
                eps = (1 + guide_w) * eps_conditional - guide_w * eps_unconditional
            
            # Denoise step
            x = (
                self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        # FIXED: Only clamp to ensure valid range without shifting
        # The model's sampling process should already produce values in [0,1] range
        # if trained on [0,1] data. The previous (x + 1.0) / 2.0 transformation 
        # was causing data range mismatch issues with projections.
        x = torch.clamp(x, 0.0, 1.0)  # Ensure valid range
        return x 