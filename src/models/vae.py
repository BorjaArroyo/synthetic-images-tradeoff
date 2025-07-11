"""Variational Autoencoder implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Convolutional encoder for VAE."""
    
    def __init__(self, in_channels: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, 32, 4, 2, 1),      # 28x28 -> 14x14
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1),     # 14x14 -> 7x7
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 1, 1),    # 7x7 -> 7x7
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 2 * latent_dim)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with spatial label concatenation.
        
        Args:
            x: Input images
            y: Class labels
            
        Returns:
            Tuple of (mean, logvar)
        """
        # Convert y to one-hot and add spatial dimensions
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()  # [B, num_classes]
        y_onehot = y_onehot.view(-1, self.num_classes, 1, 1)  # [B, num_classes, 1, 1]
        y_onehot = y_onehot.expand(-1, -1, x.size(2), x.size(3))  # [B, num_classes, H, W]
        x = torch.cat([x, y_onehot], dim=1)
        output = self.net(x)
        # Split into mean and logvar
        mean, logvar = output.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -4, 4)
        return mean, logvar


class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE."""
    
    def __init__(self, n_channels: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.GELU(),
            nn.Linear(256, 128 * 7 * 7),
            nn.GELU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.Conv2d(128, 256, 3, 1, 1), # 7x7 -> 7x7
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7x7 -> 14x14
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14x14 -> 28x28
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, n_channels, 3, 1, 1),           # 28x28 -> 28x28
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with label concatenation.
        
        Args:
            z: Latent vector
            y: Class labels
            
        Returns:
            Reconstructed image
        """
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float().reshape(z.shape[0], -1)  # [B, num_classes]
        z = torch.cat([z, y_onehot], dim=1)
        return self.net(z)


class VAE(nn.Module):
    """Conditional Variational Autoencoder matching vae-flow-dp logic."""
    
    def __init__(self, num_channels: int = 1, latent_dim: int = 128, num_classes: int = 10):
        """Initialize VAE.
        
        Args:
            num_channels: Number of input channels
            latent_dim: Dimensionality of latent space
            num_classes: Number of classes for conditioning
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.encoder = ConvEncoder(num_channels, latent_dim, num_classes)
        self.decoder = ConvDecoder(num_channels, latent_dim, num_classes)
        
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.
        
        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass matching original logic.
        
        Args:
            x: Input images
            y: Class labels
            
        Returns:
            Tuple of (reconstructed_x, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encoder(x, y)
        z = self.reparameterize(z_mean, z_logvar)
        if self.training:
            z = F.dropout(z, p=0.1)
        x_recon = self.decoder(z, y)
        return x_recon, z_mean, z_logvar
    
    def encode(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters.
        
        Args:
            x: Input images
            y: Class labels
            
        Returns:
            Tuple of (mean, logvar)
        """
        return self.encoder(x, y)
    
    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image.
        
        Args:
            z: Latent vector
            y: Class labels
            
        Returns:
            Reconstructed image
        """
        return self.decoder(z, y)
    
    def reconstruct(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct input images using mean of latent distribution.
        
        Args:
            x: Input images
            y: Class labels
            
        Returns:
            Reconstructed images
        """
        z_mean, z_logvar = self.encoder(x, y)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z, y)
        return x_recon
    
    def sample(self, num_samples: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sample new images.
        
        Args:
            num_samples: Number of samples to generate
            y: Class labels
            device: Device to generate on
            
        Returns:
            Generated images
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, y) 