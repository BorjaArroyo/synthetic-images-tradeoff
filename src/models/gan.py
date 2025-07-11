"""Conditional GAN implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Conditional Generator for GAN matching vae-flow-dp logic."""
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, img_shape: tuple = (1, 28, 28)):
        """Initialize Generator.
        
        Args:
            latent_dim: Dimensionality of latent space
            num_classes: Number of classes for conditioning
            img_shape: Shape of output images (channels, height, width)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.init_size = img_shape[1] // 4  # For 28x28 -> 7x7
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.GroupNorm(32, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with one-hot label concatenation.
        
        Args:
            noise: Latent noise vector (z)
            labels: Class labels
            
        Returns:
            Generated images
        """
        label_onehot = F.one_hot(labels, num_classes=self.num_classes).float().reshape(noise.size(0), -1)
        # Concatenate along feature dimension
        gen_input = torch.cat([noise, label_onehot], dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """Conditional Discriminator for GAN matching vae-flow-dp logic."""
    
    def __init__(self, num_classes: int = 10, img_shape: tuple = (1, 28, 28)):
        """Initialize Discriminator.
        
        Args:
            num_classes: Number of classes for conditioning
            img_shape: Shape of input images (channels, height, width)
        """
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + num_classes, 64, 3, 2, 1),
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
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
    
    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatial label concatenation.
        
        Args:
            img: Input images
            labels: Class labels
            
        Returns:
            Discriminator predictions (sigmoid activated)
        """
        label_onehot = F.one_hot(labels, num_classes=self.num_classes).float().reshape(img.size(0), -1)
        label_maps = label_onehot.unsqueeze(-1).unsqueeze(-1)
        label_maps = label_maps.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat((img, label_maps), dim=1)
        validity = self.model(x)
        return validity.view(-1, 1).sigmoid() 