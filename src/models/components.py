"""Reusable CNN components for dimensionality reduction models (improved version)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import copy


class ImageEncoder(nn.Module):
    """Advanced convolutional encoder optimized for 28x28 images with best practices."""
    
    def __init__(self, in_channels: int, hidden_dims: Tuple[int, ...] = (32, 64, 128), 
                 output_dim: int = 256, dropout_rate: float = 0.25):
        """Initialize the encoder.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            hidden_dims: Hidden dimensions for conv layers
            output_dim: Final output dimension before latent projection
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        layers = []
        prev_dim = in_channels
        
        # Build convolutional layers with improved architecture
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # First layer: 28x28 -> 14x14
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU()
                ])
            elif i == 1:
                # Second layer: 14x14 -> 7x7
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout2d(dropout_rate)  # Add dropout for regularization
                ])
            else:
                # Additional layers: maintain 7x7, increase depth
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout2d(dropout_rate)
                ])
            prev_dim = hidden_dim
        
        # Final processing with better architecture
        layers.extend([
            nn.AdaptiveAvgPool2d((7, 7)),  # Ensure exactly 7x7
            nn.Flatten(),
            nn.Linear(prev_dim * 7 * 7, 512),  # Intermediate layer
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
            nn.GELU()
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Encoded features [B, output_dim]
        """
        return self.net(x)


class ImageDecoder(nn.Module):
    """Advanced convolutional decoder optimized for 28x28 images with GAN-inspired improvements."""
    
    def __init__(self, latent_dim: int, out_channels: int = 1, 
                 hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout_rate: float = 0.25):
        """Initialize the decoder.
        
        Args:
            latent_dim: Dimensionality of latent space
            out_channels: Number of output channels (1 for grayscale, 3 for RGB)
            hidden_dims: Hidden dimensions for deconv layers (reverse order)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        
        # Project from latent to feature map with better architecture
        first_hidden = hidden_dims[0]
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, first_hidden * 7 * 7),
            nn.GELU(),
            nn.Unflatten(1, (first_hidden, 7, 7))
        )
        
        # Build deconvolutional layers with GAN-inspired improvements
        layers = []
        prev_dim = first_hidden
        
        # Add intermediate conv layer for feature refinement (VAE-inspired)
        layers.extend([
            nn.Conv2d(prev_dim, prev_dim, 3, stride=1, padding=1),
            nn.GroupNorm(min(8, prev_dim), prev_dim),
            nn.GELU()
        ])
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            if i == 1:
                # First upsampling: 7x7 -> 14x14 (GAN-inspired)
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),  # Reduce checkerboard artifacts
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU for better gradients
                    nn.Dropout2d(dropout_rate)
                ])
            elif i == 2:
                # Second upsampling: 14x14 -> 28x28
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout_rate)
                ])
            else:
                # Additional layers: maintain size, refine features
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout_rate)
                ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.extend([
            nn.Conv2d(prev_dim, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Latent tensor [B, latent_dim]
            
        Returns:
            Reconstructed image [B, out_channels, 28, 28]
        """
        x = self.latent_projection(z)
        return self.net(x)


class ConditionalImageEncoder(nn.Module):
    """Conditional encoder that can handle class labels (VAE-inspired)."""
    
    def __init__(self, in_channels: int, num_classes: int, latent_dim: int, 
                 hidden_dims: Tuple[int, ...] = (32, 64, 128), dropout_rate: float = 0.25):
        """Initialize conditional encoder.
        
        Args:
            in_channels: Number of input image channels
            num_classes: Number of class labels
            latent_dim: Output latent dimension (for VAE: 2*latent_dim for mu,logvar)
            hidden_dims: Hidden conv dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Build network with label conditioning
        layers = []
        prev_dim = in_channels + num_classes  # Concatenate labels spatially
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU()
                ])
            elif i == 1:
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout2d(dropout_rate)
                ])
            else:
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.GELU(),
                    nn.Dropout2d(dropout_rate)
                ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(prev_dim * 7 * 7, latent_dim)  # Note: for VAE, this should be 2*latent_dim
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with label conditioning.
        
        Args:
            x: Input images [B, C, H, W]
            y: Class labels [B]
            
        Returns:
            Encoded features [B, latent_dim]
        """
        # Spatial label concatenation (VAE approach)
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        y_onehot = y_onehot.view(-1, self.num_classes, 1, 1)
        y_onehot = y_onehot.expand(-1, -1, x.size(2), x.size(3))
        
        x_conditioned = torch.cat([x, y_onehot], dim=1)
        return self.net(x_conditioned)


class ConditionalImageDecoder(nn.Module):
    """Conditional decoder that can handle class labels (VAE-inspired)."""
    
    def __init__(self, latent_dim: int, num_classes: int, out_channels: int = 1,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout_rate: float = 0.25):
        """Initialize conditional decoder.
        
        Args:
            latent_dim: Latent space dimensionality
            num_classes: Number of class labels
            out_channels: Output image channels
            hidden_dims: Hidden dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.out_channels = out_channels
        
        # Project latent + labels to feature map
        first_hidden = hidden_dims[0]
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, first_hidden * 7 * 7),
            nn.GELU(),
            nn.Unflatten(1, (first_hidden, 7, 7))
        )
        
        # Build decoder layers
        layers = []
        prev_dim = first_hidden
        
        # Intermediate refinement layer
        layers.extend([
            nn.Conv2d(prev_dim, prev_dim, 3, stride=1, padding=1),
            nn.GroupNorm(min(8, prev_dim), prev_dim),
            nn.GELU()
        ])
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            if i == 1:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout_rate)
                ])
            elif i == 2:
                layers.extend([
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            else:
                layers.extend([
                    nn.Conv2d(prev_dim, hidden_dim, 3, stride=1, padding=1),
                    nn.GroupNorm(min(8, hidden_dim), hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Conv2d(prev_dim, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass with label conditioning.
        
        Args:
            z: Latent vector [B, latent_dim]
            y: Class labels [B]
            
        Returns:
            Reconstructed image [B, out_channels, 28, 28]
        """
        # Concatenate labels with latent vector
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        z_conditioned = torch.cat([z, y_onehot], dim=1)
        
        x = self.latent_projection(z_conditioned)
        return self.net(x)


class ImprovedAffineCouplingLayer(nn.Module):
    """Improved affine coupling layer with better stability and expressiveness."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, mask_type: str = 'even',
                 dropout_rate: float = 0.1):
        """Initialize improved coupling layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for transformation network
            mask_type: 'even' or 'odd' for alternating masks
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.mask_type = mask_type
        
        # Create binary mask
        mask = torch.arange(input_dim) % 2
        if mask_type == 'odd':
            mask = 1 - mask
        self.register_buffer('mask', mask.float())
        
        # Improved transformation networks with dropout
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # GELU for smoother gradients
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training stability."""
        for module in [self.scale_net, self.translate_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with improved numerical stability."""
        masked_x = x * self.mask
        scale = self.scale_net(masked_x) * (1 - self.mask)
        translate = self.translate_net(masked_x) * (1 - self.mask)
        
        # Apply transformation with clamping for stability
        scale = torch.clamp(scale, -3, 3)  # Prevent extreme scaling
        y = masked_x + (1 - self.mask) * (x * torch.exp(scale) + translate)
        
        # Log determinant
        log_det = torch.sum(scale * (1 - self.mask), dim=1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation with improved stability."""
        masked_y = y * self.mask
        scale = self.scale_net(masked_y) * (1 - self.mask)
        translate = self.translate_net(masked_y) * (1 - self.mask)
        
        # Clamp for stability
        scale = torch.clamp(scale, -3, 3)
        x = masked_y + (1 - self.mask) * (y - translate) * torch.exp(-scale)
        
        return x


class EarlyStopping:
    """Enhanced early stopping utility with best weights restoration and learning rate scheduling."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True, verbose: bool = False,
                 monitor_improvement: bool = True):
        """Initialize improved early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            verbose: Whether to print stopping info
            monitor_improvement: Whether to track improvement trends
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.monitor_improvement = monitor_improvement
        
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.best_epoch = 0
        self.loss_history = []
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Check if training should stop with enhanced monitoring.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        self.loss_history.append(val_loss)
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"   New best validation loss: {val_loss:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            
        # Check for improvement trend if enabled
        if self.monitor_improvement and len(self.loss_history) >= 5:
            recent_trend = self._check_improvement_trend()
            if not recent_trend and self.counter >= self.patience // 2:
                if self.verbose:
                    print(f"   Loss plateaued - considering early stop")
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f"   Early stopping at epoch {epoch} (best was epoch {self.best_epoch})")
                print(f"   Best validation loss: {self.best_loss:.6f}")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def _check_improvement_trend(self) -> bool:
        """Check if there's been recent improvement."""
        if len(self.loss_history) < 5:
            return True
        
        recent_losses = self.loss_history[-5:]
        # Check if there's a decreasing trend
        improvements = sum(1 for i in range(1, len(recent_losses)) 
                          if recent_losses[i] < recent_losses[i-1])
        return improvements >= 1  # At least some improvement


# Backward compatibility aliases
AffineCouplingLayer = ImprovedAffineCouplingLayer 