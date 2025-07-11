"""CNN-optimized dimensionality reducers with early stopping for image data."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

from ..core.interfaces import DimensionalityReducer


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {name: param.clone() for name, param in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class CNNAutoencoder(nn.Module):
    """CNN-based autoencoder optimized for image data."""
    
    def __init__(self, num_channels: int = 1, latent_dim: int = 2):
        super(CNNAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 28x28 -> 2D latent (corrected architecture)
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 7x7  
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 7x7 -> 7x7 (same size, increase depth)
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Ensure exactly 7x7
            
            # Flatten and project to latent space
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder: latent_dim -> 28x28 (corrected architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer: maintain 28x28
            nn.ConvTranspose2d(32, num_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CNNBetaVAE(nn.Module):
    """CNN-based β-VAE optimized for image data."""
    
    def __init__(self, num_channels: int = 1, latent_dim: int = 2):
        super(CNNBetaVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 28x28 -> latent_dim (corrected architecture)
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 7x7 -> 7x7 (increase depth)
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),  # Ensure exactly 7x7
            
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True)
        )
        
        # VAE heads
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: latent_dim -> 28x28 (corrected architecture)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer: maintain 28x28
            nn.ConvTranspose2d(32, num_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class CNNNormalizingFlow(nn.Module):
    """CNN-based normalizing flow for image data."""
    
    def __init__(self, num_channels: int = 1, latent_dim: int = 2, num_flows: int = 4):
        super(CNNNormalizingFlow, self).__init__()
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        
        # CNN encoder to reduce dimensionality first
        self.cnn_encoder = nn.Sequential(
            # 28x28x1 -> 14x14x32
            nn.Conv2d(num_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14x14x32 -> 7x7x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True)
        )
        
        # Flow layers
        self.flow_layers = nn.ModuleList([
            CNNAffineCouplingLayer(256, hidden_dim=128)
            for _ in range(num_flows)
        ])
        
        # Final projection to latent space
        self.final_projection = nn.Linear(256, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode with CNN
        h = self.cnn_encoder(x)
        
        # Pass through flow layers
        z = h
        log_det_sum = 0
        
        for layer in self.flow_layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        
        # Final projection
        z_final = self.final_projection(z)
        
        return z_final, log_det_sum


class CNNAffineCouplingLayer(nn.Module):
    """Affine coupling layer for CNN normalizing flow."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.half_dim = input_dim // 2
        
        self.scale_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.half_dim),
            nn.Tanh()  # Stabilize training
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - self.half_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        
        log_det = torch.sum(s, dim=1)
        
        return y, log_det


class CNNAutoencoderReducer(DimensionalityReducer):
    """CNN-based autoencoder reducer with early stopping."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 max_epochs: int = 200, patience: int = 15, min_delta: float = 1e-6,
                 device: Optional[torch.device] = None):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create CNN autoencoder
        num_channels = input_shape[0] if len(input_shape) == 3 else 1
        self.model = CNNAutoencoder(num_channels=num_channels, latent_dim=latent_dim).to(self.device)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        self._fitted = False
        
    def fit(self, data: torch.Tensor):
        """Fit CNN autoencoder with early stopping."""
        data = data.to(self.device)
        
        # Split into train/validation
        indices = torch.randperm(len(data))
        split_idx = int(0.9 * len(data))
        train_data = data[indices[:split_idx]]
        val_data = data[indices[split_idx:]]
        
        print(f"   CNN Autoencoder: {len(train_data)} train, {len(val_data)} val samples")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training
            train_loss = 0
            for i in range(0, len(train_data), 32):  # Mini-batches
                batch = train_data[i:i+32]
                optimizer.zero_grad()
                
                encoded, decoded = self.model(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_data)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(val_data), 32):
                    batch = val_data[i:i+32]
                    encoded, decoded = self.model(batch)
                    loss = criterion(decoded, batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_data)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.max_epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            self.model.train()
        
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted CNN autoencoder."""
        if not self._fitted:
            raise RuntimeError("CNN Autoencoder must be fitted before transform")
        
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            results = []
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                encoded, _ = self.model(batch)
                results.append(encoded.cpu())
            
            return torch.cat(results, dim=0).numpy()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to latent space."""
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            encoded, _ = self.model(data)
            return encoded
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        latent = latent.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            decoded = self.model.decoder(latent)
            return decoded.view(-1, *self.input_shape)


class CNNBetaVAEReducer(DimensionalityReducer):
    """CNN-based β-VAE reducer with early stopping."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 beta: float = 4.0, max_epochs: int = 200, patience: int = 15,
                 device: Optional[torch.device] = None):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.max_epochs = max_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create CNN β-VAE
        num_channels = input_shape[0] if len(input_shape) == 3 else 1
        self.model = CNNBetaVAE(num_channels=num_channels, latent_dim=latent_dim).to(self.device)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
        self._fitted = False
        
    def fit(self, data: torch.Tensor):
        """Fit CNN β-VAE with early stopping."""
        data = data.to(self.device)
        
        # Split into train/validation
        indices = torch.randperm(len(data))
        split_idx = int(0.9 * len(data))
        train_data = data[indices[:split_idx]]
        val_data = data[indices[split_idx:]]
        
        print(f"   CNN β-VAE (β={self.beta}): {len(train_data)} train, {len(val_data)} val samples")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training
            train_loss = 0
            for i in range(0, len(train_data), 32):
                batch = train_data[i:i+32]
                optimizer.zero_grad()
                
                recon, mu, logvar, z = self.model(batch)
                
                # β-VAE loss
                recon_loss = F.mse_loss(recon, batch, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kld_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_data)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(val_data), 32):
                    batch = val_data[i:i+32]
                    recon, mu, logvar, z = self.model(batch)
                    recon_loss = F.mse_loss(recon, batch, reduction='sum')
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.beta * kld_loss
                    val_loss += loss.item()
            
            val_loss /= len(val_data)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.max_epochs}, Train: {train_loss:.1f}, Val: {val_loss:.1f}")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            self.model.train()
        
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted CNN β-VAE."""
        if not self._fitted:
            raise RuntimeError("CNN β-VAE must be fitted before transform")
        
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            results = []
            for i in range(0, len(data), 32):
                batch = data[i:i+32]
                mu, logvar = self.model.encode(batch)
                results.append(mu.cpu())  # Use mean for deterministic encoding
            
            return torch.cat(results, dim=0).numpy()
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode data to latent space."""
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            mu, logvar = self.model.encode(data)
            return mu  # Use mean for deterministic encoding
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        latent = latent.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            decoded = self.model.decode(latent)
            return decoded


class CNNNormalizingFlowReducer(DimensionalityReducer):
    """CNN-based normalizing flow reducer with early stopping."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 num_flows: int = 4, max_epochs: int = 300, patience: int = 20,
                 device: Optional[torch.device] = None):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create CNN normalizing flow
        num_channels = input_shape[0] if len(input_shape) == 3 else 1
        self.model = CNNNormalizingFlow(
            num_channels=num_channels, 
            latent_dim=latent_dim, 
            num_flows=num_flows
        ).to(self.device)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
        self._fitted = False
        
    def fit(self, data: torch.Tensor):
        """Fit CNN normalizing flow with early stopping."""
        data = data.to(self.device)
        
        # Split into train/validation
        indices = torch.randperm(len(data))
        split_idx = int(0.9 * len(data))
        train_data = data[indices[:split_idx]]
        val_data = data[indices[split_idx:]]
        
        print(f"   CNN Normalizing Flow: {len(train_data)} train, {len(val_data)} val samples")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
        
        self.model.train()
        for epoch in range(self.max_epochs):
            # Training
            train_loss = 0
            for i in range(0, len(train_data), 16):  # Smaller batches for flow
                batch = train_data[i:i+16]
                optimizer.zero_grad()
                
                z_final, log_det_sum = self.model(batch)
                
                # Negative log-likelihood loss
                loss = -torch.mean(log_det_sum) + 0.5 * torch.mean(z_final**2)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_data)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(val_data), 16):
                    batch = val_data[i:i+16]
                    z_final, log_det_sum = self.model(batch)
                    loss = -torch.mean(log_det_sum) + 0.5 * torch.mean(z_final**2)
                    val_loss += loss.item()
            
            val_loss /= len(val_data)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{self.max_epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            self.model.train()
        
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted CNN normalizing flow."""
        if not self._fitted:
            raise RuntimeError("CNN Normalizing Flow must be fitted before transform")
        
        data = data.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            results = []
            for i in range(0, len(data), 16):
                batch = data[i:i+16]
                z_final, _ = self.model(batch)
                results.append(z_final.cpu())
            
            return torch.cat(results, dim=0).numpy() 