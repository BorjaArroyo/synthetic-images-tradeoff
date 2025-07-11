"""Neural network-based dimensionality reducers optimized for image data (improved version)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

from ..core.interfaces import DimensionalityReducer
from .components import ImageEncoder, ImageDecoder, ImprovedAffineCouplingLayer, EarlyStopping


class AutoencoderReducer(DimensionalityReducer):
    """Improved autoencoder reducer using advanced components with better architecture."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 max_epochs: int = 200, patience: int = 15, min_delta: float = 1e-6,
                 learning_rate: float = 1e-3, dropout_rate: float = 0.25,
                 device: Optional[torch.device] = None):
        """Initialize improved autoencoder reducer.
        
        Args:
            input_shape: Shape of input data (C, H, W)
            latent_dim: Dimensionality of latent space
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for regularization
            device: Device for computation
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device or torch.device('cpu')
        
        # Extract number of channels
        self.num_channels = input_shape[0]
        
        # Create improved model using enhanced components
        self.encoder = ImageEncoder(
            in_channels=self.num_channels,
            hidden_dims=(32, 64, 128),
            output_dim=512,  # Larger intermediate representation
            dropout_rate=dropout_rate
        )
        
        self.latent_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = ImageDecoder(
            latent_dim=latent_dim,
            out_channels=self.num_channels,
            hidden_dims=(256, 128, 64),  # Better progression
            dropout_rate=dropout_rate
        )
        
        # Move to device
        self.encoder.to(self.device)
        self.latent_projection.to(self.device)
        self.decoder.to(self.device)
        
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit the improved autoencoder to data."""
        print(f"   Enhanced Autoencoder: Training with improved architecture")
        
        # Ensure data is on correct device
        data = data.to(self.device)
        n_samples = data.shape[0]
        
        # Split data for validation
        indices = torch.randperm(n_samples)
        split_idx = int(0.9 * n_samples)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        print(f"   Train: {len(train_indices)}, Val: {len(val_indices)} samples")
        
        # Create data loaders with better batch size
        batch_size = min(128, len(train_indices) // 8)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_data),
            batch_size=batch_size, shuffle=False
        )
        
        # Enhanced optimizer with better settings
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.latent_projection.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=self.learning_rate, weight_decay=1e-4, eps=1e-8)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8, verbose=False, min_lr=1e-6
        )
        
        # Enhanced early stopping
        early_stopping = EarlyStopping(
            patience=self.patience, 
            min_delta=self.min_delta, 
            verbose=True,
            monitor_improvement=True
        )
        
        # Training loop with improved stability
        for epoch in range(self.max_epochs):
            # Training phase
            self.encoder.train()
            self.latent_projection.train()
            self.decoder.train()
            
            train_loss = 0.0
            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.encoder(batch_data)
                latent = self.latent_projection(features)
                reconstructed = self.decoder(latent)
                
                # Enhanced loss with perceptual component
                recon_loss = F.mse_loss(reconstructed, batch_data)
                
                # Optional: Add small regularization on latent space
                latent_reg = 0.01 * torch.mean(latent**2)
                total_loss = recon_loss + latent_reg
                
                total_loss.backward()
                
                # Enhanced gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + 
                    list(self.latent_projection.parameters()) + 
                    list(self.decoder.parameters()), 
                    max_norm=2.0
                )
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation phase
            self.encoder.eval()
            self.latent_projection.eval()
            self.decoder.eval()
            
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, in val_loader:
                    batch_data = batch_data.to(self.device)
                    
                    features = self.encoder(batch_data)
                    latent = self.latent_projection(features)
                    reconstructed = self.decoder(latent)
                    
                    recon_loss = F.mse_loss(reconstructed, batch_data)
                    latent_reg = 0.01 * torch.mean(latent**2)
                    val_loss += (recon_loss + latent_reg).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1}/{self.max_epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(val_loss, self, epoch + 1):
                break
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data to latent space."""
        if not self.fitted:
            raise ValueError("Model must be fitted before transformation")
        
        self.encoder.eval()
        self.latent_projection.eval()
        
        data = data.to(self.device)
        
        with torch.no_grad():
            features = self.encoder(data)
            latent = self.latent_projection(features)
        
        return latent.cpu().numpy()
    
    def state_dict(self):
        """Get state dict for early stopping."""
        return {
            'encoder': self.encoder.state_dict(),
            'latent_projection': self.latent_projection.state_dict(),
            'decoder': self.decoder.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for early stopping."""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.latent_projection.load_state_dict(state_dict['latent_projection'])
        self.decoder.load_state_dict(state_dict['decoder'])


class BetaVAEReducer(DimensionalityReducer):
    """Improved β-VAE reducer using enhanced components with better training stability."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 beta: float = 4.0, max_epochs: int = 200, patience: int = 15,
                 learning_rate: float = 1e-3, dropout_rate: float = 0.25,
                 device: Optional[torch.device] = None):
        """Initialize improved β-VAE reducer."""
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device or torch.device('cpu')
        
        # Extract number of channels
        self.num_channels = input_shape[0]
        
        # Create improved model using enhanced components
        self.encoder = ImageEncoder(
            in_channels=self.num_channels,
            hidden_dims=(32, 64, 128),
            output_dim=512,
            dropout_rate=dropout_rate
        )
        
        # VAE heads with better architecture
        self.mu_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        self.logvar_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = ImageDecoder(
            latent_dim=latent_dim,
            out_channels=self.num_channels,
            hidden_dims=(256, 128, 64),
            dropout_rate=dropout_rate
        )
        
        # Move to device
        self.encoder.to(self.device)
        self.mu_head.to(self.device)
        self.logvar_head.to(self.device)
        self.decoder.to(self.device)
        
        self.fitted = False
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with improved stability."""
        # Clamp logvar for stability (VAE best practice)
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def fit(self, data: torch.Tensor):
        """Fit the improved β-VAE to data."""
        print(f"   Enhanced β-VAE (β={self.beta}): Training with improved architecture")
        
        # Ensure data is on correct device
        data = data.to(self.device)
        n_samples = data.shape[0]
        
        # Split data for validation
        indices = torch.randperm(n_samples)
        split_idx = int(0.9 * n_samples)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        print(f"   Train: {len(train_indices)}, Val: {len(val_indices)} samples")
        
        # Create data loaders
        batch_size = min(128, len(train_indices) // 8)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_data),
            batch_size=batch_size, shuffle=False
        )
        
        # Enhanced optimizer
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters()},
            {'params': self.mu_head.parameters()},
            {'params': self.logvar_head.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=self.learning_rate, weight_decay=1e-4, eps=1e-8)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8, verbose=False, min_lr=1e-6
        )
        
        early_stopping = EarlyStopping(
            patience=self.patience, 
            min_delta=1e-4,  # Slightly larger delta for VAE
            verbose=True,
            monitor_improvement=True
        )
        
        # β annealing for better training
        beta_schedule = lambda epoch: min(self.beta, epoch / 50.0)  # Gradual β increase
        
        # Training loop
        for epoch in range(self.max_epochs):
            current_beta = beta_schedule(epoch)
            
            # Training phase
            self.encoder.train()
            self.mu_head.train()
            self.logvar_head.train()
            self.decoder.train()
            
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            
            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.encoder(batch_data)
                mu = self.mu_head(features)
                logvar = self.logvar_head(features)
                z = self.reparameterize(mu, logvar)
                reconstructed = self.decoder(z)
                
                # Enhanced VAE loss
                recon_loss = F.mse_loss(reconstructed, batch_data, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # β-VAE loss with current β
                total_loss = recon_loss + current_beta * kl_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + 
                    list(self.mu_head.parameters()) + 
                    list(self.logvar_head.parameters()) + 
                    list(self.decoder.parameters()), 
                    max_norm=2.0
                )
                optimizer.step()
                
                train_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
            
            # Validation phase
            self.encoder.eval()
            self.mu_head.eval()
            self.logvar_head.eval()
            self.decoder.eval()
            
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, in val_loader:
                    batch_data = batch_data.to(self.device)
                    
                    features = self.encoder(batch_data)
                    mu = self.mu_head(features)
                    logvar = self.logvar_head(features)
                    z = self.reparameterize(mu, logvar)
                    reconstructed = self.decoder(z)
                    
                    recon_loss = F.mse_loss(reconstructed, batch_data, reduction='mean')
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    total_loss = recon_loss + current_beta * kl_loss
                    val_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_recon_loss /= len(train_loader)
            train_kl_loss /= len(train_loader)
            
            scheduler.step(val_loss)
            
            # Print progress with more details
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1}/{self.max_epochs}, Total: {train_loss:.3f}, "
                      f"Recon: {train_recon_loss:.3f}, KL: {train_kl_loss:.3f}, "
                      f"β: {current_beta:.2f}, LR: {current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(val_loss, self, epoch + 1):
                break
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data to latent space using mean."""
        if not self.fitted:
            raise ValueError("Model must be fitted before transformation")
        
        self.encoder.eval()
        self.mu_head.eval()
        
        data = data.to(self.device)
        
        with torch.no_grad():
            features = self.encoder(data)
            mu = self.mu_head(features)
        
        return mu.cpu().numpy()
    
    def state_dict(self):
        """Get state dict for early stopping."""
        return {
            'encoder': self.encoder.state_dict(),
            'mu_head': self.mu_head.state_dict(),
            'logvar_head': self.logvar_head.state_dict(),
            'decoder': self.decoder.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for early stopping."""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.mu_head.load_state_dict(state_dict['mu_head'])
        self.logvar_head.load_state_dict(state_dict['logvar_head'])
        self.decoder.load_state_dict(state_dict['decoder'])


class NormalizingFlowReducer(DimensionalityReducer):
    """Significantly improved normalizing flow reducer with enhanced stability and expressiveness."""
    
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int = 2, 
                 num_flows: int = 8, max_epochs: int = 300, patience: int = 25,
                 learning_rate: float = 1e-3, dropout_rate: float = 0.1,
                 device: Optional[torch.device] = None):
        """Initialize enhanced normalizing flow reducer."""
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_flows = num_flows
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device or torch.device('cpu')
        
        # Extract number of channels
        self.num_channels = input_shape[0]
        
        # Create improved model using enhanced components
        self.encoder = ImageEncoder(
            in_channels=self.num_channels,
            hidden_dims=(32, 64, 128),
            output_dim=latent_dim,  # Direct to latent space
            dropout_rate=dropout_rate
        )
        
        # Enhanced normalizing flow layers with better initialization
        self.flows = nn.ModuleList([
            ImprovedAffineCouplingLayer(
                latent_dim, 
                hidden_dim=256,  # Larger networks
                mask_type='even' if i % 2 == 0 else 'odd',
                dropout_rate=dropout_rate
            )
            for i in range(num_flows)
        ])
        
        # Move to device
        self.encoder.to(self.device)
        self.flows.to(self.device)
        
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit the enhanced normalizing flow to data."""
        print(f"   Enhanced Normalizing Flow: Training with {self.num_flows} improved coupling layers")
        
        # Ensure data is on correct device
        data = data.to(self.device)
        n_samples = data.shape[0]
        
        # Split data for validation
        indices = torch.randperm(n_samples)
        split_idx = int(0.9 * n_samples)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        print(f"   Train: {len(train_indices)}, Val: {len(val_indices)} samples")
        
        # Create data loaders
        batch_size = min(128, len(train_indices) // 8)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_data),
            batch_size=batch_size, shuffle=False
        )
        
        # Enhanced optimizer for flows
        optimizer = torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': self.learning_rate},
            {'params': self.flows.parameters(), 'lr': self.learning_rate * 0.5}  # Slower for flows
        ], weight_decay=1e-4, eps=1e-8)
        
        # Cosine annealing scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-7
        )
        
        early_stopping = EarlyStopping(
            patience=self.patience, 
            min_delta=1e-5,
            verbose=True,
            monitor_improvement=True
        )
        
        best_likelihood = -float('inf')
        
        # Training loop with enhanced stability
        for epoch in range(self.max_epochs):
            # Training phase
            self.encoder.train()
            self.flows.train()
            
            train_loss = 0.0
            train_likelihood = 0.0
            
            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through encoder
                z = self.encoder(batch_data)
                
                # Forward pass through flows
                log_det_sum = 0.0
                for flow in self.flows:
                    z, log_det = flow(z)
                    log_det_sum += log_det
                
                # Enhanced loss computation
                # Standard normal prior
                log_prior = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
                log_likelihood = log_prior + log_det_sum
                
                # Mean likelihood (not negative for monitoring)
                likelihood = torch.mean(log_likelihood)
                
                # Loss to minimize (negative log likelihood)
                loss = -likelihood
                
                loss.backward()
                
                # Enhanced gradient clipping for flows
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.flows.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                
                train_loss += loss.item()
                train_likelihood += likelihood.item()
            
            # Validation phase
            self.encoder.eval()
            self.flows.eval()
            
            val_loss = 0.0
            val_likelihood = 0.0
            with torch.no_grad():
                for batch_data, in val_loader:
                    batch_data = batch_data.to(self.device)
                    
                    z = self.encoder(batch_data)
                    
                    log_det_sum = 0.0
                    for flow in self.flows:
                        z, log_det = flow(z)
                        log_det_sum += log_det
                    
                    log_prior = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
                    log_likelihood = log_prior + log_det_sum
                    likelihood = torch.mean(log_likelihood)
                    
                    loss = -likelihood
                    val_loss += loss.item()
                    val_likelihood += likelihood.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_likelihood /= len(train_loader)
            val_likelihood /= len(val_loader)
            
            scheduler.step()
            
            # Track best likelihood
            if val_likelihood > best_likelihood:
                best_likelihood = val_likelihood
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1}/{self.max_epochs}, "
                      f"Loss: {train_loss:.4f}, Likelihood: {train_likelihood:.4f}, "
                      f"Val Likelihood: {val_likelihood:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping check (minimize val_loss)
            if early_stopping(val_loss, self, epoch + 1):
                break
        
        print(f"   Training completed. Best validation likelihood: {best_likelihood:.4f}")
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data to latent space."""
        if not self.fitted:
            raise ValueError("Model must be fitted before transformation")
        
        self.encoder.eval()
        
        data = data.to(self.device)
        
        with torch.no_grad():
            z = self.encoder(data)
        
        return z.cpu().numpy()
    
    def state_dict(self):
        """Get state dict for early stopping."""
        return {
            'encoder': self.encoder.state_dict(),
            'flows': self.flows.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for early stopping."""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.flows.load_state_dict(state_dict['flows']) 