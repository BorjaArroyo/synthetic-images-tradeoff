"""Privacy estimation pipeline for the tradeoff evaluation framework."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from abc import ABC, abstractmethod

from ..core.interfaces import PrivacyEstimator, DimensionalityReducer, Visualizer
from ..data.processor import ImageProcessor


class KDEPrivacyEstimator(PrivacyEstimator):
    """Privacy estimation using Kernel Density Estimation."""
    
    def __init__(self, bandwidth: float = 1.0):
        """Initialize the KDE estimator.
        
        Args:
            bandwidth: Bandwidth parameter for KDE
        """
        self.bandwidth = bandwidth
    
    def estimate(
        self,
        query_data: torch.Tensor,
        numerator_data: torch.Tensor,
        denominator_data: torch.Tensor
    ) -> float:
        """Estimate privacy using KDE.
        
        Args:
            query_data: Query sample
            numerator_data: Data for numerator
            denominator_data: Data for denominator
            
        Returns:
            Estimated privacy budget (epsilon)
        """
        # Convert tensors to numpy arrays
        query_np = query_data.cpu().numpy()
        numerator_np = numerator_data.cpu().numpy()
        denominator_np = denominator_data.cpu().numpy()
        
        # Fit KDE models
        kde = KernelDensity(bandwidth=self.bandwidth, algorithm="ball_tree").fit(numerator_np)
        log_likelihood_numerator = kde.score_samples(query_np)

        kde = kde.fit(denominator_np)
        log_likelihood_denominator = kde.score_samples(query_np)
        
        # Compute epsilon as log-ratio
        epsilon = log_likelihood_numerator - log_likelihood_denominator
        
        return float(epsilon)


class ClassifierPrivacyEstimator(PrivacyEstimator):
    """Privacy estimation using a classifier-based approach."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6, C: float = 0.8):
        """Initialize the classifier-based estimator.
        
        Args:
            max_iter: Maximum iterations for logistic regression
            tol: Tolerance for convergence
            C: Inverse of regularization strength
        """
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
    
    def estimate(
        self,
        query_data: torch.Tensor,
        numerator_data: torch.Tensor,
        denominator_data: torch.Tensor
    ) -> float:
        """Estimate privacy using a classifier.
        
        Args:
            query_data: Query sample
            numerator_data: Data for numerator
            denominator_data: Data for denominator
            
        Returns:
            Estimated privacy budget (epsilon)
        """
        # Convert tensors to numpy arrays
        query_np = query_data.cpu().numpy()
        numerator_np = numerator_data.cpu().numpy()
        denominator_np = denominator_data.cpu().numpy()
        
        # Prepare training data
        X = np.concatenate([numerator_np, denominator_np], axis=0)
        y = np.concatenate([
            np.ones(len(numerator_np), dtype=int),
            np.zeros(len(denominator_np), dtype=int)
        ])
        
        # Train classifier
        clf = LogisticRegression(
            max_iter=self.max_iter,
            tol=self.tol,
            C=self.C
        )
        clf.fit(X, y)
        
        # Predict logit for query
        logit_query = clf.decision_function(query_np)
        
        return float(logit_query)


class KNNPrivacyEstimator(PrivacyEstimator):
    """K-Nearest Neighbors based privacy estimator."""
    
    def __init__(
        self,
        k: int = 5,
        metric: str = 'euclidean',
        device: Optional[torch.device] = None
    ):
        """Initialize KNN privacy estimator.
        
        Args:
            k: Number of nearest neighbors
            metric: Distance metric to use
            device: Device to use for computation
        """
        self.k = k
        self.metric = metric
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.knn = NearestNeighbors(n_neighbors=k, metric=metric)
    
    def estimate(
        self,
        query_data: torch.Tensor,
        numerator_data: torch.Tensor,
        denominator_data: torch.Tensor
    ) -> float:
        """Estimate privacy using K-Nearest Neighbors.
        
        Args:
            query_data: Query data samples
            numerator_data: Synthetic data samples
            denominator_data: Real data samples
            
        Returns:
            Estimated privacy budget (epsilon)
        """
        # Convert tensors to numpy arrays
        query_np = query_data.cpu().numpy()
        synthetic_np = numerator_data.cpu().numpy()
        real_np = denominator_data.cpu().numpy()
        
        # Fit KNN on synthetic data
        self.knn.fit(synthetic_np)
        
        # Find k-th nearest neighbors for query points
        distances, _ = self.knn.kneighbors(query_np)
        synthetic_distances = distances[:, -1]  # k-th nearest neighbor distances
        
        # Find k-th nearest neighbors in real data
        self.knn.fit(real_np)
        distances, _ = self.knn.kneighbors(query_np)
        real_distances = distances[:, -1]  # k-th nearest neighbor distances
        
        # Compute privacy budget (epsilon)
        # For overfitting/privacy leakage: query is closer to synthetic than real
        # So real_distances > synthetic_distances, giving positive epsilon
        epsilon = np.log(real_distances / synthetic_distances).mean()
        
        return float(epsilon)


class NeuralNetworkPrivacyEstimator(PrivacyEstimator):
    """Privacy estimation using neural networks for density ratio estimation.
    
    This extends the classifier approach to handle non-linear relationships.
    The key is to extract the logit (pre-activation) values from the final layer,
    which approximates the log density ratio log(P(x)/Q(x)).
    
    For complex, high-dimensional data where the density ratio has non-linear
    structure, neural networks can capture these relationships better than
    linear logistic regression.
    """
    
    def __init__(self, 
                 hidden_layers: List[int] = [128, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 100,
                 early_stopping_patience: int = 10,
                 device: Optional[torch.device] = None,
                 random_state: int = 42):
        """Initialize the neural network-based estimator.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'leaky_relu')
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use for computation
            random_state: Random seed for reproducibility
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.model = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def _create_model(self, input_dim: int) -> torch.nn.Module:
        """Create the neural network model for density ratio estimation."""
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            
            # Add activation
            if self.activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(torch.nn.Tanh())
            elif self.activation == 'leaky_relu':
                layers.append(torch.nn.LeakyReLU(0.01))
            else:
                raise ValueError(f"Unknown activation: {self.activation}")
            
            # Add dropout for regularization
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final layer (no activation - we want raw logits)
        layers.append(torch.nn.Linear(prev_dim, 1))
        
        return torch.nn.Sequential(*layers)
    
    def estimate(
        self,
        query_data: torch.Tensor,
        numerator_data: torch.Tensor,
        denominator_data: torch.Tensor
    ) -> float:
        """Estimate privacy using neural network-based density ratio estimation.
        
        Theory: Train a neural network to distinguish between numerator and denominator
        distributions. The logit output approximates log(P_numerator(x) / P_denominator(x)),
        which is exactly the privacy budget ε.
        
        Args:
            query_data: Query sample
            numerator_data: Data for numerator (typically synthetic data from victim model)
            denominator_data: Data for denominator (typically synthetic data from ablated model)
            
        Returns:
            Estimated privacy budget (epsilon) as log-density-ratio
        """
        # Convert tensors to numpy arrays and then back to tensors on correct device
        query_tensor = query_data.to(self.device)
        numerator_tensor = numerator_data.to(self.device)
        denominator_tensor = denominator_data.to(self.device)
        
        # Prepare training data for binary classification
        # Label 1: numerator data, Label 0: denominator data
        X = torch.cat([numerator_tensor, denominator_tensor], dim=0)
        y = torch.cat([
            torch.ones(len(numerator_tensor), device=self.device),
            torch.zeros(len(denominator_tensor), device=self.device)
        ])
        
        # Create model
        input_dim = X.shape[1]
        self.model = self._create_model(input_dim).to(self.device)
        
        print(f"🧠 Using Neural Network for density ratio estimation")
        print(f"   Architecture: {input_dim} → {' → '.join(map(str, self.hidden_layers))} → 1")
        print(f"   Activation: {self.activation}, Dropout: {self.dropout_rate}")
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training setup
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass - get raw logits
                logits = self.model(batch_X).squeeze()
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress occasionally
            if (epoch + 1) % 20 == 0:
                print(f"   📊 Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
        # Get predictions for query data
        self.model.eval()
        with torch.no_grad():
            # Get raw logits (these approximate log density ratio)
            query_logits = self.model(query_tensor).squeeze()
            
            # Also get probabilities for insight
            query_probs = torch.sigmoid(query_logits)
        
        # The logit output directly approximates log(P_numerator(x) / P_denominator(x))
        epsilon = float(query_logits.mean().cpu())
        
        print(f"   📊 Neural network estimates: P(victim)={query_probs.mean():.4f}")
        print(f"   🎯 Logit (log density ratio): {epsilon:.4f}")
        
        return epsilon 