#!/usr/bin/env python3
"""
Complete Experimental Runner - Tradeoff with VAE-Flow-DP Logic

This script runs full experiments across different DPSGD sigma values,
replicating the functionality of the original main2.py from vae-flow-dp.

Usage:
    python run_experiments.py --config config/experiment_config.yaml
    python run_experiments.py --arch cvae --dataset mnist --sigmas 0.0,0.1,0.2,0.3
    python run_experiments.py --arch cgan --dataset pathmnist --quick_test
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import time
from typing import Dict, List, Optional, Any
from functools import partial

# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

import torch
import torch.utils.data
import numpy as np
import mlflow
import mlflow.pytorch

# Import visualization module - automatically configures fonts
import src.visualization as viz

# Import framework components - direct imports to avoid circular dependencies
from src.models.vae import VAE
from src.models.gan import Generator, Discriminator
from src.models.diffusion import DiffusionModel
from src.models.sampler import sample, generate_data
from src.core.trainer import train_vae, train_gan, train_diffusion
from src.models.scheduler import GapAwareDStepScheduler
from src.data.datasets import get_dataset_config, get_subset, filter_query
from src.utility.metrics import ClassificationUtilityMetric
from src.metrics.fidelity import compute_fidelity_metrics
from src.privacy.estimator import KDEPrivacyEstimator, ClassifierPrivacyEstimator, KNNPrivacyEstimator, NeuralNetworkPrivacyEstimator
from src.privacy.pipeline import PrivacyPipeline, PrivacyConfig
from src.privacy.analysis import ecdf_distance_curves, compute_privacy_distance_metrics
# create_privacy_visualizer removed - use functions from src.visualization.privacy directly
from src.privacy.reducer_manager import get_reducer_manager
from src.core.interfaces import DimensionalityReducer
import math


class ExperimentRunner:
    """Complete experimental runner for DP generative models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment runner with configuration."""
        import os
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup MLflow
        mlflow_uri = config.get('mlflow_uri', f'file://{os.getcwd()}/mlruns')
        mlflow.set_tracking_uri(mlflow_uri)
        self.experiment_name = config['experiment_name']
        mlflow.set_experiment(self.experiment_name)
        
        # Dataset configuration
        self.dataset_name = config['dataset']
        self.dataset_config = get_dataset_config(self.dataset_name)
        
        # Privacy evaluation configuration
        self.reducer_type = config.get('privacy_reducer', 'umap')
        self.density_method = config.get('privacy_density_method', 'kde')
        self.enable_plotting = config.get('enable_plotting', True)
        
        # Initialize reducer manager once per experiment session
        # This ensures the same reducer is reused across all seeds
        query_class = 3 if self.dataset_name == 'organamnist' else 0
        self.reducer_manager = get_reducer_manager(default_query_class=query_class)
        
        print(f"🎯 Experiment Runner Initialized")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Device: {self.device}")
        print(f"   Privacy Method: {self.reducer_type} + {self.density_method}")
        print(f"   Reducer Manager: Shared across all seeds")
        
    def _create_dataloader(self, n_per_class: int, exclude_class: Optional[int] = None, 
                          add_query: bool = False, batch_size: Optional[int] = None, train: bool = True):
        """Create dataloader with specified parameters."""
        # Use config batch size if not provided
        if batch_size is None:
            batch_size = self.config.get('batch_size', 64)
            
        subset_fn = partial(
            get_subset, 
            n_per_class=n_per_class, 
            exclude_class=exclude_class,
            add_query_element=add_query,
            n_classes=self.dataset_config['num_classes']
        )
        
        loader, dataset = self.dataset_config['load_fn'](
            batch_size=batch_size,
            num_workers=2,
            subset_fn=subset_fn,
            train=train
        )
        
        return loader, dataset

    def _get_reducer(self) -> DimensionalityReducer:
        """Get dimensionality reducer based on YAML configuration."""
        if self.reducer_type == 'none':
            return None
        
        # Get reducer parameters from config
        reducer_config = self.config.get('reducer_config', {})
        reducer_params = reducer_config.get(self.reducer_type, {})
        
        # Set default parameters if not found in config
        default_params = {
            'umap': {
                'n_components': 2, 
                'n_neighbors': 15, 
                'min_dist': 0.1,
                'random_state': 42
            },
            'tsne': {
                'n_components': 2, 
                'perplexity': 30.0,
                'random_state': 42
            },
            'isomap': {
                'n_components': 2,
                'n_neighbors': 15,
                'eigen_solver': 'auto',
                'path_method': 'auto'
            },
            'normalizing_flow': {
                'latent_dim': 2,
                'patience': 50,
                'max_epochs': 1000,
                'learning_rate': 1e-4,
                'num_flows': 4
            },
            'autoencoder': {
                'latent_dim': 2,
                'patience': 50,
                'max_epochs': 1000,
                'learning_rate': 1e-4,
            },
            'beta_vae': {
                'latent_dim': 2,
                'beta': 4.0,
                'patience': 50,
                'max_epochs': 1000,
                'learning_rate': 1e-4,
            }
        }
        
        # Use config params or fallback to defaults
        if not reducer_params and self.reducer_type in default_params:
            reducer_params = default_params[self.reducer_type]
            print(f"⚠️  Using default parameters for {self.reducer_type}")
        elif not reducer_params:
            raise ValueError(f"Unknown reducer type: {self.reducer_type}")
        
        # Print informative messages
        if self.reducer_type == 'isomap':
            print("🌐 Using ISOMAP - Geodesic distance preserving manifold learning")
        elif self.reducer_type == 'normalizing_flow':
            print("🌊 Using Normalizing Flow - Optimal for distribution coherence")
        elif self.reducer_type == 'autoencoder':
            print("🧠 Using Autoencoder - Good balance for domain-specific learning")
        elif self.reducer_type == 'beta_vae':
            print("🎯 Using β-VAE - Disentangled representations")
        
        print(f"🔄 Getting reducer: {self.reducer_type} for {self.dataset_name} with params: {reducer_params}")
        
        # Get or create the dataset-level reducer using the shared instance
        reducer = self.reducer_manager.get_or_create_reducer(
            dataset_name=self.dataset_name,
            reducer_type=self.reducer_type,
            reducer_params=reducer_params
        )
        
        return reducer

    def _get_density_estimator(self):
        """Get density estimator based on YAML configuration."""
        # Get density estimator parameters from config
        density_config = self.config.get('density_config', {})
        estimator_params = density_config.get(self.density_method, {})
        
        # Set default parameters if not found in config
        default_params = {
            'kde': {'bandwidth': 1.5},
            'classifier': {
                'max_iter': 1000,
                'tol': 1e-6,
                'C': 1.0,
                'solver': 'lbfgs',
                'random_state': 42
            },
            'neural': {
                'hidden_layers': [128, 64],
                'activation': 'relu',
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 100,
                'early_stopping_patience': 10,
                'random_state': 42
            },
            'knn': {
                'k': 10,
                'metric': 'euclidean'
            }
        }
        
        # Use config params or fallback to defaults
        if not estimator_params and self.density_method in default_params:
            estimator_params = default_params[self.density_method]
            print(f"⚠️  Using default parameters for {self.density_method}")
        elif not estimator_params:
            raise ValueError(f"Unknown density method: {self.density_method}")
        
        print(f"🔄 Creating {self.density_method} estimator with params: {estimator_params}")
        
        if self.density_method == 'kde':
            return KDEPrivacyEstimator(**estimator_params)
        elif self.density_method == 'classifier':
            return ClassifierPrivacyEstimator(**estimator_params)
        elif self.density_method == 'neural':
            # Add device to neural estimator params
            estimator_params['device'] = self.device
            return NeuralNetworkPrivacyEstimator(**estimator_params)
        elif self.density_method == 'knn':
            # Add device to KNN estimator params
            estimator_params['device'] = self.device
            return KNNPrivacyEstimator(**estimator_params)
        else:
            raise ValueError(f"Unknown density method: {self.density_method}. "
                           f"Available options: 'kde', 'classifier', 'neural', 'knn'")

    def evaluate_privacy(self, ablated_model: torch.nn.Module, victim_model: torch.nn.Module, 
                        full_model: torch.nn.Module, arch: str, run_id: str) -> Dict[str, float]:
        """Evaluate privacy metrics using proper density ratio estimation."""
        print(f"🔒 Evaluating privacy using {self.reducer_type} + {self.density_method}...")
        
        # Determine query class based on dataset - use class 3 for organamnist for worse privacy guarantees
        query_class = 3 if self.dataset_name == 'organamnist' else 0
        print(f"   Using query class {query_class} for {self.dataset_name} dataset")
        
        # Generate synthetic data from both models
        print(f"📊 Generating synthetic data...")
        privacy_samples_per_class = self.config.get('samples_per_class', 1000)  # Use config value
        print(f"   Using {privacy_samples_per_class} samples per class for privacy evaluation")
        
        synthetic_ablated = generate_data(
            model=ablated_model,
            n_classes=self.dataset_config['num_classes'],
            samples_per_class=privacy_samples_per_class,
            device=self.device,
            exclude_class=None  # Include ALL classes, including class 0
        )
        print(f"   Ablated data shape: {synthetic_ablated.shape}")
        
        synthetic_victim = generate_data(
            model=victim_model,
            n_classes=self.dataset_config['num_classes'],
            samples_per_class=privacy_samples_per_class,
            device=self.device,
            exclude_class=None  # Include ALL classes, including class 0
        )
        print(f"   Victim data shape: {synthetic_victim.shape}")
        
        # Generate synthetic data from full model for ECDF analysis
        synthetic_full = generate_data(
            model=full_model,
            n_classes=self.dataset_config['num_classes'],
            samples_per_class=privacy_samples_per_class,
            device=self.device,
            exclude_class=None  # Include ALL classes
        )
        print(f"   Full model data shape: {synthetic_full.shape}")
        
        # Get query sample using the same approach as vae-flow-dp
        # This ensures we get the SAME query sample that was used during victim training
        subset_fn = partial(
            get_subset, 
            n_per_class=1, 
            exclude_class=query_class, 
            add_query_element=True,
            n_classes=self.dataset_config['num_classes']
        )
        
        _, query_dataset = self.dataset_config['load_fn'](
            batch_size=self.config.get('batch_size', 64), 
            subset_fn=subset_fn, 
            train=True
        )
        
        # Extract the specific query sample from the target class using filter_query
        from src.data.datasets import filter_query
        query_data, query_label = filter_query(query_dataset, query_class)
        query_data = query_data.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
        print(f"   Query data shape: {query_data.shape}")
        print(f"   Query label: {query_label} (should be {query_class})")
        
        # Verify the query is from the excluded class
        assert query_label == query_class, f"Query should be from excluded class {query_class}, got {query_label}"
        
        # Setup privacy pipeline
        reducer = self._get_reducer()
        estimator = self._get_density_estimator()
        
        # Visualization handled by new src.visualization module
        visualizer = None  # Not needed with new visualization system
        
        # Create privacy config and pipeline
        privacy_config = PrivacyConfig(
            estimator=estimator,
            reducer=reducer,
            visualizer=visualizer
        )
        privacy_pipeline = PrivacyPipeline(privacy_config)
        
        # Prepare data for privacy evaluation
        # Traditional reducers (UMAP, t-SNE, ISOMAP) need flattened data
        # Neural reducers (autoencoder, etc.) need spatial dimensions
        if self.reducer_type in ['umap', 'tsne', 'isomap', 'none']:
            # Flatten for traditional reducers
            query_input = query_data.flatten(1)
            victim_input = synthetic_victim.flatten(1)     # Numerator: WITH query
            ablated_input = synthetic_ablated.flatten(1)   # Denominator: WITHOUT query
            print(f"📐 Flattened data shapes:")
            print(f"   Query: {query_input.shape}")
            print(f"   Victim (numerator): {victim_input.shape}")
            print(f"   Ablated (denominator): {ablated_input.shape}")
        else:
            # Keep spatial dimensions for neural reducers
            query_input = query_data
            victim_input = synthetic_victim     # Numerator: WITH query
            ablated_input = synthetic_ablated   # Denominator: WITHOUT query
            print(f"📐 Spatial data shapes:")
            print(f"   Query: {query_input.shape}")
            print(f"   Victim (numerator): {victim_input.shape}")
            print(f"   Ablated (denominator): {ablated_input.shape}")
        
        # Evaluate privacy using density ratio estimation
        print(f"🔍 Starting privacy evaluation...")
        privacy_results = privacy_pipeline.evaluate(
            query_data=query_input,
            synthetic_data=victim_input,    # Numerator: synthetic data from victim model (WITH query)
            real_data=ablated_input         # Denominator: synthetic data from ablated model (WITHOUT query)
        )
        
        empirical_epsilon = privacy_results['epsilon']
        
        # Create projection comparison plot if plotting is enabled
        if self.enable_plotting:
            try:
                os.makedirs("figs", exist_ok=True)
                print("📊 Creating projection comparison plot...")
                
                # Use the projected/reduced data from privacy_results instead of raw input
                projected_query = privacy_results['query_data']
                projected_victim = privacy_results['synthetic_data']  # This was the victim/numerator data
                projected_ablated = privacy_results['real_data']     # This was the ablated/denominator data
                
                # Convert to numpy if needed
                if isinstance(projected_query, torch.Tensor):
                    projected_query = projected_query.cpu().numpy()
                if isinstance(projected_victim, torch.Tensor):
                    projected_victim = projected_victim.cpu().numpy()
                if isinstance(projected_ablated, torch.Tensor):
                    projected_ablated = projected_ablated.cpu().numpy()
                
                viz.plot_projections_comparison(
                    projected_numerator=projected_victim,
                    projected_denominator=projected_ablated,
                    projected_query=projected_query,
                    save_path="figs/projection_comparison.png"
                )
                
                # Also create hexbin difference plot
                print("📊 Creating hexbin difference plot...")
                viz.plot_hexbin_difference(
                    numerator=projected_victim,
                    denominator=projected_ablated,
                    projected_query=projected_query,
                    save_path="figs/hexbin_difference.png"
                )
                
                # Log artifacts if we have an active run
                try:
                    mlflow.log_artifact("figs/projection_comparison.png")
                    mlflow.log_artifact("figs/hexbin_difference.png")
                except:
                    print(f"   ⚠️  Could not log privacy visualization artifacts to MLflow")
            except Exception as e:
                print(f"   Warning: Privacy visualization plotting failed: {e}")
        
        # Additional ECDF analysis if enabled
        if self.enable_plotting:
            try:
                # Set seeds for deterministic data sampling in ECDF analysis
                torch.manual_seed(42)
                np.random.seed(42)
                
                # Use same sample count for ECDF analysis
                ecdf_sample_limit = min(privacy_samples_per_class * self.dataset_config['num_classes'], 1000)
                ecdf_per_class = ecdf_sample_limit // self.dataset_config['num_classes']
                
                # Get training data for ECDF analysis - INCLUDE ALL CLASSES for complete analysis
                train_loader, _ = self._create_dataloader(n_per_class=ecdf_per_class, exclude_class=None)
                
                # Collect training data samples
                train_samples = []
                for batch in train_loader:
                    batch_data = batch[0]  # Extract data from batch
                    for i in range(batch_data.size(0)):
                        train_samples.append(batch_data[i])
                        if len(train_samples) >= ecdf_sample_limit:
                            break
                    if len(train_samples) >= ecdf_sample_limit:
                        break
                
                # Stack samples ensuring consistent shapes
                train_data = torch.stack(train_samples[:ecdf_sample_limit])
                
                # Get real holdout data (test set) - INCLUDE ALL CLASSES for complete reference
                test_loader, _ = self._create_dataloader(n_per_class=ecdf_per_class, exclude_class=None, train=False)
                
                # Collect holdout/test data samples
                holdout_samples = []
                for batch in test_loader:
                    batch_data = batch[0]  # Extract data from batch
                    for i in range(batch_data.size(0)):
                        holdout_samples.append(batch_data[i])
                        if len(holdout_samples) >= ecdf_sample_limit:
                            break
                    if len(holdout_samples) >= ecdf_sample_limit:
                        break
                
                # Stack holdout samples ensuring consistent shapes
                holdout_data = torch.stack(holdout_samples[:ecdf_sample_limit])
                
                # Compute ECDF curves with proper privacy analysis semantics
                # - ref_data: Real holdout data (never seen during training)
                # - synthetic_data: Full model output (the main model being evaluated)
                ecdf_holdout, ecdf_full = ecdf_distance_curves(
                    train_data=train_data,
                    ref_data=holdout_data,                # Real holdout data as proper reference
                    synthetic_data=synthetic_full[:ecdf_sample_limit]  # Full model as target for privacy analysis
                )
                
                # Log x-range information for debugging
                ref_distances = ecdf_holdout.cdf.quantiles
                print(f"   📏 ECDF x-range: 0.0 to {1.5 * ref_distances.max():.4f} (1.5x ref max: {ref_distances.max():.4f})")
                
                # Plot ECDF comparison
                os.makedirs("figs", exist_ok=True)
                viz.plot_ecdf_comparison(
                    ecdf_holdout, ecdf_full,
                    save_path="figs/ecdf_comparison.png",
                )
                # Log artifact if we have an active run
                try:
                    mlflow.log_artifact("figs/ecdf_comparison.png")
                except:
                    print(f"   ⚠️  Could not log ECDF artifact to MLflow")
                
                # Compute additional privacy metrics
                distance_metrics = compute_privacy_distance_metrics(
                    train_data, holdout_data, synthetic_full[:ecdf_sample_limit]
                )
                
                # Log additional metrics if we have an active run
                try:
                    for key, value in distance_metrics.items():
                        mlflow.log_metric(f"PRI_{key}", value)
                except:
                    print(f"   ⚠️  Could not log distance metrics to MLflow")
                    
            except Exception as e:
                print(f"   Warning: ECDF analysis failed: {e}")
        
        # Log main privacy metrics if we have an active run
        try:
            mlflow.log_metric("PRI_empirical_epsilon", empirical_epsilon)
            mlflow.log_param("privacy_reducer", self.reducer_type)
            mlflow.log_param("privacy_density_method", self.density_method)
        except:
            print(f"   ⚠️  Could not log privacy metrics to MLflow")
        
        final_results = {
            'epsilon': empirical_epsilon,
            'reducer_type': self.reducer_type,
            'density_method': self.density_method
        }
        
        print(f"   Empirical ε: {empirical_epsilon:.4f} ({self.reducer_type} + {self.density_method})")
        return final_results
    
    def evaluate_fidelity(self, model: torch.nn.Module, arch: str, run_id: str) -> Dict[str, float]:
        """Evaluate fidelity metrics using proper computer vision metrics."""
        print("🎨 Evaluating fidelity (FID, LPIPS, PSNR, IS, SSIM)...")
        
        # Fidelity measures overall model quality - no class exclusion needed
        exclude_class = None
        
        # Get fidelity samples per class from config, with dataset-specific adjustments
        base_samples_per_class = self.config.get('samples_per_class', 500)
        
        # Adjust parameters based on dataset to avoid memory issues
        if self.dataset_name == 'octmnist':
            # OctMNIST-specific parameters to avoid memory pressure
            samples_per_class = min(200, base_samples_per_class)  # Cap at 200 for memory
            batch_size = 32          # Reduced from 64
            print("   🔧 Using reduced parameters for OctMNIST (memory optimization)")
        else:
            samples_per_class = base_samples_per_class
            batch_size = 64
            
        print(f"   Using {samples_per_class} samples per class for fidelity evaluation")
        
        # Generate synthetic data from all classes for comprehensive fidelity assessment
        synthetic_data = generate_data(
            model=model,
            n_classes=self.dataset_config['num_classes'],
            samples_per_class=samples_per_class,
            device=self.device,
            exclude_class=exclude_class
        )
        
        # Create synthetic dataloader
        synthetic_loader = torch.utils.data.DataLoader(
            synthetic_data, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Get real data loader - include all classes for fair fidelity comparison
        real_loader, _ = self._create_dataloader(
            n_per_class=samples_per_class, 
            exclude_class=exclude_class, 
            batch_size=batch_size
        )
        
        # Use original fidelity computation with proper metrics
        selected_metrics = {"FID", "LPIPS", "PSNR", "IS", "SSIM"}
        fidelity_results = compute_fidelity_metrics(
            real_loader=real_loader,
            fake_loader=synthetic_loader, 
            metrics=selected_metrics,
            device=self.device
        )
        
        # Log fidelity metrics - just log to current run if active
        try:
            self._log_fidelity_metrics(fidelity_results)
        except Exception as e:
            print(f"⚠️  MLflow logging failed for fidelity: {e}")
        
        print(f"   FID: {fidelity_results.get('FID', 'N/A'):.4f}")
        print(f"   LPIPS: {fidelity_results.get('LPIPS', 'N/A'):.4f}")
        print(f"   PSNR: {fidelity_results.get('PSNR', 'N/A'):.4f}")
        print(f"   IS: {fidelity_results.get('IS', 'N/A'):.4f}")
        print(f"   SSIM: {fidelity_results.get('SSIM', 'N/A'):.4f}")
        
        return fidelity_results
    
    def _log_fidelity_metrics(self, fidelity_results: Dict[str, float]):
        """Helper method to log fidelity metrics to MLflow."""
        for metric, value in fidelity_results.items():
            if metric == "FID":
                mlflow.log_metric("FID_frechet_inception_distance", value)
            elif metric == "LPIPS":
                mlflow.log_metric("FID_perceptual_similarity", value)
            elif metric == "PSNR":
                mlflow.log_metric("FID_peak_signal_noise_ratio", value)
            elif metric == "IS":
                mlflow.log_metric("FID_inception_score", value)
            elif metric == "SSIM":
                mlflow.log_metric("FID_structural_similarity", value)
            else:
                mlflow.log_metric(f"FID_{metric.lower()}", value)
    
    def evaluate_utility(self, model: torch.nn.Module, arch: str, run_id: str) -> Dict[str, float]:
        """Evaluate utility metrics."""
        print("📊 Evaluating utility...")
        
        # Generate synthetic training data - INCLUDE ALL CLASSES for fair evaluation
        utility_samples_per_class = self.config.get('samples_per_class', 1000)
        print(f"   Using {utility_samples_per_class} samples per class for utility evaluation")
        
        synthetic_data, synthetic_labels = generate_data(
            model=model,
            n_classes=self.dataset_config['num_classes'],
            samples_per_class=utility_samples_per_class,
            device=self.device,
            exclude_class=None,  # Include ALL classes for utility evaluation
            return_labels=True
        )
        
        # Get real test data - also include all classes
        test_loader, _ = self._create_dataloader(n_per_class=utility_samples_per_class, exclude_class=None, train=False)
        
        # Use all available test data - simple and effective
        total_test_samples = len(test_loader.dataset)
        test_data = torch.stack([
            test_loader.dataset[i][0] for i in range(total_test_samples)
        ])
        
        # Handle label shape inconsistencies (especially for MedMNIST datasets)
        test_labels_raw = [test_loader.dataset[i][1] for i in range(total_test_samples)]
        test_labels = torch.tensor(test_labels_raw)
        
        # Ensure labels are 1D (flatten if needed)
        if test_labels.dim() > 1:
            test_labels = test_labels.squeeze()
            
        print(f"   📊 Using all {total_test_samples} test samples")
        print(f"   🎯 Synthetic classes: {torch.unique(synthetic_labels).tolist()}")
        print(f"   🎯 Test classes: {torch.unique(test_labels).tolist()}")
        
        # Check and fix data normalization mismatch
        synthetic_range = (synthetic_data.min().item(), synthetic_data.max().item())
        test_range = (test_data.min().item(), test_data.max().item())
        
        print(f"   📊 Data ranges - Synthetic: {synthetic_range[0]:.3f}-{synthetic_range[1]:.3f}, Test: {test_range[0]:.3f}-{test_range[1]:.3f}")
        
        # Normalize both datasets to [0,1] if they're not already
        if synthetic_range[0] < 0 or synthetic_range[1] > 1:
            print(f"   🔧 Normalizing synthetic data to [0,1]...")
            synthetic_data = (synthetic_data - synthetic_data.min()) / (synthetic_data.max() - synthetic_data.min())
        
        if test_range[0] < 0 or test_range[1] > 1:
            print(f"   🔧 Normalizing test data to [0,1]...")
            test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())
        
        # Standardize both to similar distributions (zero mean, unit variance)
        print(f"   🔧 Standardizing data distributions...")
        
        # Calculate stats
        synthetic_mean, synthetic_std = synthetic_data.mean(), synthetic_data.std()
        test_mean, test_std = test_data.mean(), test_data.std()
        
        print(f"   📊 Before standardization:")
        print(f"      Synthetic: mean={synthetic_mean:.3f}, std={synthetic_std:.3f}")
        print(f"      Test: mean={test_mean:.3f}, std={test_std:.3f}")
        
        # Standardize synthetic data to match test data statistics
        synthetic_data = (synthetic_data - synthetic_mean) / (synthetic_std + 1e-8)
        synthetic_data = synthetic_data * test_std + test_mean
        
        # Verify and print final stats
        final_synthetic_mean, final_synthetic_std = synthetic_data.mean(), synthetic_data.std()
        print(f"   📊 After standardization:")
        print(f"      Synthetic: mean={final_synthetic_mean:.3f}, std={final_synthetic_std:.3f}")
        print(f"      Test: mean={test_mean:.3f}, std={test_std:.3f}")
        
        # Verify class consistency
        synthetic_classes = set(torch.unique(synthetic_labels).tolist())
        test_classes = set(torch.unique(test_labels).tolist())
        
        if synthetic_classes != test_classes:
            print(f"   ⚠️  Class mismatch detected!")
            print(f"      Synthetic: {sorted(synthetic_classes)}")
            print(f"      Test: {sorted(test_classes)}")
            print(f"   🔧 Filtering to common classes...")
            
            # Use intersection of classes
            common_classes = synthetic_classes.intersection(test_classes)
            if not common_classes:
                print(f"   ❌ No common classes found!")
                return {'accuracy': 0.0}
            
            # Filter synthetic data to common classes
            synthetic_mask = torch.isin(synthetic_labels, torch.tensor(list(common_classes)))
            synthetic_data = synthetic_data[synthetic_mask]
            synthetic_labels = synthetic_labels[synthetic_mask]
            
            # Filter test data to common classes  
            test_mask = torch.isin(test_labels, torch.tensor(list(common_classes)))
            test_data = test_data[test_mask]
            test_labels = test_labels[test_mask]
            
            print(f"   ✅ Using {len(common_classes)} common classes: {sorted(common_classes)}")
        
        # Setup utility metric with correct dataset configuration
        utility_metric = ClassificationUtilityMetric(
            epochs=self.config.get('utility_epochs', 100),
            batch_size=self.config.get('batch_size', 64),
            device=self.device,
            num_channels=self.dataset_config['num_channels'],
            num_classes=self.dataset_config['num_classes']
        )
        
        # Compute utility
        task_info = {
            'train_data': synthetic_data,
            'train_labels': synthetic_labels,
            'test_data': test_data,
            'test_labels': test_labels
        }
        
        utility_score = utility_metric.compute(synthetic_data, task_info)
        
        # Generate GradCAM analysis within the utility context
        try:
            self._create_gradcam_analysis(utility_metric.model, synthetic_data, synthetic_labels, 
                                        test_data, test_labels, arch, run_id)
        except Exception as e:
            print(f"❌ Failed to create GradCAM analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # Convert to percentage for display only
        utility_percentage = utility_score * 100.0
        
        # Log utility metrics with clear naming (as fraction 0-1) - just log to current run if active
        try:
            mlflow.log_metric("UTI_classification_accuracy", utility_score)
        except Exception as e:
            print(f"⚠️  MLflow logging failed for utility: {e}")
        
        print(f"   Classification accuracy: {utility_percentage:.2f}%")
        return {'accuracy': utility_percentage}
    
    def _create_gradcam_analysis(self, classifier_model: torch.nn.Module, 
                               synthetic_data: torch.Tensor, synthetic_labels: torch.Tensor,
                               test_data: torch.Tensor, test_labels: torch.Tensor,
                               arch: str, run_id: str):
        """Create GradCAM analysis showing what features the classifier focuses on.
        
        Args:
            classifier_model: Trained classifier model
            synthetic_data: Synthetic training data
            synthetic_labels: Synthetic labels
            test_data: Real test data  
            test_labels: Real test labels
            arch: Architecture name
            run_id: MLflow run ID
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            print(f"🔍 Generating GradCAM analysis...")
            
            # Find the last convolutional layer for GradCAM
            target_layers = []
            for name, module in classifier_model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layers = [module]
            
            if not target_layers:
                print("⚠️  No convolutional layers found for GradCAM - skipping analysis")
                return
            
            print(f"🎯 Using target layer: {target_layers[0]}")
            
            # Font configuration handled automatically by visualization module
            
            # Initialize GradCAM
            cam = GradCAM(model=classifier_model, target_layers=target_layers)
            
            # Select representative samples for visualization - reduced to match image grid
            samples_per_class = 2  # Reasonable number for comparison
            num_display_classes = min(4, self.dataset_config['num_classes'])  # Limit display classes
            
            fig, axes = plt.subplots(num_display_classes, samples_per_class * 2, 
                                   figsize=(12, 3 * num_display_classes))
            
            # Ensure axes is 2D for consistent indexing
            if num_display_classes == 1:
                axes = axes.reshape(1, -1)
            elif samples_per_class * 2 == 1:
                axes = axes.reshape(-1, 1)
            
            samples_found = False
            
            for class_idx in range(num_display_classes):
                # Get synthetic samples for this class
                synthetic_mask = synthetic_labels == class_idx
                synthetic_available = synthetic_mask.sum() > 0
                
                # Get real samples for this class
                real_mask = test_labels == class_idx
                real_available = real_mask.sum() > 0
                
                if synthetic_available and real_available:
                    synthetic_samples = synthetic_data[synthetic_mask][:samples_per_class]
                    real_samples = test_data[real_mask][:samples_per_class]
                    
                    # Generate GradCAMs for available samples
                    num_samples = min(len(synthetic_samples), len(real_samples), samples_per_class)
                    
                    for sample_idx in range(num_samples):
                        samples_found = True
                        
                        # Synthetic GradCAM
                        syn_input = synthetic_samples[sample_idx:sample_idx+1]
                        syn_target = [ClassifierOutputTarget(class_idx)]
                        syn_grayscale_cam = cam(input_tensor=syn_input, targets=syn_target)
                        syn_grayscale_cam = syn_grayscale_cam[0, :]
                        
                        # Real GradCAM
                        real_input = real_samples[sample_idx:sample_idx+1]
                        real_target = [ClassifierOutputTarget(class_idx)]
                        real_grayscale_cam = cam(input_tensor=real_input, targets=real_target)
                        real_grayscale_cam = real_grayscale_cam[0, :]
                        
                        # Plot synthetic
                        col_idx = sample_idx * 2
                        if num_display_classes > 1:
                            ax_syn = axes[class_idx, col_idx]
                            ax_real = axes[class_idx, col_idx + 1]
                        else:
                            ax_syn = axes[col_idx]
                            ax_real = axes[col_idx + 1]
                        
                        # Show original image
                        syn_img = syn_input[0, 0].cpu().numpy()
                        ax_syn.imshow(syn_img, cmap='gray')
                        ax_syn.imshow(syn_grayscale_cam, alpha=0.4, cmap='jet')
                        ax_syn.axis('off')
                        
                        # Show real image
                        real_img = real_input[0, 0].cpu().numpy()
                        ax_real.imshow(real_img, cmap='gray')
                        ax_real.imshow(real_grayscale_cam, alpha=0.4, cmap='jet')
                        ax_real.axis('off')
                
                else:
                    print(f"⚠️  Insufficient samples for class {class_idx} - synthetic: {synthetic_available}, real: {real_available}")
                    # Fill empty plots
                    for sample_idx in range(samples_per_class):
                        col_idx = sample_idx * 2
                        if num_display_classes > 1:
                            axes[class_idx, col_idx].text(0.5, 0.5, f'Class {class_idx}\nNo Synthetic', 
                                                        ha='center', va='center', transform=axes[class_idx, col_idx].transAxes)
                            axes[class_idx, col_idx].axis('off')
                            axes[class_idx, col_idx + 1].text(0.5, 0.5, f'Class {class_idx}\nNo Real', 
                                                            ha='center', va='center', transform=axes[class_idx, col_idx + 1].transAxes)
                            axes[class_idx, col_idx + 1].axis('off')
                        else:
                            axes[col_idx].text(0.5, 0.5, f'Class {class_idx}\nNo Synthetic', 
                                             ha='center', va='center', transform=axes[col_idx].transAxes)
                            axes[col_idx].axis('off')
                            axes[col_idx + 1].text(0.5, 0.5, f'Class {class_idx}\nNo Real', 
                                                  ha='center', va='center', transform=axes[col_idx + 1].transAxes)
                            axes[col_idx + 1].axis('off')
            
            if not samples_found:
                print("⚠️  No valid samples found for GradCAM analysis")
                plt.close(fig)
                return
            
            plt.tight_layout()
            
            # Save and log to MLflow (simply add as artifact to current run)
            gradcam_path = f"gradcam_analysis_{arch}.png"
            plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow if we have an active run
            try:
                mlflow.log_artifact(gradcam_path)
                print(f"🔍 Saved GradCAM analysis: {gradcam_path}")
            except Exception as mlflow_error:
                print(f"⚠️  MLflow logging failed for GradCAM analysis: {mlflow_error}")
                print(f"   GradCAM analysis saved locally: {gradcam_path}")
            
            # Clean up local file
            if os.path.exists(gradcam_path):
                os.remove(gradcam_path)
                
        except Exception as e:
            print(f"❌ Failed to create GradCAM analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def run_single_experiment(self, arch: str, sigma: float, epochs: int, n_per_class: int, seed: int) -> Dict[str, Any]:
        """Run a complete experiment for given parameters."""
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT: {arch.upper()} | σ={sigma} | epochs={epochs} | n_per_class={n_per_class}")
        print(f"🔒 Privacy Method: {self.reducer_type} + {self.density_method}")
        print(f"🌱 Seed: {seed}")
        if self.config.get('force_retrain', False):
            print(f"🔄 Force Retrain: ENABLED (will overwrite existing models)")
        else:
            print(f"🔄 Force Retrain: DISABLED (will reuse existing models)")
        print(f"{'='*60}")
        
        # Ensure no active run before starting
        if mlflow.active_run():
            print("⚠️  Ending existing active MLflow run...")
            mlflow.end_run()
        
        # Store current sigma for plot labeling
        self.config['current_sigma'] = sigma
        
        # Set random seeds for reproducibility
        self._set_random_seeds(seed)
        
        # Check if experiment already exists
        force_retrain = self.config.get('force_retrain', False)
        existing_run_id = self._check_existing_complete_run(arch, sigma, epochs, n_per_class, seed)
        
        new_run_started = False
        if existing_run_id and not force_retrain:
            print(f"✅ Found existing complete experiment (Run ID: {existing_run_id})")
            print("🔄 Loading existing models for evaluation...")
            run_id = existing_run_id
        else:
            if existing_run_id and force_retrain:
                print(f"🔄 Found existing experiment but forcing retrain (reusing Run ID: {existing_run_id})")
                print("🔄 Will overwrite models in existing run...")
                run_id = existing_run_id
            else:
                print(f"🆕 No existing experiment found, starting new training...")
                # Start single MLflow run for the entire experiment with descriptive name
                run_name = f"{arch}_sigma_{sigma:.2f}_seed_{seed}"
                run = mlflow.start_run(run_name=run_name)
                run_id = run.info.run_id
                new_run_started = True
                print(f"🆕 Started new experiment (Run ID: {run_id})")
                print(f"   📁 Experiment: {self.experiment_name}")
                print(f"   🏃 Run: {run_name}")
        
        # Train models if we need to (either new run or force retrain)
        if new_run_started or (existing_run_id and force_retrain):
            # Start/resume the MLflow run for training
            with mlflow.start_run(run_id=run_id):
                # Log experiment parameters (only for new runs)
                if new_run_started:
                    run_name = f"{arch}_sigma_{sigma:.2f}_seed_{seed}"
                    mlflow.log_params({
                        'arch': arch,
                        'sigma': sigma,
                        'epochs': epochs,
                        'n_per_class': n_per_class,
                        'dataset': self.dataset_name,
                        'device': str(self.device),
                        'experiment_type': 'tradeoff_analysis',
                        'privacy_reducer': self.reducer_type,
                        'privacy_density_method': self.density_method,
                        'run_name': run_name,
                        'seed': seed
                    })
                
                # Train all 3 models within the same run
                print("🎯 Training FULL model (all classes - for fidelity & utility baseline)...")
                full_model = self._train_and_save_model(
                    arch=arch, sigma=sigma, epochs=epochs, n_per_class=n_per_class,
                    exclude_class=None, add_query=False, model_name="full_model"
                )
                
                # Determine query class for privacy analysis - use class 3 for organamnist for worse privacy guarantees
                query_class = 3 if self.dataset_name == 'organamnist' else 0
                print(f"🔒 Using query class {query_class} for privacy analysis on {self.dataset_name} dataset")
                
                print("🔒 Training ABLATED model (exclude target class - privacy baseline)...")
                ablated_model = self._train_and_save_model(
                    arch=arch, sigma=sigma, epochs=epochs, n_per_class=n_per_class,
                    exclude_class=query_class, add_query=False, model_name="ablated_model"
                )
                
                print("🎯 Training VICTIM model (with query sample - privacy analysis)...")
                victim_model = self._train_and_save_model(
                    arch=arch, sigma=sigma, epochs=epochs, n_per_class=n_per_class,
                    exclude_class=query_class, add_query=True, model_name="victim_model"
                )
        
        # Load models for evaluation (whether from existing or newly trained)
        with mlflow.start_run(run_id=run_id):
            try:
                full_model = mlflow.pytorch.load_model(f"runs:/{run_id}/full_model")
                ablated_model = mlflow.pytorch.load_model(f"runs:/{run_id}/ablated_model") 
                victim_model = mlflow.pytorch.load_model(f"runs:/{run_id}/victim_model")
                print("✅ Successfully loaded all 3 models")
            except Exception as e:
                print(f"❌ Error loading models: {e}")
                raise
        
        # Evaluate all metrics using appropriate models within a single run context
        print("📊 Starting comprehensive evaluation...")
        
        # Start a single run context for all evaluations
        with mlflow.start_run(run_id=run_id):
            
            # Create synthetic image grid from full model within the run context
            self._create_synthetic_image_grid(full_model, arch, run_id)
            
            # Create victim model query visualization
            query_class = 3 if self.dataset_name == 'organamnist' else 0
            self._create_victim_model_visualization(victim_model, ablated_model, query_class, arch, run_id)
            
            # Initialize results with sentinel values (not valid results)
            privacy_results = {}
            fidelity_results = {}
            utility_results = {}
            
            # Privacy: Compare ablated vs victim models (and use full model for ECDF)
            try:
                privacy_results = self.evaluate_privacy(ablated_model, victim_model, full_model, arch, run_id)
                print(f"✅ Privacy evaluation completed: ε = {privacy_results['epsilon']:.4f}")
            except Exception as e:
                print(f"❌ Privacy evaluation failed: {e}")
                print("   Using fallback privacy values")
                privacy_results = {'epsilon': float('nan')}  # Use NaN to indicate failure
                mlflow.log_metric("PRI_evaluation_error", 1.0)
                mlflow.set_tag("privacy_error", str(e))
            
            # Fidelity: Use full model (best quality baseline)
            try:
                fidelity_results = self.evaluate_fidelity(full_model, arch, run_id)
                print(f"✅ Fidelity evaluation completed: FID = {fidelity_results.get('FID', 'N/A'):.4f}")
            except Exception as e:
                print(f"❌ Fidelity evaluation failed: {e}")
                print("   Using fallback fidelity values")
                fidelity_results = {'FID': float('nan'), 'LPIPS': float('nan'), 'PSNR': float('nan'), 'IS': float('nan'), 'SSIM': float('nan')}
                mlflow.log_metric("FID_evaluation_error", 1.0)
                mlflow.set_tag("fidelity_error", str(e))
            
            # Utility: Use full model (best utility baseline)
            try:
                utility_results = self.evaluate_utility(full_model, arch, run_id)
                print(f"✅ Utility evaluation completed: Accuracy = {utility_results['accuracy']:.2f}%")
            except Exception as e:
                print(f"❌ Utility evaluation failed: {e}")
                print("   Using fallback utility values")
                utility_results = {'accuracy': float('nan')}  # Use NaN to indicate failure
                mlflow.log_metric("UTI_evaluation_error", 1.0)
                mlflow.set_tag("utility_error", str(e))
            
            # Log tradeoff summary with clear naming
            epsilon = privacy_results.get('epsilon', float('nan'))
            accuracy = utility_results.get('accuracy', float('nan'))
            
            mlflow.log_metric("PRI_empirical_epsilon", epsilon)
            # Convert percentage back to fraction for MLflow logging
            accuracy_fraction = accuracy / 100.0 if not math.isnan(accuracy) else float('nan')
            mlflow.log_metric("UTI_classification_accuracy", accuracy_fraction)
            
            # Log tradeoff metric (with safe calculation)
            try:
                if not (math.isnan(epsilon) or math.isnan(accuracy)):
                    # Both values are valid
                    tradeoff_score = accuracy * (1.0 / max(abs(epsilon), 0.1))
                else:
                    # One or both values are invalid
                    tradeoff_score = float('nan')
                    print(f"⚠️  Cannot calculate tradeoff: ε={epsilon}, accuracy={accuracy}")
                
                mlflow.log_metric("TRADEOFF_utility_privacy_score", tradeoff_score)
            except Exception as e:
                print(f"⚠️  Tradeoff calculation failed: {e}")
                tradeoff_score = float('nan')
                mlflow.log_metric("TRADEOFF_utility_privacy_score", tradeoff_score)
                mlflow.set_tag("tradeoff_error", str(e))
            
            # Log model usage metadata
            mlflow.log_dict({
                'model_usage': {
                    'privacy_analysis': {'ablated': 'ablated_model', 'victim': 'victim_model'},
                    'fidelity_analysis': 'full_model',
                    'utility_analysis': 'full_model'
                },
                'evaluation_status': {
                    'privacy_completed': 'epsilon' in privacy_results,
                    'fidelity_completed': 'FID' in fidelity_results, 
                    'utility_completed': 'accuracy' in utility_results
                }
            }, "model_usage.json")
            
            # Explicitly mark run as successful
            mlflow.set_tag("run_status", "completed")
            print("✅ MLflow run completed successfully")
        
        # Compile comprehensive results
        results = {
            'run_id': run_id,
            'arch': arch,
            'sigma': sigma,
            'epochs': epochs,
            'n_per_class': n_per_class,
            'seed': seed,
            'models': {
                'full_model': 'full_model',
                'ablated_model': 'ablated_model', 
                'victim_model': 'victim_model'
            },
            'privacy': privacy_results,
            'fidelity': fidelity_results,
            'utility': utility_results,
            'tradeoff_score': tradeoff_score
        }
        
        print(f"✅ Experiment completed!")
        
        # Safe display of results with NaN handling
        epsilon = privacy_results.get('epsilon', float('nan'))
        fid_score = fidelity_results.get('FID', float('nan'))
        accuracy = utility_results.get('accuracy', float('nan'))
        
        print(f"   🔒 Privacy ε: {epsilon if not math.isnan(epsilon) else 'Failed'}")
        print(f"   🎨 Fidelity FID: {fid_score if not math.isnan(fid_score) else 'Failed'}")
        print(f"   📊 Utility: {accuracy if not math.isnan(accuracy) else 'Failed'}{'%' if not math.isnan(accuracy) else ''}")
        print(f"   ⚖️  Tradeoff Score: {tradeoff_score if not math.isnan(tradeoff_score) else 'Failed'}")
        print(f"   🆔 Run ID: {run_id}")
        print(f"")
        print(f"📊 MLflow Metrics:")
        print(f"   PRI_empirical_epsilon: {epsilon if not math.isnan(epsilon) else 'NaN'}")
        print(f"   FID_frechet_inception_distance: {fid_score if not math.isnan(fid_score) else 'NaN'}")
        print(f"   UTI_classification_accuracy: {accuracy if not math.isnan(accuracy) else 'NaN'}")
        print(f"   TRADEOFF_utility_privacy_score: {tradeoff_score if not math.isnan(tradeoff_score) else 'NaN'}")
        
        return results
    
    def _check_existing_complete_run(self, arch: str, sigma: float, epochs: int, n_per_class: int, seed: int) -> Optional[str]:
        """Check if a complete experiment (with all 3 models) already exists."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.arch = '{arch}' and params.sigma = '{sigma}' and "
                            f"params.epochs = '{epochs}' and params.n_per_class = '{n_per_class}' and "
                            f"params.seed = '{seed}' and "
                            f"params.experiment_type = 'tradeoff_analysis'"
            )
            
            for _, run in runs.iterrows():
                run_id = run['run_id']
                # Check if all 3 models exist
                try:
                    mlflow.pytorch.load_model(f"runs:/{run_id}/full_model")
                    mlflow.pytorch.load_model(f"runs:/{run_id}/ablated_model")
                    mlflow.pytorch.load_model(f"runs:/{run_id}/victim_model")
                    return run_id  # All models exist
                except:
                    continue  # This run is incomplete
            
        except Exception:
            pass
        return None
    
    def _train_and_save_model(self, arch: str, sigma: float, epochs: int, n_per_class: int,
                             exclude_class: Optional[int], add_query: bool, model_name: str) -> torch.nn.Module:
        """Train a model and save it as an artifact in the current MLflow run."""
        
        # Create dataloader
        train_loader, _ = self._create_dataloader(
            n_per_class=n_per_class,
            exclude_class=exclude_class,
            add_query=add_query,
            batch_size=self.config.get('batch_size', 64),
            train=True
        )
        
        # Initialize model
        if arch == 'cvae':
            model = VAE(
                num_channels=self.dataset_config['num_channels'],
                latent_dim=128,
                num_classes=self.dataset_config['num_classes']
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 1e-3))
            
            train_vae(
                model=model,
                trainloader=train_loader,
                optimizer=optimizer,
                device=self.device,
                epochs=epochs,
                sigma=sigma,
                log_metrics=True
            )
            
        elif arch == 'diffusion':
            model = DiffusionModel(
                num_channels=self.dataset_config['num_channels'],
                num_classes=self.dataset_config['num_classes'],
                timesteps=400,  # TeaPearce uses 400 timesteps
                n_feat=128,     # TeaPearce's feature dimension
                drop_prob=0.1   # TeaPearce's context dropout
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 1e-4))
            
            train_diffusion(
                model=model,
                trainloader=train_loader,
                optimizer=optimizer,
                device=self.device,
                epochs=epochs,
                sigma=sigma,
                log_metrics=True
            )
            
        elif arch == 'cgan':
            generator = Generator(
                latent_dim=100,
                num_classes=self.dataset_config['num_classes'],
                img_shape=self.dataset_config['img_shape']
            ).to(self.device)
            
            discriminator = Discriminator(
                num_classes=self.dataset_config['num_classes'],
                img_shape=self.dataset_config['img_shape']
            ).to(self.device)
            
            optimizer_g = torch.optim.Adam(generator.parameters(), lr=self.config.get('learning_rate', 1e-3), betas=(0.5, 0.999))
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.config.get('learning_rate', 1e-3), betas=(0.5, 0.999))
            
            scheduler = GapAwareDStepScheduler(
                grace=5,
                thresh=0.6,
                max_d_steps=50,
                target_loss=math.log(2) if sigma == 0.0 else math.log(1)
            )
            
            train_gan(
                generator=generator,
                discriminator=discriminator,
                trainloader=train_loader,
                optimizer_G=optimizer_g,
                optimizer_D=optimizer_d,
                device=self.device,
                epochs=epochs,
                latent_dim=100,
                scheduler=scheduler,
                sigma=sigma,
                log_metrics=True
            )
            
            model = generator
        
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Save model with descriptive name
        mlflow.pytorch.log_model(model, model_name)
        
        # Log model metadata
        mlflow.log_dict({
            'model_type': model_name,
            'arch': arch,
            'exclude_class': exclude_class,
            'add_query': add_query,
            'purpose': self._get_model_purpose(model_name)
        }, f"{model_name}_metadata.json")
        
        print(f"✅ {model_name} training completed and saved")
        return model
    
    def _get_model_purpose(self, model_name: str) -> str:
        """Get the purpose description for a model type."""
        purposes = {
            'full_model': 'Baseline for fidelity and utility evaluation (trained on all classes)',
            'ablated_model': 'Privacy baseline (trained excluding target class)',
            'victim_model': 'Privacy analysis (trained with query sample from target class)'
        }
        return purposes.get(model_name, 'Unknown purpose')
    
    def _create_synthetic_image_grid(self, model: torch.nn.Module, arch: str, run_id: str):
        """Generate and save a grid of synthetic images from all classes.
        
        Args:
            model: Trained generative model
            arch: Architecture type
            run_id: MLflow run ID
        """
        try:
            import matplotlib.pyplot as plt
            import torchvision.utils as vutils
            
            print(f"🎨 Generating synthetic image grid...")
            
            # Generate images for all classes at once
            samples_per_class = 3  # Reduced from 8 to 3 for cleaner visualization
            
            # Generate data and labels together
            all_images, all_labels = generate_data(
                model=model,
                n_classes=self.dataset_config['num_classes'],
                samples_per_class=samples_per_class,
                device=self.device,
                exclude_class=None,  # Include all classes
                return_labels=True
            )
            
            print(f"   Generated {len(all_images)} images across {self.dataset_config['num_classes']} classes")
            
            # Organize images by class for proper grid layout
            class_images = []
            for class_idx in range(self.dataset_config['num_classes']):
                class_mask = all_labels == class_idx
                class_samples = all_images[class_mask][:samples_per_class]  # Ensure we get exactly samples_per_class
                class_images.append(class_samples)
            
            # Combine all class images in row-wise order (class 0 row, class 1 row, etc.)
            organized_images = torch.cat(class_images, dim=0)
            
            # Create grid - organize by class in rows
            nrow = samples_per_class  # 3 samples per row (per class)
            grid = vutils.make_grid(
                organized_images, 
                nrow=nrow,
                padding=3,
                normalize=True,
                scale_each=True
            )
            
            # Font configuration handled automatically by visualization module
            
            # Create figure with better proportions
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
            
            # Add class labels on the left side for each row
            for i in range(self.dataset_config['num_classes']):
                # Calculate y position for each class row
                row_height = grid.shape[1] // self.dataset_config['num_classes']
                y_pos = (i + 0.5) * row_height
                ax.text(-30, y_pos, f'Class {i}', 
                       ha='right', va='center', fontsize=16, fontweight='bold',
                       rotation=90)
            
            plt.tight_layout()
            
            # Save and log to MLflow (simply add as artifact to current run)
            grid_path = f"synthetic_image_grid_{arch}.png"
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow if we have an active run
            try:
                mlflow.log_artifact(grid_path)
                print(f"🎨 Saved synthetic image grid: {grid_path}")
            except Exception as mlflow_error:
                print(f"⚠️  MLflow logging failed for synthetic image grid: {mlflow_error}")
                print(f"   Synthetic image grid saved locally: {grid_path}")
            
            # Clean up local file
            if os.path.exists(grid_path):
                os.remove(grid_path)
                
        except Exception as e:
            print(f"❌ Failed to create synthetic image grid: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_victim_model_visualization(self, victim_model: torch.nn.Module, ablated_model: torch.nn.Module, 
                                          query_class: int, arch: str, run_id: str):
        """Create victim model query visualization showing impact of single sample on generation.
        
        Args:
            victim_model: Trained victim model (includes query sample)
            ablated_model: Trained ablated model (excludes query sample)
            query_class: The class index for the query sample
            arch: Architecture type
            run_id: MLflow run ID
        """
        try:
            from src.visualization.privacy import create_victim_model_query_grid
            
            print(f"🎯 Generating victim model query visualization...")
            
            # Create the comparative visualization
            grid_path = f"victim_model_query_grid_{arch}.png"
            create_victim_model_query_grid(
                victim_model=victim_model,
                query_class=query_class,
                ablated_model=ablated_model,  # Include for comparison
                samples_per_row=8,
                num_rows=3,
                device=self.device,
                save_path=grid_path
            )
            
            # Log to MLflow if we have an active run
            try:
                mlflow.log_artifact(grid_path)
                print(f"🎯 Saved victim model query visualization: {grid_path}")
            except Exception as mlflow_error:
                print(f"⚠️  MLflow logging failed for victim model visualization: {mlflow_error}")
                print(f"   Victim model visualization saved locally: {grid_path}")
            
            # Clean up local file
            if os.path.exists(grid_path):
                os.remove(grid_path)
                
        except Exception as e:
            print(f"❌ Failed to create victim model visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def run_experiments(self) -> List[Dict[str, Any]]:
        """Run all experiments specified in config."""
        all_results = []
        
        archs = self.config['architectures']
        sigmas = self.config['sigma_values']
        n_seeds = self.config.get('n_seeds', 1)  # Default to 1 if not specified
        
        print(f"🎯 STARTING EXPERIMENTS")
        print(f"🧪 MLflow Experiment: {self.experiment_name}")
        print(f"🔒 Privacy Method: {self.reducer_type} + {self.density_method}")
        print(f"📊 Will create {len(archs)} × {len(sigmas)} × {n_seeds} = {len(archs) * len(sigmas) * n_seeds} runs")
        print(f"🌱 Seeds per configuration: {n_seeds}")
        print(f"🏃 Run naming: {{arch}}_sigma_{{value}}_seed_{{seed}} (e.g., cvae_sigma_0.1_seed_0)")
        force_retrain = self.config.get('force_retrain', False)
        print(f"🔄 Force Retrain: {'ENABLED' if force_retrain else 'DISABLED'}")
        print("")
        
        for arch in archs:
            # Set epochs based on architecture (like original main2.py)
            if arch == 'cvae':
                epochs = self.config.get('vae_epochs', 1000)
            elif arch == 'cgan':
                epochs = self.config.get('gan_epochs', 10000)
            elif arch == 'diffusion':
                epochs = self.config.get('diffusion_epochs', 1000)
            else:
                epochs = self.config.get('default_epochs', 1000)
            
            n_per_class = self.config.get('n_per_class', 10)
            
            for sigma in sigmas:
                for seed in range(n_seeds):
                    try:
                        results = self.run_single_experiment(arch, sigma, epochs, n_per_class, seed)
                        all_results.append(results)
                    except Exception as e:
                        print(f"❌ Experiment failed: {arch}, σ={sigma}, seed={seed}")
                        print(f"   Error: {e}")
                        continue
        
        # Summary
        print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        print(f"Total experiments: {len(all_results)}")
        print(f"📁 MLflow Experiment: {self.experiment_name}")
        print(f"Results available in MLflow: {self.config['mlflow_uri']}")
        
        # Quick summary table
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"{'Run Name':<20} {'Sigma':<6} {'Seed':<4} {'Privacy ε':<10} {'Utility %':<10} {'Run ID':<15}")
        print("-" * 80)
        for result in all_results:
            run_name = f"{result['arch']}_sigma_{result['sigma']:.1f}_seed_{result['seed']}"
            print(f"{run_name:<20} {result['sigma']:<6.1f} {result['seed']:<4} "
                  f"{result['privacy']['epsilon']:<10.4f} "
                  f"{result['utility']['accuracy']:<10.1f} "
                  f"{result['run_id'][:12]:<15}")
        
        print(f"\n🔍 MLflow Organization:")
        print(f"   📁 Experiment: {self.experiment_name}")
        print(f"   🏃 Runs: Each sigma/architecture/seed combination")
        print(f"   📊 Compare runs to analyze privacy-utility tradeoffs")
        print(f"   📈 Aggregate across seeds for confidence intervals")
        print(f"   🎯 Use MLflow UI: mlflow ui --port 5000")
        
        return all_results

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducible experiments."""
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For deterministic CuDNN operations (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🌱 Set random seeds to {seed} for reproducibility")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config(dataset: str = 'mnist', quick_test: bool = False, 
                         privacy_reducer: str = 'umap', privacy_density_method: str = 'kde') -> Dict[str, Any]:
    """Create default configuration."""
    import os
    mlflow_uri = f'file://{os.getcwd()}/mlruns'
    
    if quick_test:
        config = {
            'experiment_name': f'dp_models_{dataset}_quick_test',
            'dataset': dataset,
            'n_seeds': 2,  # Quick test with fewer seeds
            'architectures': ['cvae'],
            'sigma_values': [0.0, 0.1],
            'vae_epochs': 10,
            'gan_epochs': 10,
            'diffusion_epochs': 10,
            'n_per_class': 10,
            'mlflow_uri': mlflow_uri,
            'force_retrain': False,
            'privacy_reducer': privacy_reducer,
            'privacy_density_method': privacy_density_method,
            'enable_plotting': True
        }
    else:
        config = {
            'experiment_name': f'dp_models_{dataset}',
            'dataset': dataset,
            'n_seeds': 5,  # Default to 5 seeds for confidence intervals
            'architectures': ['cvae', 'cgan', 'diffusion'],
            'sigma_values': [0.0, 0.1, 0.2, 0.3, 0.4],
            'vae_epochs': 1000,
            'gan_epochs': 10000,
            'diffusion_epochs': 1000,
            'n_per_class': 10,
            'mlflow_uri': mlflow_uri,
            'force_retrain': False,
            'privacy_reducer': privacy_reducer,
            'privacy_density_method': privacy_density_method,
            'enable_plotting': True
        }
    
    return config


def main():
    """Main entry point."""
    # Ensure no active MLflow runs from previous crashes
    if mlflow.active_run():
        print("⚠️  Found active MLflow run from previous session, ending it...")
        mlflow.end_run()
    
    parser = argparse.ArgumentParser(description='Complete Experimental Runner')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--arch', choices=['cvae', 'cgan', 'diffusion'], help='Architecture to run')
    parser.add_argument('--dataset', choices=['mnist', 'pathmnist', 'octmnist', 'organamnist'], 
                       default='mnist', help='Dataset to use')
    parser.add_argument('--sigmas', type=str, help='Comma-separated sigma values (e.g., 0.0,0.1,0.2)')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test with minimal epochs')
    parser.add_argument('--force_retrain', action='store_true', help='Force retrain even if models exist')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config(args.dataset, args.quick_test)
        
        # Override with command line arguments
        if args.arch:
            config['architectures'] = [args.arch]
        if args.sigmas:
            config['sigma_values'] = [float(s.strip()) for s in args.sigmas.split(',')]
        if args.force_retrain:
            config['force_retrain'] = True
    
    print("🎯 EXPERIMENTAL RUNNER")
    print("="*40)
    print(f"Dataset: {config['dataset']}")
    print(f"Architectures: {config['architectures']}")
    print(f"Sigma values: {config['sigma_values']}")
    print(f"Experiment: {config['experiment_name']}")
    print("="*40)
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_experiments()
    
    # Summary
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"Total experiments: {len(results)}")
    print(f"📁 MLflow Experiment: {config['experiment_name']}")
    print(f"Results available in MLflow: {config['mlflow_uri']}")
    
    # Quick summary table
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"{'Run Name':<20} {'Sigma':<6} {'Seed':<4} {'Privacy ε':<10} {'Utility %':<10} {'Run ID':<15}")
    print("-" * 80)
    for result in results:
        run_name = f"{result['arch']}_sigma_{result['sigma']:.1f}_seed_{result['seed']}"
        print(f"{run_name:<20} {result['sigma']:<6.1f} {result['seed']:<4} "
              f"{result['privacy']['epsilon']:<10.4f} "
              f"{result['utility']['accuracy']:<10.1f} "
              f"{result['run_id'][:12]:<15}")
    
    print(f"\n🔍 MLflow Organization:")
    print(f"   📁 Experiment: {config['experiment_name']}")
    print(f"   🏃 Runs: Each sigma/architecture/seed combination")
    print(f"   📊 Compare runs to analyze privacy-utility tradeoffs")
    print(f"   📈 Aggregate across seeds for confidence intervals")
    print(f"   🎯 Use MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main() 