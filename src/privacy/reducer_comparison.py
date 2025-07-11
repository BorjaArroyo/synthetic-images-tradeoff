"""Comprehensive reducer comparison for distribution coherence in privacy analysis."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import mlflow
from pathlib import Path
import os
import psutil

from ..core.interfaces import DimensionalityReducer
from .cnn_reducers import CNNAutoencoderReducer, CNNBetaVAEReducer, CNNNormalizingFlowReducer
from ..models.reducers import AutoencoderReducer, BetaVAEReducer, NormalizingFlowReducer
from .reducer_manager import IsomapWrapper


class DistributionCoherenceAnalyzer:
    """Analyze how well dimensionality reducers preserve distribution properties."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def _check_memory_and_sample(self, data_size: int, analysis_type: str = 'default') -> int:
        """Simple memory check - only limit if we're really constrained."""
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Only impose limits if memory is very low or dataset is extremely large
        if available_memory_gb < 2:  # Less than 2GB available
            return min(data_size, 1000)
        elif available_memory_gb < 4 and data_size > 10000:  # Less than 4GB and large dataset
            return min(data_size, 5000)
        elif analysis_type == 'global' and data_size > 20000:  # Pairwise distances are O(n²)
            return min(data_size, 10000)
        else:
            # Use all data - let the system handle it
            return data_size

    def evaluate_reducer_coherence(self, 
                                 reducer_type: str,
                                 dataset_name: str,
                                 train_data: torch.Tensor,
                                 test_data: torch.Tensor,
                                 labels_train: Optional[torch.Tensor] = None,
                                 labels_test: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate how well a reducer preserves distribution properties."""
        
        print(f"🔍 Evaluating {reducer_type} on {dataset_name}...")
        
        # Create and fit reducer
        input_shape = train_data.shape[1:]
        reducer = self._create_reducer(reducer_type, input_shape)
        
        # Completely suppress training output for neural networks
        if reducer_type not in ['umap', 'tsne']:
            import sys
            import os
            import logging
            
            # Save original streams
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            devnull = open(os.devnull, 'w')
            
            # Set all loggers to CRITICAL
            logging.getLogger().setLevel(logging.CRITICAL)
            for name in ['torch', 'sklearn', 'matplotlib']:
                logging.getLogger(name).setLevel(logging.CRITICAL)
            
            try:
                # Suppress all output during training
                sys.stdout = devnull
                sys.stderr = devnull
                
                print("   🧠 Training neural network...", end=" ", flush=True)
                reducer.fit(train_data)
                print("✓")
                
            finally:
                # Restore output
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                devnull.close()
                logging.getLogger().setLevel(logging.WARNING)
        else:
            # For UMAP/t-SNE, also suppress their output
            import sys
            import os
            import logging
            import warnings
            
            # Save original streams
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            devnull = open(os.devnull, 'w')
            
            # Suppress all warnings and logs
            logging.getLogger().setLevel(logging.CRITICAL)
            for name in ['umap', 'sklearn', 'matplotlib']:
                logging.getLogger(name).setLevel(logging.CRITICAL)
            warnings.filterwarnings('ignore')
            
            try:
                # Suppress all output
                sys.stdout = devnull
                sys.stderr = devnull
                
                print(f"   🔄 Fitting {reducer_type}...", end=" ", flush=True)
                reducer.fit(train_data)
                print("✓")
                
            finally:
                # Restore output
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                devnull.close()
                logging.getLogger().setLevel(logging.WARNING)
                warnings.resetwarnings()
        
        # Transform data
        train_reduced = self._transform_data(reducer, train_data)
        test_reduced = self._transform_data(reducer, test_data)
        
        # Compute ONLY density preservation - the only metric that matters for privacy
        print(f"   🧮 Computing density coherence...", end=" ", flush=True)
        
        density_results = self._compute_density_preservation_detailed(train_data, train_reduced, test_data, test_reduced)
        
        print("✓")
        
        # Extract results
        density_preservation = density_results['combined_preservation']
        distance_preservation = density_results['distance_preservation'] 
        angle_preservation = density_results['angle_preservation']
        
        # Only density matters for privacy analysis, but show detailed breakdown
        results = {
            'density_preservation': density_preservation,
            'distance_preservation': distance_preservation,
            'angle_preservation': angle_preservation,
            'aggregate_score': density_preservation  # Same as combined density
        }
        
        print(f"   ✅ DENSITY COHERENCE: {density_preservation:.3f} (dist: {distance_preservation:.3f}, angle: {angle_preservation:.3f})")
        
        return results
    
    def _create_reducer(self, reducer_type: str, input_shape: Tuple[int, ...]) -> DimensionalityReducer:
        """Factory method to create reducers using improved models."""
        if reducer_type == 'normalizing_flow':
            # Use improved normalizing flow from models folder
            return NormalizingFlowReducer(
                input_shape=input_shape, latent_dim=2, num_flows=8, 
                max_epochs=300, patience=20, device=self.device
            )
        elif reducer_type == 'beta_vae':
            # Use improved β-VAE from models folder
            return BetaVAEReducer(
                input_shape=input_shape, latent_dim=2, beta=4.0, 
                max_epochs=200, patience=15, device=self.device
            )
        elif reducer_type == 'autoencoder':
            # Use improved autoencoder from models folder
            return AutoencoderReducer(
                input_shape=input_shape, latent_dim=2, 
                max_epochs=200, patience=15, device=self.device
            )
        elif reducer_type == 'cnn_autoencoder':
            # Force CNN version (legacy)
            return CNNAutoencoderReducer(input_shape, latent_dim=2, device=self.device)
        elif reducer_type == 'cnn_beta_vae':
            # Force CNN version (legacy)
            return CNNBetaVAEReducer(input_shape, latent_dim=2, beta=4.0, device=self.device)
        elif reducer_type == 'cnn_normalizing_flow':
            # Force CNN version (legacy)
            return CNNNormalizingFlowReducer(input_shape, latent_dim=2, device=self.device)
        elif reducer_type == 'umap':
            return UMAPReducerStandalone(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        elif reducer_type == 'tsne':
            return TSNEReducerStandalone(n_components=2, perplexity=30.0, random_state=42)
        elif reducer_type == 'isomap':
            return IsomapWrapper(n_components=2, n_neighbors=15)
        else:
            raise ValueError(f"Unknown reducer type: {reducer_type}")
    
    def _transform_data(self, reducer, data: torch.Tensor) -> np.ndarray:
        """Transform data using reducer, handling different interfaces."""
        if hasattr(reducer, 'transform'):
            return reducer.transform(data)
        elif hasattr(reducer, 'encode'):
            return reducer.encode(data).cpu().numpy()
        else:
            raise ValueError(f"Reducer {type(reducer)} has no transform or encode method")
    
    def _compute_density_preservation_detailed(self, train_orig: torch.Tensor, train_reduced: np.ndarray,
                                               test_orig: torch.Tensor, test_reduced: np.ndarray) -> Dict[str, float]:
        """Compute how well density structure is preserved using k-NN density estimation and angle preservation."""
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Use all data with simple memory checks
        train_samples = self._check_memory_and_sample(len(train_orig), 'density')
        test_samples = self._check_memory_and_sample(len(test_orig), 'density')
        
        # Sample training data if needed
        if train_samples < len(train_orig):
            train_indices = np.random.choice(len(train_orig), train_samples, replace=False)
            train_orig_flat = train_orig[train_indices].flatten(1).cpu().numpy()
            train_reduced_subset = train_reduced[train_indices]
        else:
            train_orig_flat = train_orig.flatten(1).cpu().numpy()
            train_reduced_subset = train_reduced
        
        # Sample test data if needed
        if test_samples < len(test_orig):
            test_indices = np.random.choice(len(test_orig), test_samples, replace=False)
            test_orig_flat = test_orig[test_indices].flatten(1).cpu().numpy()
            test_reduced_subset = test_reduced[test_indices]
        else:
            test_orig_flat = test_orig.flatten(1).cpu().numpy()
            test_reduced_subset = test_reduced
        
        # Normalize data to [0, 1] for fair comparison across dimensions
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale original data
        scaler_orig = MinMaxScaler()
        train_orig_scaled = scaler_orig.fit_transform(train_orig_flat)
        test_orig_scaled = scaler_orig.transform(test_orig_flat)
        
        # Scale reduced data  
        scaler_reduced = MinMaxScaler()
        train_reduced_scaled = scaler_reduced.fit_transform(train_reduced_subset)
        test_reduced_scaled = scaler_reduced.transform(test_reduced_subset)
        
        # ======= DISTANCE PRESERVATION (existing) =======
        # Simple adaptive k
        k_orig = max(5, min(100, int(np.sqrt(train_samples))))
        k_reduced = max(5, min(100, int(np.sqrt(train_samples))))
        
        # Fit k-NN models
        knn_orig = NearestNeighbors(n_neighbors=k_orig, metric='euclidean')
        knn_orig.fit(train_orig_scaled)
        
        knn_reduced = NearestNeighbors(n_neighbors=k_reduced, metric='euclidean')
        knn_reduced.fit(train_reduced_scaled)
        
        # Compute k-NN distances (density ∝ 1/distance^d)
        distances_orig, _ = knn_orig.kneighbors(test_orig_scaled)
        distances_reduced, _ = knn_reduced.kneighbors(test_reduced_scaled)
        
        # Use k-th nearest neighbor distance for density estimation
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        density_orig = 1.0 / (distances_orig[:, -1] + eps)  # k-th neighbor distance
        density_reduced = 1.0 / (distances_reduced[:, -1] + eps)
        
        # Check for degenerate cases (constant densities)
        if len(np.unique(density_orig)) <= 1 or len(np.unique(density_reduced)) <= 1:
            # If densities are constant, correlation is undefined
            # Return 0.0 for no preservation
            return {
                'distance_preservation': 0.0,
                'angle_preservation': 0.0,
                'combined_preservation': 0.0
            }
        
        # Check if we have enough variance for meaningful correlation
        if np.std(density_orig) < 1e-10 or np.std(density_reduced) < 1e-10:
            return {
                'distance_preservation': 0.0,
                'angle_preservation': 0.0,
                'combined_preservation': 0.0
            }
        
        # Compute distance preservation correlation
        distance_correlation, _ = spearmanr(density_orig, density_reduced)
        distance_preservation = max(0.0, distance_correlation)
        
        # ======= ANGLE PRESERVATION (new) =======
        # Sample subset for angle computation to avoid O(n²) complexity
        max_angle_samples = min(500, len(test_orig_scaled))  # Limit for computational efficiency
        if max_angle_samples < len(test_orig_scaled):
            angle_indices = np.random.choice(len(test_orig_scaled), max_angle_samples, replace=False)
            test_orig_angle = test_orig_scaled[angle_indices]
            test_reduced_angle = test_reduced_scaled[angle_indices]
        else:
            test_orig_angle = test_orig_scaled
            test_reduced_angle = test_reduced_scaled
        
        # Compute cosine similarities (angles) between all pairs of test points
        cosine_orig = cosine_similarity(test_orig_angle)
        cosine_reduced = cosine_similarity(test_reduced_angle)
        
        # Extract unique pairwise similarities (upper triangle, exclude diagonal)
        n_angle_samples = len(test_orig_angle)
        if n_angle_samples > 1:
            # Get upper triangle indices
            triu_indices = np.triu_indices(n_angle_samples, k=1)
            cosine_orig_flat = cosine_orig[triu_indices]
            cosine_reduced_flat = cosine_reduced[triu_indices]
            
            # Check for degenerate angle cases
            if len(np.unique(cosine_orig_flat)) <= 1 or len(np.unique(cosine_reduced_flat)) <= 1:
                angle_preservation = 0.0
            elif np.std(cosine_orig_flat) < 1e-10 or np.std(cosine_reduced_flat) < 1e-10:
                angle_preservation = 0.0
            else:
                # Compute angle preservation correlation
                angle_correlation, _ = spearmanr(cosine_orig_flat, cosine_reduced_flat)
                angle_preservation = max(0.0, angle_correlation)
        else:
            angle_preservation = 0.0
        
        # ======= COMBINED DENSITY PRESERVATION =======
        # Combine distance and angle preservation with weights
        # Distance tends to be more important for density, but angles capture geometric structure
        distance_weight = 0.7  # Distance preservation is primary
        angle_weight = 0.3     # Angle preservation is secondary but important
        
        combined_preservation = (distance_weight * distance_preservation + 
                               angle_weight * angle_preservation)
        
        return {
            'distance_preservation': distance_preservation,
            'angle_preservation': angle_preservation,
            'combined_preservation': combined_preservation
        }
    
    def compare_all_reducers(self, dataset_name: str, train_data: torch.Tensor, 
                           test_data: torch.Tensor, labels_train: Optional[torch.Tensor] = None,
                           labels_test: Optional[torch.Tensor] = None,
                           include_cnn: bool = True) -> Dict[str, Dict[str, float]]:
        """Compare all available reducers for distribution coherence."""
        
        # All reducer types to test
        reducers_to_test = ['umap', 'tsne', 'normalizing_flow', 'autoencoder', 'beta_vae', 'cnn_autoencoder', 'cnn_beta_vae', 'cnn_normalizing_flow', 'isomap']
        
        # Check if input is image data
        
        results = {}
        
        for reducer_type in reducers_to_test:
            # Ensure completely clean MLflow state before each reducer
            try:
                while mlflow.active_run() is not None:
                    mlflow.end_run()
            except:
                pass
            
            try:
                # Start a parent run for this reducer
                with mlflow.start_run(run_name=f"reducer_comparison_{reducer_type}") as parent_run:
                    
                    # Evaluate the reducer (this will create nested runs for stability)
                    result = self.evaluate_reducer_coherence(
                        reducer_type=reducer_type,
                        dataset_name=dataset_name,
                        train_data=train_data,
                        test_data=test_data,
                        labels_train=labels_train,
                        labels_test=labels_test
                    )
                    results[reducer_type] = result
                    
                    # Log main results to parent run
                    for metric, value in result.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"coherence_{metric}", value)
                    mlflow.log_param("reducer_type", reducer_type)
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("is_cnn_optimized", "cnn_" in reducer_type)
                
                # Parent run automatically ended by context manager
                
            except Exception as e:
                print(f"❌ Failed to evaluate {reducer_type}: {e}")
                import traceback
                traceback.print_exc()
                # Ensure cleanup on failure
                try:
                    while mlflow.active_run() is not None:
                        mlflow.end_run()
                except:
                    pass
                continue
        
        # Print comparison table
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, float]]):
        """Print a formatted comparison table."""
        print(f"\n📊 REDUCER COMPARISON RESULTS")
        print("=" * 80)
        
        # Detailed explanation of metrics
        print(f"\n🔍 ENHANCED DENSITY EVALUATION:")
        print(f"   🎯 COMBINED DENSITY: Distance (70%) + Angle (30%) preservation for comprehensive density analysis") 
        print(f"   📏 DISTANCE: Preserves k-NN distance rankings (local density)")
        print(f"   📐 ANGLE: Preserves cosine similarity rankings (geometric structure)")
        print(f"   🧠 CNN vs FC: CNN = Convolutional, FC = Fully Connected")
        print("=" * 80)
        
        # Enhanced table header with distance and angle details
        print(f"{'Reducer':<25} {'Combined':<10} {'Distance':<10} {'Angle':<10}")
        print("-" * 80)
        
        # Sort by combined density preservation 
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1].get('density_preservation', 0), 
                               reverse=True)
        
        for reducer_type, result in sorted_results:
            # Add emoji indicators for CNN vs FC
            display_name = reducer_type
            if 'cnn_' in reducer_type:
                display_name = f"🧠 {reducer_type}"
            elif reducer_type in ['autoencoder', 'beta_vae', 'normalizing_flow']:
                display_name = f"📏 {reducer_type}"  # FC indicator
            elif reducer_type in ['umap', 'tsne', 'isomap']:
                display_name = f"🔄 {reducer_type}"  # Classic ML
            
            combined_score = result.get('density_preservation', 0.0)
            distance_score = result.get('distance_preservation', 0.0)
            angle_score = result.get('angle_preservation', 0.0)
            
            print(f"{display_name:<25} {combined_score:<10.3f} {distance_score:<10.3f} {angle_score:<10.3f}")
        
        print("=" * 80)
        
        # Recommendation
        best_reducer = sorted_results[0][0] if sorted_results else "None"
        best_result = sorted_results[0][1] if sorted_results else {}
        best_combined = best_result.get('density_preservation', 0)
        best_distance = best_result.get('distance_preservation', 0)
        best_angle = best_result.get('angle_preservation', 0)
        
        print(f"🏆 WINNER: {best_reducer.upper()}")
        print(f"   📊 Combined: {best_combined:.3f} | Distance: {best_distance:.3f} | Angle: {best_angle:.3f}")
        
        # CNN vs FC analysis based on combined density
        cnn_results = {k: v for k, v in results.items() if 'cnn_' in k}
        fc_results = {k: v for k, v in results.items() if k in ['autoencoder', 'beta_vae', 'normalizing_flow']}
        classic_results = {k: v for k, v in results.items() if k in ['umap', 'tsne', 'isomap']}
        
        if cnn_results and fc_results:
            # Combined scores
            avg_cnn_combined = np.mean([r['density_preservation'] for r in cnn_results.values()])
            avg_fc_combined = np.mean([r['density_preservation'] for r in fc_results.values()])
            
            # Distance scores
            avg_cnn_distance = np.mean([r['distance_preservation'] for r in cnn_results.values()])
            avg_fc_distance = np.mean([r['distance_preservation'] for r in fc_results.values()])
            
            # Angle scores
            avg_cnn_angle = np.mean([r['angle_preservation'] for r in cnn_results.values()])
            avg_fc_angle = np.mean([r['angle_preservation'] for r in fc_results.values()])
            
            print(f"\n🧠 ARCHITECTURE COMPARISON:")
            print(f"   🧠 CNN: Combined={avg_cnn_combined:.3f} | Distance={avg_cnn_distance:.3f} | Angle={avg_cnn_angle:.3f}")
            print(f"   📏 FC:  Combined={avg_fc_combined:.3f} | Distance={avg_fc_distance:.3f} | Angle={avg_fc_angle:.3f}")
            
            if classic_results:
                avg_classic_combined = np.mean([r['density_preservation'] for r in classic_results.values()])
                avg_classic_distance = np.mean([r['distance_preservation'] for r in classic_results.values()])
                avg_classic_angle = np.mean([r['angle_preservation'] for r in classic_results.values()])
                print(f"   🔄 Classic: Combined={avg_classic_combined:.3f} | Distance={avg_classic_distance:.3f} | Angle={avg_classic_angle:.3f}")
            
            if avg_cnn_combined > avg_fc_combined:
                print(f"   ✅ CNN wins by {avg_cnn_combined - avg_fc_combined:.3f} (combined)")
            else:
                print(f"   ⚠️  FC wins by {avg_fc_combined - avg_cnn_combined:.3f} (combined)")
            
            # Distance vs Angle insights
            print(f"\n📐 DISTANCE vs ANGLE INSIGHTS:")
            distance_leader = "CNN" if avg_cnn_distance > avg_fc_distance else "FC"
            angle_leader = "CNN" if avg_cnn_angle > avg_fc_angle else "FC"
            print(f"   📏 Distance leader: {distance_leader}")
            print(f"   📐 Angle leader: {angle_leader}")
            
            if distance_leader != angle_leader:
                print(f"   🔄 Different strengths: {distance_leader} better at distance, {angle_leader} better at angles")
            else:
                print(f"   🎯 Consistent: {distance_leader} better at both metrics")
        
        # Show top 3 combined density performers
        print(f"\n🥇 TOP 3 COMBINED DENSITY PRESERVATION:")
        for i, (reducer_name, result) in enumerate(sorted_results[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            combined = result['density_preservation']
            distance = result['distance_preservation'] 
            angle = result['angle_preservation']
            print(f"   {medal} {reducer_name.upper():<20} {combined:.3f} (d:{distance:.3f}, a:{angle:.3f})")
        
        # Distance vs Angle correlation analysis
        all_distances = [r['distance_preservation'] for r in results.values()]
        all_angles = [r['angle_preservation'] for r in results.values()]
        
        if len(all_distances) > 2 and np.std(all_distances) > 1e-6 and np.std(all_angles) > 1e-6:
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(all_distances, all_angles)
            print(f"\n🔗 DISTANCE-ANGLE CORRELATION: {corr:.3f} (p={p_value:.3f})")
            if corr > 0.5:
                print(f"   ✅ Strong positive correlation - reducers good at distance tend to be good at angles")
            elif corr < -0.5:
                print(f"   ⚠️  Strong negative correlation - distance and angle preservation trade off")
            else:
                print(f"   🔄 Weak correlation - distance and angle capture different aspects")


# Standalone reducer implementations that work with the comparison framework

class UMAPReducerStandalone(DimensionalityReducer):
    """Standalone UMAP reducer for comparison."""
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, 
                 min_dist: float = 0.1, random_state: int = 42):
        try:
            from umap import UMAP
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state
            )
            self._fitted = False
        except ImportError:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    def fit(self, data: torch.Tensor):
        """Fit UMAP on data."""
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        print(f"   Fitting UMAP on {data_np.shape} data...")
        self.umap.fit(data_np)
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted UMAP."""
        if not self._fitted:
            raise RuntimeError("UMAP must be fitted before transform")
        
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        return self.umap.transform(data_np)


class TSNEReducerStandalone(DimensionalityReducer):
    """Standalone t-SNE reducer for comparison."""
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, random_state: int = 42):
        try:
            from sklearn.manifold import TSNE
            self.tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            self._fitted_data = None
            self._fitted_reduced = None
        except ImportError:
            raise ImportError("sklearn not available")
    
    def fit(self, data: torch.Tensor):
        """Fit t-SNE (store training data for transforms)."""
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        print(f"   Fitting t-SNE on {data_np.shape} data...")
        self._fitted_data = data_np
        self._fitted_reduced = self.tsne.fit_transform(data_np)
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using t-SNE (refit on combined data)."""
        if self._fitted_data is None:
            raise RuntimeError("t-SNE must be fitted before transform")
        
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        # For t-SNE, we need to refit on combined data
        # This is a limitation of t-SNE - no true out-of-sample transforms
        combined_data = np.vstack([self._fitted_data, data_np])
        combined_reduced = self.tsne.fit_transform(combined_data)
        
        # Return only the new data portion
        return combined_reduced[len(self._fitted_data):]

