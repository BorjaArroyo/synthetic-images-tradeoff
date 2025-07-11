"""Dataset-level reducer management for consistent privacy analysis."""
import os
import hashlib
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
import numpy as np
import mlflow
from mlflow.sklearn import log_model as log_sklearn_model, load_model as load_sklearn_model
from functools import lru_cache

from ..core.interfaces import DimensionalityReducer
from ..data.datasets import get_dataset_config
import src.visualization as viz


class UMAPWrapper(DimensionalityReducer):
    """UMAP dimensionality reducer wrapper."""
    
    def __init__(self, **kwargs):
        try:
            from umap import UMAP
            self.umap = UMAP(**kwargs)
            self._fitted = False
        except ImportError:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    def fit(self, data: torch.Tensor):
        """Fit UMAP on data."""
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        print(f"🎯 Fitting UMAP on data shape: {data_np.shape}")
        self.umap.fit(data_np)
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted UMAP."""
        if not self._fitted:
            raise RuntimeError("UMAP must be fitted before transform")
        
        data_np = data.cpu().numpy()
        original_shape = data_np.shape
        
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        # Perform the actual transformation
        transformed = self.umap.transform(data_np)
        
        print(f"🔄 UMAP transforming data: {original_shape} -> {transformed.shape}")
        return transformed


class TSNEWrapper(DimensionalityReducer):
    """t-SNE dimensionality reducer wrapper."""
    
    def __init__(self, **kwargs):
        try:
            from sklearn.manifold import TSNE
            self.base_perplexity = kwargs.get('perplexity', 30.0)
            # Don't pass perplexity to TSNE constructor yet - we'll set it dynamically
            tsne_kwargs = {k: v for k, v in kwargs.items() if k != 'perplexity'}
            self.tsne_kwargs = tsne_kwargs
            self._fitted_data = None
        except ImportError:
            raise ImportError("sklearn not available")
    
    def fit(self, data: torch.Tensor):
        """Fit t-SNE (no actual fitting, just mark as ready)."""
        self._fitted_data = data
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using t-SNE."""
        if self._fitted_data is None:
            raise RuntimeError("t-SNE must be fitted before transform")
        
        data_np = data.cpu().numpy()
        original_shape = data_np.shape
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        # Dynamically adjust perplexity based on sample size
        n_samples = len(data_np)
        adjusted_perplexity = min(self.base_perplexity, max(1, n_samples - 1))
        
        # Create t-SNE with adjusted perplexity
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=adjusted_perplexity, **self.tsne_kwargs)
        
        # Perform the actual transformation
        transformed = tsne.fit_transform(data_np)
        
        print(f"🔄 t-SNE transforming data: {original_shape} -> {transformed.shape} (perplexity: {adjusted_perplexity})")
        return transformed


class IsomapWrapper(DimensionalityReducer):
    """ISOMAP dimensionality reducer wrapper."""
    
    def __init__(self, **kwargs):
        try:
            from sklearn.manifold import Isomap
            # Increase default neighbors for better noise robustness
            self.base_n_neighbors = kwargs.get('n_neighbors', 15)  # Increased from 5 to 15
            # Don't pass n_neighbors to Isomap constructor yet - we'll set it dynamically
            isomap_kwargs = {k: v for k, v in kwargs.items() if k != 'n_neighbors'}
            self.isomap_kwargs = isomap_kwargs
            self.isomap = None
            self._fitted = False
        except ImportError:
            raise ImportError("sklearn not available")
    
    def fit(self, data: torch.Tensor):
        """Fit ISOMAP on data."""
        import warnings
        from scipy.sparse import SparseEfficiencyWarning
        
        data_np = data.cpu().numpy()
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        # Use consistent n_neighbors for reproducible projections
        # FIXED: Don't dynamically adjust n_neighbors to avoid projection inconsistencies
        # between different data sizes (real vs synthetic)
        n_samples = len(data_np)
        if n_samples < self.base_n_neighbors:
            print(f"⚠️  Warning: Sample size ({n_samples}) smaller than n_neighbors ({self.base_n_neighbors})")
            print(f"   Using n_neighbors = {n_samples - 1} for this batch")
            adjusted_n_neighbors = max(1, n_samples - 1)
        else:
            adjusted_n_neighbors = self.base_n_neighbors
        
        print(f"🌐 Fitting ISOMAP on data shape: {data_np.shape} (n_neighbors: {adjusted_n_neighbors})")
        
        # Create ISOMAP with adjusted n_neighbors
        from sklearn.manifold import Isomap
        self.isomap = Isomap(n_neighbors=adjusted_n_neighbors, **self.isomap_kwargs)
        
        # Suppress scipy sparse efficiency warning during Isomap fitting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
            self.isomap.fit(data_np)
        
        self._fitted = True
    
    def transform(self, data: torch.Tensor) -> np.ndarray:
        """Transform data using fitted ISOMAP."""
        if not self._fitted:
            raise RuntimeError("ISOMAP must be fitted before transform")
        
        data_np = data.cpu().numpy()
        original_shape = data_np.shape
        
        if data_np.ndim > 2:
            data_np = data_np.reshape(data_np.shape[0], -1)
        
        # FIXED: Add data range validation to ensure consistency
        data_min, data_max = data_np.min(), data_np.max()
        if data_min < -0.1 or data_max > 1.1:
            print(f"⚠️  Warning: Data range [{data_min:.3f}, {data_max:.3f}] outside expected [0,1]")
            print(f"   Consider checking diffusion model output range consistency")
        
        # Perform the actual transformation
        transformed = self.isomap.transform(data_np)
        
        print(f"🔄 ISOMAP transforming data: {original_shape} -> {transformed.shape}")
        return transformed


class ReducerManager:
    """Manages dataset-level dimensionality reducers using MLflow for persistence."""
    
    def __init__(self, mlflow_tracking: bool = True, default_query_class: Optional[int] = None):
        """Initialize reducer manager.
        
        Args:
            mlflow_tracking: Whether to track reducers in MLflow
            default_query_class: Default class to highlight in visualizations (None for no highlighting)
        """
        self.mlflow_tracking = mlflow_tracking
        self.default_query_class = default_query_class
        self._fitted_reducers: Dict[str, DimensionalityReducer] = {}
    
    def set_default_query_class(self, query_class: Optional[int]):
        """Set the default query class for visualizations.
        
        Args:
            query_class: Class to highlight in visualizations (None for no highlighting)
        """
        self.default_query_class = query_class
        print(f"🎯 Set default query class to: {query_class}")
    
    def _get_reducer_key(self, dataset_name: str, reducer_type: str, 
                        reducer_params: Dict[str, Any]) -> str:
        """Generate unique key for reducer configuration.
        
        Args:
            dataset_name: Name of the dataset
            reducer_type: Type of reducer (umap, tsne, etc.)
            reducer_params: Reducer parameters
            
        Returns:
            Unique key for this reducer configuration
        """
        # Create reproducible hash of configuration
        config_str = f"{dataset_name}_{reducer_type}_{sorted(reducer_params.items())}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{dataset_name}_{reducer_type}_{config_hash}"
    
    def _get_original_training_data(self, dataset_name: str, n_samples_per_class: int = 200) -> torch.Tensor:
        """Get representative training data for reducer fitting (original real data only).
        
        Args:
            dataset_name: Name of the dataset
            n_samples_per_class: Number of samples per class to use
            
        Returns:
            Original training data tensor (no augmentation)
        """
        from ..data.datasets import get_subset
        from functools import partial
        
        dataset_config = get_dataset_config(dataset_name)
        
        # Create subset function that includes ALL classes for complete data distribution
        subset_fn = partial(
            get_subset,
            n_per_class=n_samples_per_class,
            exclude_class=None,  # Include all classes
            add_query_element=False,
            n_classes=dataset_config['num_classes']
        )
        
        # Load training data
        _, dataset = dataset_config['load_fn'](
            batch_size=64,
            num_workers=2,
            subset_fn=subset_fn,
            train=True  # Use training split
        )
        
        # Extract data (ignore labels for reducer training)
        training_data = torch.stack([dataset[i][0] for i in range(len(dataset))])
        
        print(f"📊 Original training data: {training_data.shape} from {dataset_name}")
        return training_data
    
    def _get_training_data(self, dataset_name: str, n_samples_per_class: int = 200) -> torch.Tensor:
        """Get augmented training data for reducer fitting.
        
        Args:
            dataset_name: Name of the dataset
            n_samples_per_class: Number of samples per class to use
            
        Returns:
            Augmented training data tensor
        """
        original_data = self._get_original_training_data(dataset_name, n_samples_per_class)
        return self._augment_training_data(original_data)
    
    def _augment_training_data(self, training_data: torch.Tensor) -> torch.Tensor:
        """Augment training data with noise and local mixups to make Isomap robust to DPSGD variations.
        
        Args:
            training_data: Original training data tensor
            
        Returns:
            Augmented training data tensor
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Use the same sigma values as DPSGD experiments
        sigma_values = [0.01, 0.05, 0.25, 0.5]
        
        print(f"🔧 Augmenting training data with DPSGD-matched sigma values: {sigma_values}")
        
        # For very large datasets, consider using fewer sigma values to reduce memory usage
        # e.g., sigma_values = [0.01, 0.03, 0.05] for 4x augmentation instead of 6x
        
        # Convert to list for easier manipulation
        augmented_data = [training_data]  # Start with originals
        
        # 1. Add Gaussian noise for each sigma value (skip sigma=0.00 since we already have originals)
        for sigma in sigma_values:
            if sigma > 0:
                # Generate noise for the entire batch at once
                noise = torch.normal(0, sigma, training_data.shape)
                noised = torch.clamp(training_data + noise, 0, 1)
                augmented_data.append(noised)
                print(f"   Added {len(noised)} images with sigma={sigma}")
        
        # # 2. Add local mixups using nearest neighbors
        # print("   Computing nearest neighbors for local mixup...")
        
        # # Flatten data for nearest neighbor computation
        # flat_data = training_data.flatten(1).numpy()  # (N, flattened_features)
        # neighbors = NearestNeighbors(n_neighbors=5).fit(flat_data)
        
        # mixup_images = []
        # for i, x_i in enumerate(training_data):
        #     # Find nearest neighbors for this image
        #     idx = neighbors.kneighbors([flat_data[i]], return_distance=False)[0]
        #     # Select random neighbor (excluding self at idx[0])
        #     neighbor_idx = np.random.choice(idx[1:])  # Skip self
        #     x_j = training_data[neighbor_idx]
            
        #     # Mixup with Beta distribution
        #     lam = float(np.random.beta(0.4, 0.4))
        #     mix = torch.clamp(lam * x_i + (1 - lam) * x_j, 0, 1)
        #     mixup_images.append(mix)
        
        # mixup_tensor = torch.stack(mixup_images)
        # augmented_data.append(mixup_tensor)
        # print(f"   Added {len(mixup_tensor)} local mixup samples")
        
        # Combine all augmented data
        final_training_data = torch.cat(augmented_data, dim=0)
        
        print(f"📈 Augmented dataset: {training_data.shape[0]} -> {final_training_data.shape[0]} samples")
        return final_training_data
    
    def _load_reducer_from_mlflow(self, reducer_key: str) -> Optional[DimensionalityReducer]:
        """Load fitted reducer from MLflow.
        
        Args:
            reducer_key: Unique key for this reducer
            
        Returns:
            Loaded reducer instance or None if not found
        """
        if not self.mlflow_tracking:
            return None
        
        try:
            experiment_name = f"reducers_{reducer_key.split('_')[0]}"  # e.g., "reducers_mnist"
            
            # Check if experiment exists
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    return None
            except:
                return None
            
            # Search for existing runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.reducer_key = '{reducer_key}'",
                max_results=1
            )
            
            if len(runs) > 0:
                run_id = runs.iloc[0]['run_id']
                
                # Try to load the reducer model
                try:
                    # For sklearn-compatible reducers, load the actual model
                    reducer_type = reducer_key.split('_')[1]
                    if reducer_type in ['umap', 'tsne', 'isomap']:
                        sklearn_model = mlflow.sklearn.load_model(f"runs:/{run_id}/reducer_model")
                        
                        # Create wrapper and set the fitted model
                        if reducer_type == 'umap':
                            reducer = UMAPWrapper()
                            reducer.umap = sklearn_model
                            reducer._fitted = True
                        elif reducer_type == 'tsne':
                            reducer = TSNEWrapper()
                            reducer.tsne = sklearn_model
                            reducer._fitted_data = torch.tensor([])  # Mark as fitted
                        elif reducer_type == 'isomap':
                            reducer = IsomapWrapper()
                            reducer.isomap = sklearn_model
                            reducer._fitted = True
                        
                        print(f"📂 Loaded reducer from MLflow: {reducer_key} (Run: {run_id})")
                        return reducer
                    
                    else:
                        # For neural reducers, we'll need to implement model loading
                        # For now, return None to force recreation
                        print(f"⚠️  Neural reducer loading from MLflow not yet implemented: {reducer_key}")
                        return None
                        
                except Exception as e:
                    print(f"⚠️  Failed to load reducer model from MLflow: {e}")
                    return None
            
        except Exception as e:
            print(f"⚠️  Failed to search MLflow for reducer: {e}")
            return None
        
        return None

    def _save_reducer_to_mlflow(self, reducer: DimensionalityReducer, 
                               reducer_key: str, training_data: torch.Tensor) -> Optional[str]:
        """Save fitted reducer to MLflow with labeled visualization.
        
        Args:
            reducer: Fitted reducer instance
            reducer_key: Unique key for this reducer
            training_data: Training data used for fitting
            
        Returns:
            MLflow run ID or None if saving failed
        """
        if not self.mlflow_tracking:
            return None
        
        try:
            dataset_name = reducer_key.split('_')[0]
            experiment_name = f"reducers_{dataset_name}"
            
            # Store current context to restore later
            current_run = mlflow.active_run()
            current_run_id = None
            current_experiment_id = None
            
            if current_run:
                current_run_id = current_run.info.run_id
                current_experiment_id = current_run.info.experiment_id
                # End the current run temporarily
                mlflow.end_run()
            
            # Create or set the reducer experiment
            try:
                reducer_experiment = mlflow.get_experiment_by_name(experiment_name)
                if reducer_experiment is None:
                    reducer_experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    reducer_experiment_id = reducer_experiment.experiment_id
            except Exception as e:
                print(f"⚠️  Could not create/get reducer experiment: {e}")
                # Restore the previous run if it existed
                if current_run_id:
                    mlflow.start_run(run_id=current_run_id)
                return None
            
            # Start new run in reducer experiment
            with mlflow.start_run(experiment_id=reducer_experiment_id) as run:
                run_id = run.info.run_id
                
                # Log reducer metadata
                mlflow.set_tag("reducer_key", reducer_key)
                mlflow.set_tag("dataset", dataset_name)
                mlflow.log_param("training_data_shape", str(training_data.shape))
                
                # Create labeled visualization
                try:
                    self._create_labeled_visualization(reducer, training_data, dataset_name, reducer_key, self.default_query_class)
                except Exception as e:
                    print(f"⚠️  Failed to create labeled visualization: {e}")
                
                # Save the actual reducer model
                if hasattr(reducer, 'umap'):
                    # For UMAP, save the sklearn-compatible model
                    log_sklearn_model(reducer.umap, "reducer_model")
                elif hasattr(reducer, 'isomap'):
                    # For ISOMAP, save the sklearn-compatible model
                    log_sklearn_model(reducer.isomap, "reducer_model")
                elif hasattr(reducer, '_fitted_data'):
                    # For t-SNE and others, save fitted data reference
                    mlflow.log_dict({
                        'fitted_data_info': {
                            'shape': list(training_data.shape),
                            'reducer_type': type(reducer).__name__
                        }
                    }, "fitted_data_info.json")
                
                print(f"📂 Saved reducer to MLflow: {reducer_key} (Run: {run_id})")
            
            # Restore previous run context if there was one
            if current_run_id:
                mlflow.start_run(run_id=current_run_id)
                
            return run_id
                
        except Exception as e:
            print(f"❌ Failed to save reducer to MLflow: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to restore the previous run if it existed
            if current_run_id:
                try:
                    mlflow.start_run(run_id=current_run_id)
                except:
                    pass  # If we can't restore, just continue
            
            return None
    
    def _create_labeled_visualization(self, reducer: DimensionalityReducer, 
                                    training_data: torch.Tensor, dataset_name: str, 
                                    reducer_key: str, query_class: Optional[int] = None):
        """Create and save TWO labeled visualizations of reduced data: regular + query highlighted.
        
        Args:
            reducer: Fitted reducer
            training_data: Original training data to visualize (real samples only)
            dataset_name: Name of dataset
            reducer_key: Reducer identifier
            query_class: Optional class to highlight with different marker ('x')
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from ..data.datasets import get_subset
        from functools import partial
        
        # Import visualization module for automatic font configuration
        import src.visualization as viz
        
        # Get dataset configuration and labeled data
        dataset_config = get_dataset_config(dataset_name)
        
        # Get subset with labels
        subset_fn = partial(
            get_subset,
            n_per_class=200,  # Same as training data
            exclude_class=None,
            add_query_element=False,
            n_classes=dataset_config['num_classes']
        )
        
        _, dataset = dataset_config['load_fn'](
            batch_size=64,
            num_workers=2,
            subset_fn=subset_fn,
            train=True
        )
        
        # Extract labels
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Create Plot 1: Regular visualization (no query highlighting)
        viz_path_regular = viz.create_reducer_visualization(
            reducer=reducer,
            training_data=training_data,
            labels=labels,
            reducer_key=f"{reducer_key}_regular",
            dataset_name=dataset_name,
            query_class=None  # No highlighting
        )
        
        # Upload regular plot to MLflow
        mlflow.log_artifact(viz_path_regular)
        if os.path.exists(viz_path_regular):
            os.remove(viz_path_regular)
        
        print(f"📊 Created regular visualization for {reducer_key}")
        
        # Create Plot 2: Query class highlighted (if query_class is specified)
        if query_class is not None:
            viz_path_query = viz.create_reducer_visualization(
                reducer=reducer,
                training_data=training_data,
                labels=labels,
                reducer_key=f"{reducer_key}_query_class_{query_class}",
                dataset_name=dataset_name,
                query_class=query_class
            )
            
            # Upload query plot to MLflow
            mlflow.log_artifact(viz_path_query)
            if os.path.exists(viz_path_query):
                os.remove(viz_path_query)
            
            print(f"📊 Created query class visualization for {reducer_key} with class {query_class} highlighted")
        
        print(f"✅ Uploaded {'2' if query_class is not None else '1'} visualization(s) to MLflow")

    def create_visualization_with_query_class(self, reducer: DimensionalityReducer, 
                                            training_data: torch.Tensor, dataset_name: str, 
                                            reducer_key: str, query_class: int) -> str:
        """Create and save labeled visualization with a specific query class highlighted.
        
        Args:
            reducer: Fitted reducer
            training_data: Training data to visualize
            dataset_name: Name of dataset
            reducer_key: Reducer identifier
            query_class: Class to highlight with different marker ('x')
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from ..data.datasets import get_subset
        from functools import partial
        
        # Import visualization module for automatic font configuration
        import src.visualization as viz
        
        # Get dataset configuration and labeled data
        dataset_config = get_dataset_config(dataset_name)
        
        # Get subset with labels
        subset_fn = partial(
            get_subset,
            n_per_class=200,  # Same as training data
            exclude_class=None,
            add_query_element=False,
            n_classes=dataset_config['num_classes']
        )
        
        _, dataset = dataset_config['load_fn'](
            batch_size=64,
            num_workers=2,
            subset_fn=subset_fn,
            train=True
        )
        
        # Extract labels
        labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Use the visualization function with query class
        viz_path = viz.create_reducer_visualization(
            reducer=reducer,
            training_data=training_data,
            labels=labels,
            reducer_key=reducer_key,
            dataset_name=dataset_name,
            query_class=query_class
        )
        
        print(f"📊 Created labeled visualization for {reducer_key} with query class {query_class}")
        return viz_path

    def get_or_create_reducer(self, dataset_name: str, reducer_type: str,
                             reducer_params: Optional[Dict[str, Any]] = None) -> DimensionalityReducer:
        """Get or create a dataset-level reducer.
        
        Args:
            dataset_name: Name of the dataset (mnist, pathmnist, octmnist)
            reducer_type: Type of reducer (umap, tsne, autoencoder)
            reducer_params: Parameters for the reducer
            
        Returns:
            Fitted reducer instance
        """
        if reducer_params is None:
            reducer_params = {}
        
        # Generate unique key for this configuration
        reducer_key = self._get_reducer_key(dataset_name, reducer_type, reducer_params)
        
        # Check if already in memory
        if reducer_key in self._fitted_reducers:
            print(f"🔄 Using in-memory cached reducer: {reducer_key}")
            return self._fitted_reducers[reducer_key]
        
        # Try to load from MLflow
        reducer = self._load_reducer_from_mlflow(reducer_key)
        
        if reducer is None:
            print(f"🏗️  Creating new reducer: {reducer_key}")
            
            # Get training data (consistent across all experiments)
            original_training_data = self._get_original_training_data(dataset_name)
            training_data = self._augment_training_data(original_training_data)
            
            # For neural reducers, determine correct input shape from actual data
            if reducer_type in ['autoencoder', 'normalizing_flow', 'beta_vae']:
                actual_input_shape = training_data.shape[1:]  # (C, H, W)
                print(f"📐 Detected input shape for neural reducer: {actual_input_shape}")
                reducer_params['input_shape'] = actual_input_shape
            
            # Create new reducer instance
            reducer = self._create_reducer_instance(reducer_type, reducer_params)
            
            # Fit the reducer
            print(f"🎯 Fitting reducer on {len(training_data)} training samples...")
            if reducer_type in ['umap', 'tsne', 'isomap']:
                # Traditional reducers need flattened data
                reducer.fit(training_data.flatten(1))
            else:
                # Neural reducers can work with original tensor shape
                reducer.fit(training_data)
            
            # Save to MLflow (pass original data for visualization)
            self._save_reducer_to_mlflow(reducer, reducer_key, original_training_data)
        
        # Store in memory cache for this session
        self._fitted_reducers[reducer_key] = reducer
        
        return reducer
    
    def _create_reducer_instance(self, reducer_type: str, 
                                reducer_params: Dict[str, Any]) -> DimensionalityReducer:
        """Create a new reducer instance.
        
        Args:
            reducer_type: Type of reducer
            reducer_params: Parameters for the reducer
            
        Returns:
            New reducer instance
        """
        if reducer_type == 'umap':
            return UMAPWrapper(**reducer_params)
        
        elif reducer_type == 'tsne':
            return TSNEWrapper(**reducer_params)
        
        elif reducer_type == 'isomap':
            return IsomapWrapper(**reducer_params)
        
        elif reducer_type == 'autoencoder':
            # Import the CNN-based autoencoder from models
            from ..models.reducers import AutoencoderReducer
            
            # Use dynamically detected input shape or default
            input_shape = reducer_params.pop('input_shape', (1, 28, 28))
            
            print(f"🧠 Creating CNN-based Autoencoder for shape {input_shape} - automatically adapts to dataset channels")
            return AutoencoderReducer(
                input_shape=input_shape,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **reducer_params
            )
        
        elif reducer_type == 'normalizing_flow':
            # Import the CNN-based normalizing flow from models
            from ..models.reducers import NormalizingFlowReducer
            
            # Use dynamically detected input shape or default
            input_shape = reducer_params.pop('input_shape', (1, 28, 28))
            
            print(f"🌊 Creating CNN-based Normalizing Flow for shape {input_shape} - optimal for distribution coherence")
            return NormalizingFlowReducer(
                input_shape=input_shape,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **reducer_params
            )
        
        elif reducer_type == 'beta_vae':
            # Import the CNN-based β-VAE from models
            from ..models.reducers import BetaVAEReducer
            
            # Use dynamically detected input shape or default
            input_shape = reducer_params.pop('input_shape', (1, 28, 28))
            
            print(f"🎯 Creating CNN-based β-VAE for shape {input_shape} - disentangled representations")
            return BetaVAEReducer(
                input_shape=input_shape,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                **reducer_params
            )
        
        else:
            raise ValueError(f"Unknown reducer type: {reducer_type}")


# Global reducer manager instance
_reducer_manager = None

def get_reducer_manager(mlflow_tracking: bool = True, default_query_class: Optional[int] = None) -> ReducerManager:
    """Get the global reducer manager instance.
    
    Args:
        mlflow_tracking: Whether to track reducers in MLflow
        default_query_class: Default class to highlight in visualizations
        
    Returns:
        ReducerManager instance
    """
    global _reducer_manager
    if _reducer_manager is None:
        _reducer_manager = ReducerManager(mlflow_tracking=mlflow_tracking, default_query_class=default_query_class)
    return _reducer_manager 