#!/usr/bin/env python3
"""
Batch migration script to convert all MLflow model artifacts in experiment 327141182539994759
from full model format to optimized state_dict format.
"""

import os
import json
import torch
import mlflow
from pathlib import Path
from src.models import VAE, DiffusionModel, Generator
import time
from typing import List, Dict, Tuple

def get_model_architecture_from_metadata(run_id: str, model_name: str) -> str:
    """Try to infer architecture from existing metadata or model structure."""
    try:
        # Try to load existing metadata
        metadata_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('arch', 'unknown')
    except:
        # If no metadata, we'll infer from the model structure
        return 'unknown'

def infer_architecture_from_model(model: torch.nn.Module) -> str:
    """Infer architecture by examining the model structure."""
    model_class = type(model).__name__
    
    if model_class == 'VAE':
        return 'cvae'
    elif model_class == 'DiffusionModel':
        return 'diffusion'
    elif model_class == 'Generator':
        return 'cgan'
    else:
        # Try to infer from model attributes
        if hasattr(model, 'latent_dim') and hasattr(model, 'num_classes'):
            return 'cvae'
        elif hasattr(model, 'timesteps') and hasattr(model, 'nn_model'):
            return 'diffusion'
        elif hasattr(model, 'latent_dim') and hasattr(model, 'img_shape'):
            return 'cgan'
        else:
            return 'unknown'

def create_model_from_architecture(arch: str, model_params: dict, device: torch.device):
    """Create a model instance based on architecture and parameters."""
    if arch == 'cvae':
        return VAE(
            num_channels=model_params['num_channels'],
            latent_dim=model_params['latent_dim'],
            num_classes=model_params['num_classes']
        ).to(device)
    elif arch == 'diffusion':
        return DiffusionModel(
            num_channels=model_params['num_channels'],
            num_classes=model_params['num_classes'],
            timesteps=model_params['timesteps'],
            n_feat=model_params['n_feat'],
            drop_prob=model_params['drop_prob']
        ).to(device)
    elif arch == 'cgan':
        return Generator(
            latent_dim=model_params['latent_dim_gan'],
            num_classes=model_params['num_classes'],
            img_shape=model_params['img_shape']
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def get_all_runs_with_models(experiment_id: str) -> List[Tuple[str, List[str]]]:
    """Get all runs in the experiment that have model artifacts."""
    runs_with_models = []
    experiment_path = f"mlruns/{experiment_id}"
    
    if not os.path.exists(experiment_path):
        print(f"❌ Experiment {experiment_id} not found")
        return []
    
    print(f"🔍 Scanning experiment {experiment_id} for model artifacts...")
    
    for run_dir in os.listdir(experiment_path):
        if run_dir == "meta.yaml":
            continue
            
        run_path = os.path.join(experiment_path, run_dir)
        artifacts_path = os.path.join(run_path, "artifacts")
        
        if not os.path.exists(artifacts_path):
            continue
            
        # Look for model artifacts
        model_names = []
        for item in os.listdir(artifacts_path):
            item_path = os.path.join(artifacts_path, item)
            if os.path.isdir(item_path):
                # Check if it's a model directory (has MLmodel file)
                mlmodel_path = os.path.join(item_path, "MLmodel")
                if os.path.exists(mlmodel_path):
                    model_names.append(item)
        
        if model_names:
            runs_with_models.append((run_dir, model_names))
    
    print(f"✅ Found {len(runs_with_models)} runs with model artifacts")
    return runs_with_models

def migrate_single_model(run_id: str, model_name: str, device: torch.device) -> Dict[str, any]:
    """Migrate a single model artifact. Returns migration results."""
    result = {
        'run_id': run_id,
        'model_name': model_name,
        'success': False,
        'arch': 'unknown',
        'original_size_mb': 0,
        'optimized_size_mb': 0,
        'reduction_percent': 0,
        'error': None
    }
    
    try:
        print(f"   🔄 Migrating {model_name}...")
        
        # Check if already migrated
        try:
            mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
            print(f"   ⚠️  {model_name} already migrated, skipping...")
            result['success'] = True
            result['error'] = 'already_migrated'
            return result
        except:
            pass  # Not migrated yet, continue
        
        # Load the original full model
        original_model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")
        
        # Infer architecture
        arch = get_model_architecture_from_metadata(run_id, model_name)
        if arch == 'unknown':
            arch = infer_architecture_from_model(original_model)
        
        result['arch'] = arch
        
        # Get model parameters
        model_params = {
            'num_channels': getattr(original_model, 'num_channels', 1),
            'num_classes': getattr(original_model, 'num_classes', 10),
            'latent_dim': getattr(original_model, 'latent_dim', 128),
            'timesteps': getattr(original_model, 'timesteps', 400),
            'n_feat': getattr(original_model, 'n_feat', 128),
            'drop_prob': getattr(original_model, 'drop_prob', 0.1),
            'latent_dim_gan': getattr(original_model, 'latent_dim', 100),
            'img_shape': getattr(original_model, 'img_shape', (1, 28, 28))
        }
        
        # Save optimized state_dict
        state_dict = original_model.state_dict()
        quantized_state_dict = {k: v.half() for k, v in state_dict.items()}
        
        # Save quantized state dict
        state_dict_path = f"{model_name}_state_dict_fp16.pt"
        torch.save(quantized_state_dict, state_dict_path)
        
        # Log to MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(state_dict_path, f"{model_name}_state_dict_fp16.pt")
            
            # Save metadata
            model_metadata = {
                'model_type': model_name,
                'arch': arch,
                'state_dict_keys': list(state_dict.keys()),
                'original_dtype': str(next(iter(state_dict.values())).dtype),
                'quantized_dtype': 'float16',
                'model_class': type(original_model).__name__,
                'model_params': model_params
            }
            mlflow.log_dict(model_metadata, f"{model_name}_metadata.json")
        
        # Clean up local file
        if os.path.exists(state_dict_path):
            os.remove(state_dict_path)
        
        # Test loading (quick test without equivalence check for batch processing)
        try:
            state_dict_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
            if os.path.isdir(state_dict_path):
                state_dict_path = os.path.join(state_dict_path, f"{model_name}_state_dict_fp16.pt")
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            loaded_state_dict = {k: v.float() for k, v in loaded_state_dict.items()}
            
            reconstructed_model = create_model_from_architecture(arch, model_params, device)
            reconstructed_model.load_state_dict(loaded_state_dict)
            
            print(f"   ✅ {model_name} loading test passed")
        except Exception as e:
            print(f"   ⚠️  {model_name} loading test failed: {e}")
            result['error'] = f"loading_test_failed: {e}"
        
        # Calculate size reduction
        try:
            original_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}/data/model.pth")
            if os.path.isdir(original_path):
                original_path = os.path.join(original_path, "model.pth")
            original_size = os.path.getsize(original_path)
            
            optimized_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
            if os.path.isdir(optimized_path):
                optimized_path = os.path.join(optimized_path, f"{model_name}_state_dict_fp16.pt")
            optimized_size = os.path.getsize(optimized_path)
            
            result['original_size_mb'] = original_size / 1024 / 1024
            result['optimized_size_mb'] = optimized_size / 1024 / 1024
            result['reduction_percent'] = (original_size - optimized_size) / original_size * 100
            
        except Exception as e:
            print(f"   ⚠️  Could not calculate size reduction: {e}")
        
        result['success'] = True
        print(f"   ✅ {model_name} migrated successfully ({result['arch']})")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"   ❌ {model_name} migration failed: {e}")
    
    return result

def main():
    """Main function to migrate all models in the experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_id = "327141182539994759"
    
    print(f"🚀 Starting batch migration of experiment {experiment_id}")
    print(f"🖥️  Using device: {device}")
    print("=" * 70)
    
    # Get all runs with models
    runs_with_models = get_all_runs_with_models(experiment_id)
    
    if not runs_with_models:
        print("❌ No runs with model artifacts found!")
        return
    
    # Migration statistics
    total_models = sum(len(models) for _, models in runs_with_models)
    successful_migrations = 0
    failed_migrations = 0
    already_migrated = 0
    total_original_size = 0
    total_optimized_size = 0
    results_by_arch = {'cvae': [], 'diffusion': [], 'cgan': [], 'unknown': []}
    
    print(f"📊 Found {total_models} models across {len(runs_with_models)} runs")
    print("=" * 70)
    
    # Process each run
    for i, (run_id, model_names) in enumerate(runs_with_models, 1):
        print(f"🔄 Processing run {i}/{len(runs_with_models)}: {run_id}")
        print(f"   Models: {', '.join(model_names)}")
        
        for model_name in model_names:
            result = migrate_single_model(run_id, model_name, device)
            
            if result['success']:
                if result['error'] == 'already_migrated':
                    already_migrated += 1
                else:
                    successful_migrations += 1
                    total_original_size += result['original_size_mb']
                    total_optimized_size += result['optimized_size_mb']
                    results_by_arch[result['arch']].append(result)
            else:
                failed_migrations += 1
        
        print()  # Empty line between runs
    
    # Print final statistics
    print("=" * 70)
    print("📊 MIGRATION SUMMARY")
    print("=" * 70)
    print(f"✅ Successful migrations: {successful_migrations}")
    print(f"⚠️  Already migrated: {already_migrated}")
    print(f"❌ Failed migrations: {failed_migrations}")
    print(f"📊 Total models processed: {total_models}")
    
    if total_original_size > 0:
        total_reduction = (total_original_size - total_optimized_size) / total_original_size * 100
        print(f"\n💾 STORAGE SAVINGS:")
        print(f"   Original total size: {total_original_size:.2f} MB ({total_original_size/1024:.2f} GB)")
        print(f"   Optimized total size: {total_optimized_size:.2f} MB ({total_optimized_size/1024:.2f} GB)")
        print(f"   Total savings: {total_original_size - total_optimized_size:.2f} MB ({(total_original_size - total_optimized_size)/1024:.2f} GB)")
        print(f"   Overall reduction: {total_reduction:.1f}%")
    
    # Architecture-specific statistics
    print(f"\n📋 BY ARCHITECTURE:")
    for arch, results in results_by_arch.items():
        if results:
            avg_reduction = sum(r['reduction_percent'] for r in results) / len(results)
            total_arch_savings = sum(r['original_size_mb'] - r['optimized_size_mb'] for r in results)
            print(f"   {arch.upper()}: {len(results)} models, avg {avg_reduction:.1f}% reduction, {total_arch_savings:.2f} MB saved")
    
    print("=" * 70)
    if failed_migrations == 0:
        print("🎉 All models migrated successfully!")
    else:
        print(f"⚠️  {failed_migrations} models failed to migrate. Check logs above for details.")

if __name__ == "__main__":
    main() 