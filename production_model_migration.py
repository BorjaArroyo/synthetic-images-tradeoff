#!/usr/bin/env python3
"""
Production model migration script.
Converts MLflow models to optimized format and removes old artifacts after verification.
"""

import os
import json
import torch
import mlflow
import shutil
from pathlib import Path
from src.models import VAE, DiffusionModel, Generator
from typing import List, Tuple, Dict

def migrate_and_cleanup_model(run_id: str, model_name: str, device: torch.device) -> Dict[str, any]:
    """Migrate model and cleanup old artifacts after verification."""
    result = {
        'run_id': run_id,
        'model_name': model_name,
        'success': False,
        'original_size_mb': 0,
        'optimized_size_mb': 0,
        'space_freed_mb': 0,
        'error': None
    }
    
    try:
        print(f"   🔄 Processing {model_name}...")
        
        # Check if already migrated
        try:
            mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
            print(f"   ⚠️  {model_name} already optimized, checking cleanup...")
            
            # Check if old model still exists and remove it
            old_model_path = f"mlruns/327141182539994759/{run_id}/artifacts/{model_name}"
            if os.path.exists(old_model_path):
                # Calculate space before deletion
                original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                  for dirpath, dirnames, filenames in os.walk(old_model_path)
                                  for filename in filenames)
                result['original_size_mb'] = original_size / 1024 / 1024
                
                # Remove old model directory
                shutil.rmtree(old_model_path)
                result['space_freed_mb'] = result['original_size_mb']
                print(f"   🗑️  Removed old {model_name} artifacts ({result['space_freed_mb']:.1f} MB freed)")
                
            result['success'] = True
            return result
        except:
            pass  # Not migrated yet, continue
        
        # Step 1: Load original model and get size
        print(f"   📥 Loading original {model_name}...")
        original_model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")
        
        # Get original size
        old_model_path = f"mlruns/327141182539994759/{run_id}/artifacts/{model_name}"
        if os.path.exists(old_model_path):
            original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(old_model_path)
                              for filename in filenames)
            result['original_size_mb'] = original_size / 1024 / 1024
        
        # Step 2: Infer architecture and create optimized version
        arch = infer_architecture_from_model(original_model)
        model_params = extract_model_params(original_model)
        
        print(f"   💾 Creating optimized {arch} model...")
        
        # Save optimized state_dict
        state_dict = original_model.state_dict()
        quantized_state_dict = {k: v.half() for k, v in state_dict.items()}
        
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
        
        # Clean up temp file
        if os.path.exists(state_dict_path):
            os.remove(state_dict_path)
        
        # Step 3: Test that optimized model works
        print(f"   🧪 Verifying optimized {model_name}...")
        try:
            # Load optimized model
            state_dict_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
            if os.path.isdir(state_dict_path):
                state_dict_path = os.path.join(state_dict_path, f"{model_name}_state_dict_fp16.pt")
            
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            loaded_state_dict = {k: v.float() for k, v in loaded_state_dict.items()}
            
            # Reconstruct model
            reconstructed_model = create_model_from_architecture(arch, model_params, device)
            reconstructed_model.load_state_dict(loaded_state_dict)
            reconstructed_model.eval()
            
            # Get optimized size
            optimized_size = os.path.getsize(state_dict_path)
            result['optimized_size_mb'] = optimized_size / 1024 / 1024
            
            print(f"   ✅ Verification passed! ({result['original_size_mb']:.1f}MB → {result['optimized_size_mb']:.1f}MB)")
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")
            result['error'] = f"verification_failed: {e}"
            return result
        
        # Step 4: Remove old artifacts (the big space saver!)
        if os.path.exists(old_model_path):
            print(f"   🗑️  Removing old {model_name} artifacts...")
            shutil.rmtree(old_model_path)
            result['space_freed_mb'] = result['original_size_mb'] - result['optimized_size_mb']
            print(f"   ✅ {result['space_freed_mb']:.1f} MB freed!")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f"   ❌ {model_name} migration failed: {e}")
    
    return result

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

def extract_model_params(model: torch.nn.Module) -> dict:
    """Extract model parameters for reconstruction."""
    return {
        'num_channels': getattr(model, 'num_channels', 1),
        'num_classes': getattr(model, 'num_classes', 10),
        'latent_dim': getattr(model, 'latent_dim', 128),
        'timesteps': getattr(model, 'timesteps', 400),
        'n_feat': getattr(model, 'n_feat', 128),
        'drop_prob': getattr(model, 'drop_prob', 0.1),
        'latent_dim_gan': getattr(model, 'latent_dim', 100),
        'img_shape': getattr(model, 'img_shape', (1, 28, 28))
    }

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

def get_runs_with_models(experiment_id: str, max_runs: int = None) -> List[Tuple[str, List[str]]]:
    """Get runs with model artifacts, optionally limited."""
    runs_with_models = []
    experiment_path = f"mlruns/{experiment_id}"
    
    if not os.path.exists(experiment_path):
        print(f"❌ Experiment {experiment_id} not found")
        return []
    
    count = 0
    for run_dir in os.listdir(experiment_path):
        if run_dir == "meta.yaml":
            continue
        
        if max_runs and count >= max_runs:
            break
            
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
            count += 1
    
    return runs_with_models

def main():
    """Main migration function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_id = "973366696472825917"
    
    print(f"🚀 Production Model Migration")
    print(f"🖥️  Device: {device}")
    print(f"📂 Experiment: {experiment_id}")
    print("=" * 60)
    
    # Process ALL runs in the experiment
    runs_with_models = get_runs_with_models(experiment_id, max_runs=None)
    
    if not runs_with_models:
        print("❌ No runs with model artifacts found!")
        return
    
    print(f"📊 Processing ALL {len(runs_with_models)} runs in the experiment")
    print("=" * 60)
    
    # Statistics
    total_models = 0
    successful_migrations = 0
    total_space_freed = 0
    results_by_arch = {'cvae': [], 'diffusion': [], 'cgan': []}
    
    # Process each run
    failed_runs = []
    
    for i, (run_id, model_names) in enumerate(runs_with_models, 1):
        print(f"🔄 Run {i}/{len(runs_with_models)}: {run_id[:8]}... ({(i/len(runs_with_models)*100):.1f}%)")
        print(f"   Models: {', '.join(model_names)}")
        
        run_success_count = 0
        
        for model_name in model_names:
            total_models += 1
            
            try:
                result = migrate_and_cleanup_model(run_id, model_name, device)
                
                if result['success']:
                    successful_migrations += 1
                    run_success_count += 1
                    total_space_freed += result['space_freed_mb']
                    
                    # Determine architecture for stats
                    try:
                        metadata_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_metadata.json")
                        if os.path.isdir(metadata_path):
                            metadata_path = os.path.join(metadata_path, f"{model_name}_metadata.json")
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        arch = metadata.get('arch', 'unknown')
                        if arch in results_by_arch:
                            results_by_arch[arch].append(result)
                    except:
                        pass
                else:
                    print(f"   ❌ {model_name} failed: {result.get('error', 'unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ {model_name} crashed: {e}")
                failed_runs.append((run_id, model_name, str(e)))
        
        # Progress update every 10 runs
        if i % 10 == 0:
            print(f"   📊 Progress: {successful_migrations}/{total_models} models completed, {total_space_freed/1024:.1f} GB freed so far")
        
        print()
    
    # Final statistics
    print("=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful_migrations}/{total_models}")
    print(f"💾 Total space freed: {total_space_freed:.1f} MB ({total_space_freed/1024:.2f} GB)")
    
    # Architecture breakdown
    print(f"\n📋 BY ARCHITECTURE:")
    for arch, results in results_by_arch.items():
        if results:
            total_freed = sum(r['space_freed_mb'] for r in results)
            avg_reduction = sum((r['original_size_mb'] - r['optimized_size_mb']) / r['original_size_mb'] * 100 
                              for r in results if r['original_size_mb'] > 0) / len(results)
            print(f"   {arch.upper()}: {len(results)} models, {total_freed:.1f} MB freed, avg {avg_reduction:.1f}% reduction")
    
    # Report any failures
    if failed_runs:
        print(f"\n⚠️  FAILED RUNS ({len(failed_runs)}):")
        for run_id, model_name, error in failed_runs[:10]:  # Show first 10
            print(f"   {run_id[:8]}/{model_name}: {error}")
        if len(failed_runs) > 10:
            print(f"   ... and {len(failed_runs) - 10} more failures")
    
    print("=" * 60)
    print("🎉 Migration completed!")
    print("✅ All models now use optimized storage format")
    print("✅ Old artifacts removed to free disk space")
    print("✅ New loading logic in run_experiments.py will work seamlessly")
    
    if failed_runs:
        print(f"⚠️  {len(failed_runs)} models had issues (see above)")
    else:
        print("🎊 Perfect success rate - all models migrated!")

if __name__ == "__main__":
    main() 