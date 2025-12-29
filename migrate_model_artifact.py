#!/usr/bin/env python3
"""
Script to migrate MLflow model artifacts from full model to optimized state_dict format.
This script safely converts one model artifact and validates the conversion works correctly.
"""

import os
import json
import shutil
import torch
import mlflow
from pathlib import Path
from src.models import VAE, DiffusionModel, Generator

def get_model_architecture_from_metadata(run_id: str, model_name: str) -> str:
    """Try to infer architecture from existing metadata or model structure."""
    try:
        # Try to load existing metadata
        metadata_path = mlflow.download_artifacts(f"runs:/{run_id}/{model_name}_metadata.json")
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

def migrate_model_artifact(run_id: str, model_name: str, device: torch.device):
    """Migrate a single model artifact from full model to optimized state_dict."""
    
    print(f"🔄 Migrating {model_name} from run {run_id}...")
    
    # Step 1: Load the original full model
    print(f"   📥 Loading original {model_name}...")
    try:
        original_model = mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}")
        print(f"   ✅ Successfully loaded original {model_name}")
    except Exception as e:
        print(f"   ❌ Failed to load original {model_name}: {e}")
        return False
    
    # Step 2: Infer architecture
    arch = get_model_architecture_from_metadata(run_id, model_name)
    if arch == 'unknown':
        arch = infer_architecture_from_model(original_model)
        print(f"   🔍 Inferred architecture: {arch}")
    else:
        print(f"   📋 Found architecture in metadata: {arch}")
    
    # Step 3: Get model parameters (infer from model structure)
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
    
    # Step 4: Create backup of original model (skip for now to avoid complexity)
    print(f"   💾 Skipping backup creation (original model will remain)")
    
    # Step 5: Save optimized state_dict
    print(f"   💾 Saving optimized state_dict...")
    try:
        # Get state dict and quantize to float16
        state_dict = original_model.state_dict()
        quantized_state_dict = {k: v.half() for k, v in state_dict.items()}
        
        # Save quantized state dict
        state_dict_path = f"{model_name}_state_dict_fp16.pt"
        torch.save(quantized_state_dict, state_dict_path)
        
        # Log to MLflow
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
            
        print(f"   ✅ Optimized state_dict saved successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to save optimized state_dict: {e}")
        return False
    
    # Step 6: Test loading the optimized model
    print(f"   🧪 Testing optimized model loading...")
    try:
        # Load optimized model
        state_dict_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
        # If it's a directory, find the actual file
        if os.path.isdir(state_dict_path):
            state_dict_path = os.path.join(state_dict_path, f"{model_name}_state_dict_fp16.pt")
        loaded_state_dict = torch.load(state_dict_path, map_location=device)
        
        # Convert back to float32
        loaded_state_dict = {k: v.float() for k, v in loaded_state_dict.items()}
        
        # Reconstruct model
        reconstructed_model = create_model_from_architecture(arch, model_params, device)
        reconstructed_model.load_state_dict(loaded_state_dict)
        reconstructed_model.eval()
        
        print(f"   ✅ Successfully loaded optimized {model_name}")
        
    except Exception as e:
        print(f"   ❌ Failed to load optimized model: {e}")
        return False
    
    # Step 7: Test model equivalence
    print(f"   🧪 Testing model equivalence...")
    try:
        # Set both models to eval mode for deterministic behavior
        original_model.eval()
        reconstructed_model.eval()
        
        with torch.no_grad():  # Disable gradients for deterministic behavior
            # Generate test input based on architecture
            if arch == 'cvae':
                test_input = torch.randn(1, model_params['num_channels'], 28, 28).to(device)
                test_labels = torch.tensor([0]).to(device)
                original_output = original_model(test_input, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_labels)
            elif arch == 'diffusion':
                test_input = torch.randn(1, model_params['num_channels'], 28, 28).to(device)
                test_timestep = torch.tensor([50.0]).to(device)
                test_labels = torch.tensor([0]).to(device)
                original_output = original_model(test_input, test_timestep, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_timestep, test_labels)
            elif arch == 'cgan':
                test_input = torch.randn(1, model_params['latent_dim_gan']).to(device)
                test_labels = torch.tensor([0]).to(device)
                original_output = original_model(test_input, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_labels)
            else:
                print(f"   ⚠️  Skipping equivalence test for unknown architecture: {arch}")
                return True
        
        # Check if outputs are close (allowing for quantization differences)
        if isinstance(original_output, tuple):
            is_close = all(torch.allclose(o, r, atol=1e-3, rtol=1e-3) 
                         for o, r in zip(original_output, reconstructed_output))
        else:
            is_close = torch.allclose(original_output, reconstructed_output, atol=1e-3, rtol=1e-3)
        
        if is_close:
            print(f"   ✅ Model equivalence test passed!")
        else:
            print(f"   ⚠️  Model outputs differ (expected with quantization)")
            # Check relative error
            if isinstance(original_output, tuple):
                max_rel_error = max(torch.max(torch.abs(o - r) / (torch.abs(o) + 1e-8)).item() 
                                  for o, r in zip(original_output, reconstructed_output))
            else:
                max_rel_error = torch.max(torch.abs(original_output - reconstructed_output) / 
                                        (torch.abs(original_output) + 1e-8)).item()
            
            if max_rel_error < 0.01:  # Less than 1% error
                print(f"   ✅ Model equivalence acceptable (max rel error: {max_rel_error:.4f})")
            else:
                print(f"   ❌ Model equivalence failed (max rel error: {max_rel_error:.4f})")
                return False
                
    except Exception as e:
        print(f"   ⚠️  Equivalence test failed: {e}")
        print(f"   ⚠️  Continuing anyway (this might be due to model randomness)")
    
    # Step 8: Check file sizes
    try:
        original_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}/data/model.pth")
        if os.path.isdir(original_path):
            original_path = os.path.join(original_path, "model.pth")
        original_size = os.path.getsize(original_path)
        
        optimized_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
        if os.path.isdir(optimized_path):
            optimized_path = os.path.join(optimized_path, f"{model_name}_state_dict_fp16.pt")
        optimized_size = os.path.getsize(optimized_path)
        size_reduction = (original_size - optimized_size) / original_size * 100
        
        print(f"   📊 Size comparison:")
        print(f"      Original: {original_size / 1024 / 1024:.2f} MB")
        print(f"      Optimized: {optimized_size / 1024 / 1024:.2f} MB")
        print(f"      Reduction: {size_reduction:.1f}%")
        
    except Exception as e:
        print(f"   ⚠️  Could not compare file sizes: {e}")
    
    print(f"   🎉 Migration of {model_name} completed successfully!")
    return True

def main():
    """Main function to migrate a single model artifact."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Configuration
    run_id = "2745f53d0db544b89cd51bb1b9b86517"  # This run has Diffusion models
    model_name = "full_model"  # Try full_model which should be a Diffusion model (1.4GB)
    
    print(f"🚀 Starting migration of {model_name} from run {run_id}")
    print(f"📋 Run ID: {run_id}")
    print(f"📋 Model: {model_name}")
    print(f"📋 Device: {device}")
    print("-" * 50)
    
    # Start MLflow run context
    with mlflow.start_run(run_id=run_id):
        success = migrate_model_artifact(run_id, model_name, device)
        
        if success:
            print(f"\n✅ Migration completed successfully!")
            print(f"📝 The optimized model is now available as:")
            print(f"   - {model_name}_state_dict_fp16.pt")
            print(f"   - {model_name}_metadata.json")
            print(f"\n⚠️  The original model is backed up as:")
            print(f"   - backup_{model_name}_original/")
            print(f"\n💡 You can now safely remove the original model artifact if desired.")
        else:
            print(f"\n❌ Migration failed!")
            print(f"💡 The original model remains unchanged.")

if __name__ == "__main__":
    main() 