#!/usr/bin/env python3
"""
Test script to verify model optimization functionality.
This script tests the optimized model saving and loading to ensure it works correctly.
"""

import torch
import torch.nn as nn
import tempfile
import os
import json
from src.models import VAE, DiffusionModel, Generator

def create_test_model(arch: str, device: torch.device):
    """Create a small test model for the given architecture."""
    if arch == 'cvae':
        model = VAE(
            num_channels=1,
            latent_dim=32,  # Smaller for testing
            num_classes=2
        ).to(device)
    elif arch == 'diffusion':
        model = DiffusionModel(
            num_channels=1,
            num_classes=2,
            timesteps=100,  # Smaller for testing
            n_feat=64,      # Smaller for testing
            drop_prob=0.1
        ).to(device)
    elif arch == 'cgan':
        model = Generator(
            latent_dim=50,  # Smaller for testing
            num_classes=2,
            img_shape=(1, 28, 28)
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model

def test_model_optimization():
    """Test the optimized model saving and loading functionality."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test each architecture
    architectures = ['cvae', 'diffusion', 'cgan']
    
    for arch in architectures:
        print(f"\n🧪 Testing {arch.upper()} model optimization...")
        
        # Create test model
        model = create_test_model(arch, device)
        
        # Get original state dict
        original_state_dict = model.state_dict()
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"   Original model size: {original_size / 1024 / 1024:.2f} MB")
        
        # Test quantization
        quantized_state_dict = {k: v.half() for k, v in original_state_dict.items()}
        quantized_size = sum(p.numel() * 2 for p in model.parameters())  # float16 = 2 bytes
        print(f"   Quantized size (estimated): {quantized_size / 1024 / 1024:.2f} MB")
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save quantized state dict
            state_dict_path = os.path.join(temp_dir, f"{arch}_test_state_dict_fp16.pt")
            torch.save(quantized_state_dict, state_dict_path)
            
            # Check file size
            file_size = os.path.getsize(state_dict_path)
            print(f"   Actual saved file size: {file_size / 1024 / 1024:.2f} MB")
            
            # Create metadata
            metadata = {
                'arch': arch,
                'model_params': {
                    'num_channels': 1,
                    'num_classes': 2,
                    'latent_dim': 32 if arch == 'cvae' else None,
                    'timesteps': 100 if arch == 'diffusion' else None,
                    'n_feat': 64 if arch == 'diffusion' else None,
                    'drop_prob': 0.1 if arch == 'diffusion' else None,
                    'latent_dim_gan': 50 if arch == 'cgan' else None,
                    'img_shape': (1, 28, 28) if arch == 'cgan' else None
                }
            }
            
            # Save metadata
            metadata_path = os.path.join(temp_dir, f"{arch}_test_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Load and reconstruct model
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            loaded_state_dict = {k: v.float() for k, v in loaded_state_dict.items()}
            
            # Reconstruct model
            reconstructed_model = create_test_model(arch, device)
            reconstructed_model.load_state_dict(loaded_state_dict)
            
            # Test that models are equivalent
            original_output = None
            reconstructed_output = None
            
            # Generate some test input
            if arch == 'cvae':
                test_input = torch.randn(1, 1, 28, 28).to(device)
                test_labels = torch.tensor([0]).to(device)
                original_output = model(test_input, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_labels)
            elif arch == 'diffusion':
                test_input = torch.randn(1, 1, 28, 28).to(device)
                test_labels = torch.tensor([0]).to(device)
                test_timestep = torch.tensor([50.0]).to(device)  # Float, not long
                original_output = model(test_input, test_timestep, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_timestep, test_labels)
            elif arch == 'cgan':
                test_input = torch.randn(1, 50).to(device)
                test_labels = torch.tensor([0]).to(device)
                original_output = model(test_input, test_labels)
                reconstructed_output = reconstructed_model(test_input, test_labels)
            
            # Check if outputs are close (allowing for small numerical differences due to quantization)
            if original_output is not None and reconstructed_output is not None:
                if isinstance(original_output, tuple):
                    # Handle multiple outputs
                    is_close = all(torch.allclose(o, r, atol=1e-3, rtol=1e-3) 
                                 for o, r in zip(original_output, reconstructed_output))
                else:
                    is_close = torch.allclose(original_output, reconstructed_output, atol=1e-3, rtol=1e-3)
                
                if is_close:
                    print(f"   ✅ {arch.upper()} model reconstruction successful")
                else:
                    print(f"   ⚠️  {arch.upper()} model reconstruction - outputs differ (expected with quantization)")
                    # Check if the difference is reasonable (within 1% relative error)
                    if isinstance(original_output, tuple):
                        max_rel_error = max(torch.max(torch.abs(o - r) / (torch.abs(o) + 1e-8)).item() 
                                          for o, r in zip(original_output, reconstructed_output))
                    else:
                        max_rel_error = torch.max(torch.abs(original_output - reconstructed_output) / 
                                                (torch.abs(original_output) + 1e-8)).item()
                    
                    if max_rel_error < 0.01:  # Less than 1% error
                        print(f"   ✅ {arch.upper()} model reconstruction acceptable (max rel error: {max_rel_error:.4f})")
                    else:
                        print(f"   ❌ {arch.upper()} model reconstruction failed (max rel error: {max_rel_error:.4f})")
            else:
                print(f"   ⚠️  {arch.upper()} model reconstruction test skipped (no output comparison)")
        
        # Calculate size reduction
        size_reduction = (original_size - file_size) / original_size * 100
        print(f"   📊 Size reduction: {size_reduction:.1f}%")
    
    print(f"\n🎉 Model optimization test completed!")

if __name__ == "__main__":
    test_model_optimization() 