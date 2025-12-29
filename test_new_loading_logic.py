#!/usr/bin/env python3
"""
Test script to verify that the new loading logic in run_experiments.py
works correctly with migrated model artifacts.
"""

import sys
import os
import torch
import mlflow
from run_experiments import ExperimentRunner

def test_loading_logic():
    """Test the new _load_model_optimized method with migrated models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with our migrated models
    test_cases = [
        {
            'run_id': '037904fc2b6a460d897c01a5c1c9bbc3',  # CGAN model
            'model_name': 'full_model',
            'expected_arch': 'cgan'
        },
        {
            'run_id': '2745f53d0db544b89cd51bb1b9b86517',  # Diffusion model
            'model_name': 'full_model', 
            'expected_arch': 'diffusion'
        }
    ]
    
    print(f"🧪 Testing new loading logic with migrated models")
    print(f"🖥️  Using device: {device}")
    print("=" * 60)
    
    # Create a minimal experiment instance to access the loading method
    class TestExperiment:
        def __init__(self, device):
            self.device = device
            # Minimal dataset config for model reconstruction
            self.dataset_config = {
                'num_channels': 1,
                'num_classes': 10,
                'img_shape': (1, 28, 28)
            }
        
        def _load_model_optimized(self, run_id: str, model_name: str):
            """Copy of the optimized loading method from run_experiments.py"""
            try:
                # Download and load state dict
                state_dict_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_state_dict_fp16.pt")
                if os.path.isdir(state_dict_path):
                    state_dict_path = os.path.join(state_dict_path, f"{model_name}_state_dict_fp16.pt")
                state_dict = torch.load(state_dict_path, map_location=self.device)
                
                # Convert back to float32 for inference
                state_dict = {k: v.float() for k, v in state_dict.items()}
                
                # Download metadata to get model parameters
                metadata_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{model_name}_metadata.json")
                if os.path.isdir(metadata_path):
                    metadata_path = os.path.join(metadata_path, f"{model_name}_metadata.json")
                
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruct model based on architecture
                arch = metadata['arch']
                model_params = metadata['model_params']
                
                from src.models import VAE, DiffusionModel, Generator
                
                if arch == 'cvae':
                    model = VAE(
                        num_channels=model_params['num_channels'],
                        latent_dim=model_params['latent_dim'],
                        num_classes=model_params['num_classes']
                    ).to(self.device)
                elif arch == 'diffusion':
                    model = DiffusionModel(
                        num_channels=model_params['num_channels'],
                        num_classes=model_params['num_classes'],
                        timesteps=model_params['timesteps'],
                        n_feat=model_params['n_feat'],
                        drop_prob=model_params['drop_prob']
                    ).to(self.device)
                elif arch == 'cgan':
                    model = Generator(
                        latent_dim=model_params['latent_dim_gan'],
                        num_classes=model_params['num_classes'],
                        img_shape=model_params['img_shape']
                    ).to(self.device)
                else:
                    raise ValueError(f"Unknown architecture: {arch}")
                
                # Load state dict
                model.load_state_dict(state_dict)
                model.eval()
                
                return model, arch
                
            except Exception as e:
                print(f"❌ Failed to load optimized model: {e}")
                # Fallback to original method
                return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}"), 'fallback'
    
    experiment = TestExperiment(device)
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        run_id = test_case['run_id']
        model_name = test_case['model_name']
        expected_arch = test_case['expected_arch']
        
        print(f"🧪 Test {i}: Loading {model_name} from run {run_id[:8]}...")
        print(f"   Expected architecture: {expected_arch}")
        
        try:
            # Test the new loading logic
            model, actual_arch = experiment._load_model_optimized(run_id, model_name)
            
            if actual_arch == 'fallback':
                print(f"   ⚠️  Fell back to original loading method")
                all_passed = False
                continue
            
            if actual_arch != expected_arch:
                print(f"   ❌ Architecture mismatch: expected {expected_arch}, got {actual_arch}")
                all_passed = False
                continue
            
            print(f"   ✅ Successfully loaded {actual_arch} model")
            
            # Test that the model can be used for inference
            model.eval()
            with torch.no_grad():
                if actual_arch == 'cgan':
                    test_input = torch.randn(1, 100).to(device)  # Latent vector
                    test_labels = torch.tensor([0]).to(device)
                    output = model(test_input, test_labels)
                    expected_shape = (1, 1, 28, 28)  # Generated image
                elif actual_arch == 'diffusion':
                    test_input = torch.randn(1, 1, 28, 28).to(device)
                    test_timestep = torch.tensor([50.0]).to(device)
                    test_labels = torch.tensor([0]).to(device)
                    output = model(test_input, test_timestep, test_labels)
                    expected_shape = (1, 1, 28, 28)  # Denoised image
                elif actual_arch == 'cvae':
                    test_input = torch.randn(1, 1, 28, 28).to(device)
                    test_labels = torch.tensor([0]).to(device)
                    output = model(test_input, test_labels)
                    expected_shape = (1, 1, 28, 28)  # Reconstructed image (first output)
                    if isinstance(output, tuple):
                        output = output[0]  # Take first output for shape check
                
                if output.shape == expected_shape:
                    print(f"   ✅ Model inference test passed (output shape: {output.shape})")
                else:
                    print(f"   ❌ Model inference test failed (expected {expected_shape}, got {output.shape})")
                    all_passed = False
            
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            all_passed = False
        
        print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! The new loading logic works correctly.")
        print("✅ The migrated models are compatible with run_experiments.py")
    else:
        print("❌ Some tests failed. Check the issues above.")
        return False
    
    return True

def test_integration_with_run_experiments():
    """Test that the actual run_experiments.py code works with migrated models."""
    print("\n🔧 Testing integration with run_experiments.py...")
    
    try:
        # Try to import and create an experiment instance
        from run_experiments import ExperimentRunner
        
        # Use a minimal config for testing
        config = {
            'dataset_name': 'organamnist',
            'architecture': 'cgan',  # We know we have a migrated CGAN
            'experiment_name': 'test_integration',
            'run_name': 'test_optimized_loading',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Create experiment instance
        experiment = ExperimentRunner(config)
        
        # Test loading a migrated model
        run_id = '037904fc2b6a460d897c01a5c1c9bbc3'  # Our migrated CGAN
        model_name = 'full_model'
        
        print(f"   🧪 Testing _load_model_optimized with run {run_id[:8]}...")
        
        model = experiment._load_model_optimized(run_id, model_name)
        
        if model is not None:
            print(f"   ✅ Successfully loaded model using run_experiments.py logic")
            
            # Quick inference test
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(1, 100).to(experiment.device)
                test_labels = torch.tensor([0]).to(experiment.device)
                output = model(test_input, test_labels)
                print(f"   ✅ Model inference successful (output shape: {output.shape})")
            
            return True
        else:
            print(f"   ❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing new loading logic with migrated models")
    print("=" * 70)
    
    # Test the loading logic
    loading_test_passed = test_loading_logic()
    
    # Test integration with run_experiments.py
    integration_test_passed = test_integration_with_run_experiments()
    
    print("=" * 70)
    if loading_test_passed and integration_test_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The optimization is ready for production use!")
    else:
        print("❌ Some tests failed. Review the issues above.")
        sys.exit(1) 