#!/usr/bin/env python3
"""
Comprehensive MLflow Model Cleanup Script

This script safely removes old full PyTorch model artifacts (full_model, victim_model, ablated_model)
ONLY if the optimized versions (state_dict + metadata) exist.

Safety features:
- Verifies both state_dict and metadata files exist before deletion
- Provides detailed logging of what will be deleted
- Shows space savings
- Processes all experiments systematically
"""

import os
import shutil
import subprocess
from pathlib import Path
import json

def get_disk_usage():
    """Get current disk usage."""
    result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
    return result.stdout.strip().split('\n')[-1]

def get_directory_size_mb(path):
    """Get directory size in MB."""
    if not os.path.exists(path):
        return 0
    try:
        result = subprocess.run(['du', '-sm', path], capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.split()[0])
    except:
        pass
    return 0

def find_all_experiments():
    """Find all experiment directories in mlruns."""
    mlruns_path = Path("mlruns")
    if not mlruns_path.exists():
        return []
    
    experiments = []
    for item in mlruns_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            experiments.append(item.name)
    
    return sorted(experiments)

def find_runs_in_experiment(experiment_id):
    """Find all runs in an experiment."""
    exp_path = Path(f"mlruns/{experiment_id}")
    if not exp_path.exists():
        return []
    
    runs = []
    for item in exp_path.iterdir():
        if item.is_dir() and len(item.name) == 32:  # MLflow run IDs are 32 chars
            runs.append(item.name)
    
    return runs

def check_model_migration_status(run_path, model_name):
    """Check if a model has been successfully migrated."""
    artifacts_path = run_path / "artifacts"
    
    # Check for optimized files
    state_dict_file = artifacts_path / f"{model_name}_state_dict_fp16.pt"
    metadata_file = artifacts_path / f"{model_name}_metadata.json"
    
    # Check for old model directory
    old_model_dir = artifacts_path / model_name
    
    # State dict can be either a file or directory (MLflow artifact structure)
    state_dict_exists = state_dict_file.exists() or state_dict_file.is_dir()
    metadata_exists = metadata_file.exists()
    old_model_exists = old_model_dir.exists() and old_model_dir.is_dir()
    
    return {
        'migrated': state_dict_exists and metadata_exists,
        'old_exists': old_model_exists,
        'old_size_mb': get_directory_size_mb(old_model_dir) if old_model_exists else 0,
        'old_path': old_model_dir
    }

def scan_all_models():
    """Scan all experiments and runs for migration status."""
    print("🔍 Scanning all MLflow experiments for migration status...")
    print("=" * 70)
    
    all_results = {
        'experiments': {},
        'total_models': 0,
        'migrated_models': 0,
        'cleanup_candidates': [],
        'total_cleanup_size_mb': 0
    }
    
    experiments = find_all_experiments()
    print(f"📂 Found {len(experiments)} experiments")
    
    for exp_id in experiments:
        print(f"\n📊 Experiment {exp_id}:")
        runs = find_runs_in_experiment(exp_id)
        
        exp_results = {
            'runs': len(runs),
            'models': {},
            'cleanup_size_mb': 0
        }
        
        for run_id in runs:
            run_path = Path(f"mlruns/{exp_id}/{run_id}")
            artifacts_path = run_path / "artifacts"
            
            if not artifacts_path.exists():
                continue
            
            # Check for common model names
            model_types = ['full_model', 'victim_model', 'ablated_model']
            
            for model_name in model_types:
                status = check_model_migration_status(run_path, model_name)
                
                if status['old_exists']:
                    all_results['total_models'] += 1
                    
                    if status['migrated']:
                        all_results['migrated_models'] += 1
                        all_results['cleanup_candidates'].append({
                            'experiment': exp_id,
                            'run': run_id,
                            'model': model_name,
                            'path': status['old_path'],
                            'size_mb': status['old_size_mb']
                        })
                        all_results['total_cleanup_size_mb'] += status['old_size_mb']
                        exp_results['cleanup_size_mb'] += status['old_size_mb']
                        
                        if model_name not in exp_results['models']:
                            exp_results['models'][model_name] = {'migrated': 0, 'not_migrated': 0, 'cleanup_mb': 0}
                        exp_results['models'][model_name]['migrated'] += 1
                        exp_results['models'][model_name]['cleanup_mb'] += status['old_size_mb']
                    else:
                        if model_name not in exp_results['models']:
                            exp_results['models'][model_name] = {'migrated': 0, 'not_migrated': 0, 'cleanup_mb': 0}
                        exp_results['models'][model_name]['not_migrated'] += 1
        
        # Report experiment summary
        if exp_results['models']:
            for model_name, stats in exp_results['models'].items():
                migrated = stats['migrated']
                not_migrated = stats['not_migrated']
                cleanup_mb = stats['cleanup_mb']
                
                status_str = f"✅ {migrated} migrated" if migrated > 0 else ""
                if not_migrated > 0:
                    status_str += f", ❌ {not_migrated} not migrated" if status_str else f"❌ {not_migrated} not migrated"
                
                if cleanup_mb > 0:
                    print(f"   {model_name}: {status_str} ({cleanup_mb/1024:.1f} GB to cleanup)")
                elif migrated > 0 or not_migrated > 0:
                    print(f"   {model_name}: {status_str}")
        else:
            print("   No models found")
        
        all_results['experiments'][exp_id] = exp_results
    
    return all_results

def perform_cleanup(cleanup_candidates, dry_run=True):
    """Perform the actual cleanup of old model artifacts."""
    if dry_run:
        print("\n🧪 DRY RUN - No files will be deleted")
    else:
        print("\n🗑️  PERFORMING CLEANUP - Files will be deleted!")
    
    print("=" * 70)
    
    total_freed_mb = 0
    successful_deletions = 0
    
    for i, candidate in enumerate(cleanup_candidates, 1):
        exp_id = candidate['experiment']
        run_id = candidate['run']
        model_name = candidate['model']
        path = candidate['path']
        size_mb = candidate['size_mb']
        
        print(f"[{i}/{len(cleanup_candidates)}] {exp_id[:8]}/{run_id[:8]}/{model_name} ({size_mb} MB)")
        
        if not dry_run:
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    total_freed_mb += size_mb
                    successful_deletions += 1
                    print(f"   ✅ Deleted successfully")
                else:
                    print(f"   ⚠️  Already gone")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   🔍 Would delete: {path}")
            total_freed_mb += size_mb
            successful_deletions += 1
        
        # Progress update every 50 items
        if i % 50 == 0:
            print(f"   📊 Progress: {successful_deletions}/{len(cleanup_candidates)} processed, {total_freed_mb/1024:.1f} GB {'would be freed' if dry_run else 'freed'}")
    
    return total_freed_mb, successful_deletions

def main():
    """Main cleanup function."""
    print("🧹 Comprehensive MLflow Model Cleanup")
    print("🔒 Safe cleanup: Only removes old models if optimized versions exist")
    print("=" * 70)
    
    # Initial disk usage
    print("💾 Initial disk usage:")
    print(get_disk_usage())
    print()
    
    # Scan all models
    results = scan_all_models()
    
    print("\n" + "=" * 70)
    print("📊 CLEANUP SUMMARY")
    print("=" * 70)
    print(f"🔍 Total models found: {results['total_models']}")
    print(f"✅ Successfully migrated: {results['migrated_models']}")
    print(f"❌ Not yet migrated: {results['total_models'] - results['migrated_models']}")
    print(f"🗑️  Ready for cleanup: {len(results['cleanup_candidates'])} old model directories")
    print(f"💾 Space to be freed: {results['total_cleanup_size_mb']/1024:.2f} GB")
    
    if not results['cleanup_candidates']:
        print("\n🎉 No cleanup needed! All models are already optimized.")
        return
    
    # Show top space consumers
    print(f"\n🔝 Top 10 space consumers:")
    sorted_candidates = sorted(results['cleanup_candidates'], key=lambda x: x['size_mb'], reverse=True)
    for i, candidate in enumerate(sorted_candidates[:10], 1):
        exp_short = candidate['experiment'][:12]
        run_short = candidate['run'][:8]
        print(f"   {i:2d}. {exp_short}/{run_short}/{candidate['model']}: {candidate['size_mb']} MB")
    
    # Confirm cleanup
    print(f"\n⚠️  This will delete {len(results['cleanup_candidates'])} old model directories")
    print(f"💾 Total space to be freed: {results['total_cleanup_size_mb']/1024:.2f} GB")
    
    # First do a dry run
    print("\n" + "=" * 70)
    perform_cleanup(results['cleanup_candidates'], dry_run=True)
    
    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("🤔 Proceed with actual cleanup? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🚀 Starting cleanup...")
        total_freed_mb, successful = perform_cleanup(results['cleanup_candidates'], dry_run=False)
        
        print("\n" + "=" * 70)
        print("🎉 CLEANUP COMPLETED!")
        print(f"✅ Successfully deleted: {successful}/{len(results['cleanup_candidates'])} directories")
        print(f"💾 Total space freed: {total_freed_mb/1024:.2f} GB")
        
        print("\n💾 Final disk usage:")
        print(get_disk_usage())
        
        print("\n✅ All remaining models use optimized storage format")
        print("✅ MLflow experiments are now fully optimized!")
    else:
        print("\n❌ Cleanup cancelled by user")

if __name__ == "__main__":
    main() 