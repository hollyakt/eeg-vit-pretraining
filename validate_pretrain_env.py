"""
Pre-training Environment & Data Validator

Run this before starting pre-training to catch issues early.
"""

import sys
import os
from pathlib import Path
import argparse

def check_python_version():
    """Check Python version."""
    print("\n[1/8] Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  WARNING: Python 3.8+ recommended (you have {}.{})".format(
            version.major, version.minor))
        return False
    
    print("  ✓ Python version OK")
    return True


def check_packages():
    """Check required packages."""
    print("\n[2/8] Checking required packages...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
    }
    
    missing = []
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Install with: pip install " + " ".join(
            [pkg for pkg in missing if pkg != 'sklearn']))
        return False
    
    print("  ✓ All packages found")
    return True


def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print("\n[3/8] Checking PyTorch CUDA support...")
    
    import torch
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        
        print(f"  ✓ CUDA available")
        print(f"    Devices: {device_count}")
        print(f"    Device 0: {device_name}")
        print(f"    Memory: {device_props.total_memory / 1e9:.1f} GB")
        
        return True
    else:
        print("  ⚠️  CUDA NOT available - will use CPU (much slower)")
        print("    Install CUDA-enabled PyTorch for GPU support")
        return False


def check_files(working_dir):
    """Check required source files."""
    print("\n[4/8] Checking required source files...")
    
    working_dir = Path(working_dir)
    
    required_files = {
        'vision_transformer.py': 'ViT architecture',
        'pretrain_vit.py': 'Pre-training script',
        'utils.py': 'Utility functions',
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = working_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename} ({size_kb:.0f} KB)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\n  Missing files: {', '.join(missing)}")
        print(f"  Working directory: {working_dir}")
        return False
    
    print("  ✓ All source files found")
    return True


def check_data(data_dir):
    """Check data directory and files."""
    print("\n[5/8] Checking data files...")
    
    if not data_dir:
        print("  ⚠️  No data directory provided")
        print("    Skipping data validation")
        return True
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"  ✗ Directory not found: {data_dir}")
        return False
    
    print(f"  Found directory: {data_dir}")
    
    npz_files = list(data_dir.glob('*.npz'))
    
    if not npz_files:
        print("  ✗ No .npz files found in directory")
        return False
    
    print(f"  ✓ Found {len(npz_files)} .npz files")
    
    # Check first file
    try:
        import numpy as np
        first_file = npz_files[0]
        data = np.load(first_file)
        
        if 'spectrograms' not in data:
            print(f"  ✗ First file missing 'spectrograms' key")
            print(f"    Available keys: {list(data.keys())}")
            return False
        
        specs = data['spectrograms']
        print(f"  ✓ First file: {first_file.name}")
        print(f"    Shape: {specs.shape}")
        print(f"    dtype: {specs.dtype}")
        print(f"    Min: {specs.min():.6f}, Max: {specs.max():.6f}")
        
        # Validate shape
        if len(specs.shape) == 4:
            n_ch, n_win, h, w = specs.shape
            if n_ch != 128:
                print(f"  ⚠️  WARNING: Expected 128 channels, got {n_ch}")
            if h != 224 or w != 224:
                print(f"  ⚠️  WARNING: Expected 224×224, got {h}×{w}")
        
        # Validate values
        if specs.min() < -0.01 or specs.max() > 1.01:
            print(f"  ⚠️  WARNING: Values outside [0, 1] range")
            print(f"    Data range: [{specs.min():.6f}, {specs.max():.6f}]")
        
        if np.isnan(specs).any():
            print(f"  ✗ Data contains NaN values!")
            return False
        
        if np.isinf(specs).any():
            print(f"  ✗ Data contains Inf values!")
            return False
        
        print("  ✓ Data validation passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading data: {e}")
        return False


def check_disk_space(working_dir):
    """Check available disk space."""
    print("\n[6/8] Checking disk space...")
    
    import shutil
    working_dir = Path(working_dir)
    
    try:
        stat = shutil.disk_usage(working_dir)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Free: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print(f"  ⚠️  WARNING: Less than 10 GB free space")
            return False
        
        print("  ✓ Sufficient disk space")
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking disk space: {e}")
        return False


def check_memory():
    """Check available RAM."""
    print("\n[7/8] Checking system memory...")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"  Total RAM: {total_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")
        print(f"  Usage: {mem.percent}%")
        
        if available_gb < 4:
            print(f"  ⚠️  WARNING: Less than 4 GB available RAM")
            return False
        
        print("  ✓ Sufficient RAM")
        return True
        
    except ImportError:
        print("  (psutil not installed, skipping)")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not check memory: {e}")
        return True


def run_pytorch_test():
    """Test PyTorch with a simple operation."""
    print("\n[8/8] Testing PyTorch...")
    
    try:
        import torch
        
        # Test tensor creation
        x = torch.randn(2, 128, 224, 224)
        print(f"  ✓ Created tensor: {x.shape}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            x_gpu = x.to('cuda')
            print(f"  ✓ Tensor on GPU: {x_gpu.device}")
            
            # Simple operation
            y = x_gpu * 2
            print(f"  ✓ GPU computation works")
        
        print("  ✓ PyTorch works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate pre-training environment and data'
    )
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to spectrogram directory')
    parser.add_argument('--working_dir', type=str, default='.',
                       help='Working directory (where pretrain_vit.py is)')
    
    args = parser.parse_args()
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + "PRE-TRAINING ENVIRONMENT VALIDATOR".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    results = {}
    
    # Run checks
    results['python'] = check_python_version()
    results['packages'] = check_packages()
    results['pytorch'] = run_pytorch_test()
    results['cuda'] = check_pytorch_cuda()
    results['files'] = check_files(args.working_dir)
    results['data'] = check_data(args.data_dir)
    results['disk'] = check_disk_space(args.working_dir)
    results['memory'] = check_memory()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check.upper()}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! Ready to pre-train.")
        print("\nNext step:")
        print("  python pretrain_vit.py --data_dir " + (args.data_dir or "./data/") + \
              " --output_dir ./pretrain_checkpoints/")
    else:
        print("\n✗ Some checks failed. Please fix issues above before pre-training.")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
