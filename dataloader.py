import os
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple
import gc


class EEGSpectrogramFoldManager:
    """
    Manages creation of k-fold datasets from EEG spectrograms and provides
    efficient loading for vision transformers.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "folds", n_folds: int = 4, seed: int = 42):
        """
        Initialize the fold manager.
        
        Args:
            data_dir: Directory containing .npz spectrogram files
            output_dir: Directory to save fold information and data
            n_folds: Number of folds to create
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_folds = n_folds
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
    def create_folds(self, save_metadata: bool = True) -> Dict:
        """
        Create k-fold splits by randomly assigning spectrograms from each file to folds.
        
        Args:
            save_metadata: Whether to save fold metadata to disk
            
        Returns:
            Dictionary containing fold assignments and metadata
        """
        print("Creating fold assignments...")
        
        fold_assignments = {f"fold_{i}": [] for i in range(self.n_folds)}
        file_info = {}
        
        # Process each .npz file
        npz_files = list(self.data_dir.glob("*.npz"))
        print(f"Found {len(npz_files)} .npz files")
        
        for npz_file in npz_files:
            print(f"Processing {npz_file.name}...")
            
            # Load the file to get shape information
            data = np.load(npz_file)
            spectrograms_shape = data['spectrograms'].shape
            n_channels, n_windows = spectrograms_shape[0], spectrograms_shape[1]
            
            # Store file information
            file_info[npz_file.stem] = {
                'filename': npz_file.name,
                'shape': spectrograms_shape,
                'n_channels': n_channels,
                'n_windows': n_windows,
                'total_spectrograms': n_channels * n_windows
            }
            
            # Create list of all (channel, window) combinations
            all_indices = [(ch, win) for ch in range(n_channels) for win in range(n_windows)]
            
            # Randomly shuffle the indices
            random.shuffle(all_indices)
            
            # Calculate spectrograms per fold (approximately equal)
            total_spectrograms = len(all_indices)
            base_per_fold = total_spectrograms // self.n_folds
            remainder = total_spectrograms % self.n_folds
            
            # Assign indices to folds
            start_idx = 0
            for fold_idx in range(self.n_folds):
                # Some folds get one extra spectrogram if there's a remainder
                fold_size = base_per_fold + (1 if fold_idx < remainder else 0)
                fold_indices = all_indices[start_idx:start_idx + fold_size]
                
                # Store assignment
                fold_assignments[f"fold_{fold_idx}"].append({
                    'file': npz_file.stem,
                    'indices': fold_indices,
                    'count': len(fold_indices)
                })
                
                start_idx += fold_size
            
            # Clean up
            del data
            gc.collect()
        
        # Calculate fold statistics
        fold_stats = {}
        for fold_name, assignments in fold_assignments.items():
            total_spectrograms = sum(assignment['count'] for assignment in assignments)
            fold_stats[fold_name] = {
                'total_spectrograms': total_spectrograms,
                'n_files': len(assignments)
            }
        
        metadata = {
            'n_folds': self.n_folds,
            'seed': self.seed,
            'fold_assignments': fold_assignments,
            'file_info': file_info,
            'fold_stats': fold_stats,
            'data_dir': str(self.data_dir)
        }
        
        if save_metadata:
            metadata_file = self.output_dir / "fold_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Fold metadata saved to: {metadata_file}")
        
        # Print fold statistics
        print("\nFold Statistics:")
        print("-" * 50)
        for fold_name, stats in fold_stats.items():
            print(f"{fold_name}: {stats['total_spectrograms']} spectrograms from {stats['n_files']} files")
        
        return metadata
    
    def load_fold_data(self, fold_idx: int, metadata_file: str = None, 
                      flatten_channels: bool = True, normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a specific fold as 2D arrays ready for vision transformer.
        
        Args:
            fold_idx: Fold index to load (0-based)
            metadata_file: Path to metadata file (if None, uses default location)
            flatten_channels: If True, treats each channel-window as separate sample
            normalize: If True, normalizes spectrograms to [0, 1]
            
        Returns:
            Tuple of (X, y) where X is (n_samples, 224, 224) and y is labels
        """
        if metadata_file is None:
            metadata_file = self.output_dir / "fold_metadata.json"
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        fold_name = f"fold_{fold_idx}"
        if fold_name not in metadata['fold_assignments']:
            raise ValueError(f"Fold {fold_idx} not found in metadata")
        
        print(f"Loading {fold_name}...")
        
        all_spectrograms = []
        all_labels = []
        
        # Load data for this fold
        fold_assignments = metadata['fold_assignments'][fold_name]
        
        for assignment in fold_assignments:
            file_stem = assignment['file']
            indices = assignment['indices']
            
            # Load the .npz file
            npz_path = self.data_dir / f"{file_stem}.npz"
            data = np.load(npz_path)
            spectrograms = data['spectrograms']  # Shape: (128, n_windows, 224, 224)
            
            # Extract specified spectrograms
            for ch_idx, win_idx in indices:
                spectrogram = spectrograms[ch_idx, win_idx]  # Shape: (224, 224)
                
                if normalize:
                    # Normalize to [0, 1]
                    spec_min, spec_max = spectrogram.min(), spectrogram.max()
                    if spec_max > spec_min:
                        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
                
                all_spectrograms.append(spectrogram)
                
                # Create label (you can modify this based on your labeling scheme)
                # For now, using filename as label
                all_labels.append(file_stem)
            
            # Clean up
            del data
            gc.collect()
        
        # Convert to numpy arrays
        X = np.array(all_spectrograms, dtype=np.float32)  # Shape: (n_samples, 224, 224)
        y = np.array(all_labels)  # Shape: (n_samples,)
        
        print(f"Loaded {fold_name}: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    
    def create_label_mapping(self, metadata_file: str = None) -> Dict[str, int]:
        """
        Create a mapping from string labels to integer labels.
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            Dictionary mapping label names to integers
        """
        if metadata_file is None:
            metadata_file = self.output_dir / "fold_metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get unique labels from file names
        unique_labels = sorted(list(metadata['file_info'].keys()))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Save label mapping
        label_file = self.output_dir / "label_mapping.json"
        with open(label_file, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"Label mapping saved to: {label_file}")
        print("Labels:", list(label_mapping.keys()))
        
        return label_mapping


# Utility functions for common operations

def load_fold_for_training(fold_manager: EEGSpectrogramFoldManager, 
                          train_folds: List[int], 
                          val_fold: int,
                          normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training and validation data for k-fold cross-validation.
    
    Args:
        fold_manager: EEGSpectrogramFoldManager instance
        train_folds: List of fold indices to use for training
        val_fold: Fold index to use for validation
        normalize: Whether to normalize spectrograms
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    # Load training data
    X_train_list, y_train_list = [], []
    for fold_idx in train_folds:
        X_fold, y_fold = fold_manager.load_fold_data(fold_idx, normalize=normalize)
        X_train_list.append(X_fold)
        y_train_list.append(y_fold)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Load validation data
    X_val, y_val = fold_manager.load_fold_data(val_fold, normalize=normalize)
    
    return X_train, y_train, X_val, y_val


def create_pytorch_datasets(X: np.ndarray, y: np.ndarray, label_mapping: Dict[str, int]):
    """
    Create PyTorch datasets from numpy arrays.
    
    Args:
        X: Spectrograms array
        y: Labels array (string labels)
        label_mapping: Mapping from string labels to integers
        
    Returns:
        PyTorch TensorDataset
    """
    try:
        import torch
        from torch.utils.data import TensorDataset
        
        # Convert string labels to integers
        y_int = np.array([label_mapping[label] for label in y])
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension: (N, 1, 224, 224)
        y_tensor = torch.LongTensor(y_int)
        
        return TensorDataset(X_tensor, y_tensor)
        
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        return None

