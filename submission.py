import torch
import mne 
from numpy import std
from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd
import glob
from pathlib import Path
import torch.nn as nn
import numpy as np
from scipy import signal
from scipy.ndimage import zoom
import vision_transformer as vits
from timm.utils import get_state_dict, ModelEma

def extract_trials_from_events(events, sfreq):
    """Extract complete trials (target → response pairs) from events."""
    event_array, event_dict = events
    id_to_desc = {v: k for k, v in event_dict.items()}
    
    trials = []
    current_trial = None
    #print(f"  Extracting trials from {len(event_array)} events")
    for event in event_array:
        sample = event[0]
        event_id = event[2]
        desc = str(id_to_desc.get(event_id, ''))
        desc_lower = desc.lower()
        
        # New trial start
        if 'trial' in desc_lower and 'start' in desc_lower:
            if current_trial and current_trial.get('target_sample'):
                trials.append(current_trial)
            current_trial = {
                'trial_start': sample,
                'target_sample': None,
                'response_sample': None,
            }
        
        # Target event
        elif 'target' in desc_lower:
            if current_trial is None:
                current_trial = {'trial_start': sample}
            if current_trial.get('target_sample') is None:
                current_trial['target_sample'] = sample
                current_trial['target_desc'] = desc
                current_trial['target_side'] = 'left' if 'left' in desc_lower else 'right' if 'right' in desc_lower else None
        
        # Response event  
        elif 'button' in desc_lower or 'press' in desc_lower:
            if current_trial and current_trial.get('target_sample'):
                if current_trial.get('response_sample') is None:
                    if sample > current_trial['target_sample']:
                        current_trial['response_sample'] = sample
                        current_trial['response_desc'] = desc
                        current_trial['response_side'] = 'left' if 'left' in desc_lower else 'right' if 'right' in desc_lower else None
    
    # Don't forget last trial
    if current_trial and current_trial.get('target_sample'):
        trials.append(current_trial)
    
    #print(f"  Extracted {len(trials)} complete trials")
    # Calculate RT for each trial
    for trial in trials:
        if trial.get('response_sample') and trial.get('target_sample'):
            trial['rt_from_stimulus'] = (trial['response_sample'] - trial['target_sample']) / sfreq
            trial['has_response'] = True
            if trial.get('target_side') and trial.get('response_side'):
                trial['correct'] = trial['target_side'] == trial['response_side']
            else:
                trial['correct'] = None
        else:
            trial['rt_from_stimulus'] = None
            trial['has_response'] = False
            trial['correct'] = False
    
    return trials

def create_spectrograms(eeg_data, sfreq):
    """
    Create spectrograms from EEG data.
    
    Args:
        eeg_data: numpy array of shape (n_channels, n_samples) - should be (128, 200)
        sfreq: sampling frequency
    
    Returns:
        numpy array of shape (n_channels, 224, 224)
    """
    def create_fixed_window_spectrogram(window_data, fs, target_size=(224, 224), 
                                       frequency_focus=True, max_freq=50, min_freq=0.5):
        window_length = len(window_data)
        
        if window_length >= 200:
            nperseg = 64
        else:
            nperseg = min(32, window_length // 4)
        
        noverlap = nperseg // 2
        nfft = max(128, nperseg)
        
        f, t, Sxx = signal.spectrogram(
            window_data, fs=fs, window='hann', nperseg=nperseg,
            noverlap=noverlap, nfft=nfft, return_onesided=True,
            detrend='constant', scaling='density'
        )
        
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
        
        if frequency_focus:
            freq_mask = (f >= min_freq) & (f <= max_freq)
            if np.sum(freq_mask) > 0:
                Sxx_focused = Sxx_db[freq_mask, :]
            else:
                Sxx_focused = Sxx_db
        else:
            Sxx_focused = Sxx_db
        
        if Sxx_focused.size > 0:
            p1, p99 = np.percentile(Sxx_focused, [1, 99])
            Sxx_clipped = np.clip(Sxx_focused, p1, p99)
            if Sxx_clipped.max() > Sxx_clipped.min():
                Sxx_normalized = (Sxx_clipped - Sxx_clipped.min()) / (Sxx_clipped.max() - Sxx_clipped.min())
            else:
                Sxx_normalized = np.zeros_like(Sxx_clipped)
        else:
            Sxx_normalized = np.zeros((1, 1))
        
        zoom_factors = (target_size[0] / Sxx_normalized.shape[0], 
                       target_size[1] / Sxx_normalized.shape[1])
        Sxx_resized = zoom(Sxx_normalized, zoom_factors, order=1, mode='nearest')
        
        return Sxx_resized
    
    n_channels, n_samples = eeg_data.shape
    spectrograms = np.zeros((n_channels, 224, 224))
    
    for ch_idx in range(n_channels):
        spectrograms[ch_idx] = create_fixed_window_spectrogram(
            eeg_data[ch_idx], fs=sfreq, target_size=(224, 224), frequency_focus=True
        )
    
    return spectrograms

def create_model(model_name='vit_small', num_classes=1, img_size=224, 
                in_chans=128, drop_path_rate=0.1, checkpoint_path=None, 
                device='cuda', use_ema=True):
    """
    Create and load a ViT model ready for testing.
    
    Args:
        model_name: Name of the model architecture ('vit_small', 'vit_base', etc.)
        num_classes: Number of output classes (1 for regression)
        img_size: Input image size (224)
        in_chans: Number of input channels (128)
        drop_path_rate: Drop path rate (0.1)
        checkpoint_path: Path to checkpoint file (.pth)
        device: Device to load model on ('cuda' or 'cpu')
        use_ema: Whether to use EMA weights if available
    
    Returns:
        model: Loaded model ready for inference
    """
    print(f"Creating model: {model_name}")
    
    # Create ViT model
    model = vits.__dict__[model_name](
        num_classes=num_classes,
        img_size=[img_size],
        drop_path_rate=drop_path_rate,
        in_chans=in_chans
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try to get state dict from checkpoint
        state_dict = None
        if use_ema and 'model_ema' in checkpoint and checkpoint['model_ema'] is not None:
            print("  Using EMA model weights")
            state_dict = checkpoint['model_ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > 0:
            print(f"  Warning: {len(missing_keys)} missing keys (first 5): {missing_keys[:5]}")
        if len(unexpected_keys) > 0:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys (first 5): {unexpected_keys[:5]}")
        
        print("  Checkpoint loaded successfully")
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model has {n_parameters:,} trainable parameters")
    
    return model

class Model1(torch.nn.Module):
    """Model 1: Reaction Time Prediction (ContrastChange task)"""
    
    def __init__(self, SFREQ, DEVICE, checkpoint_path=None):
        super(Model1, self).__init__()
        print("Model 1 init")
        self.sfreq = SFREQ
        self.device = DEVICE
        
        # Model configuration (same as training)
        self.model_name = 'vit_small'
        self.num_classes = 1  # Regression
        self.img_size = 224
        self.in_chans = 128
        self.drop_path = 0.1
        
        # Create ViT model using create_model function
        self.vit_model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            img_size=self.img_size,
            in_chans=self.in_chans,
            drop_path_rate=self.drop_path,
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            use_ema=True
        )
        
        # Normalization stats from training
        self.rt_mean = 1.638
        self.rt_std = 0.366
    
    def create_spectrograms_from_eeg(self, data):
        """Convert batch of EEG data to spectrograms"""
        batch_spects = []
        for i in range(data.shape[0]):
            X = data[i].cpu().numpy() if data.is_cuda else data[i].numpy()
            # Remove 129th channel (reference), keep first 128
            spects = create_spectrograms(X[:-1, ...], self.sfreq)
            batch_spects.append(spects)
        return np.stack(batch_spects, 0)
    
    def forward(self, X, rt):
        """
        Forward pass for validation.
        X: Tensor of shape (batch, 129, 200) - raw EEG data
        Returns: Predictions of shape (batch, 1)
        """
        # Create spectrograms
        spects = self.create_spectrograms_from_eeg(X)
        spects_torch = torch.tensor(spects, dtype=torch.float32).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            y_pred = self.vit_model(spects_torch, classify=True)
            
            # Ensure correct shape
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)
            elif y_pred.shape[1] != 1:
                y_pred = y_pred.view(-1, 1)
        y_pred = y_pred * self.rt_std + self.rt_mean
        # calculate score
        # convert tensors to numpy for sklearn rmse
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        rt_np = rt.detach().cpu().numpy().reshape(-1)
        score = rmse(rt_np, y_pred_np) / std(rt_np)
        print(f"rt values: {rt_np}, y_pred values: {y_pred_np}")
        return score

class Model2(torch.nn.Module):
    """Model 2: P-Factor Prediction (all tasks)"""
    
    def __init__(self, SFREQ, DEVICE, checkpoint_path=None):
        super(Model2, self).__init__()
        print("Model 2 init")
        self.sfreq = SFREQ
        self.device = DEVICE
        
        # Model configuration (same as training)
        self.model_name = 'vit_small'
        self.num_classes = 1  # Regression
        self.img_size = 224
        self.in_chans = 128
        self.drop_path = 0.1
        
        # Create ViT model using create_model function
        self.vit_model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            img_size=self.img_size,
            in_chans=self.in_chans,
            drop_path_rate=self.drop_path,
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            use_ema=True
        )
        
        # Normalization stats from training
        # self.pfactor_mean = -0.091
        # self.pfactor_std = 0.944
    
    def create_spectrograms_from_eeg(self, data):
        """Convert batch of EEG data to spectrograms"""
        batch_spects = []
        for i in range(data.shape[0]):
            X = data[i].cpu().numpy() if data.is_cuda else data[i].numpy()
            # Remove 129th channel (reference), keep first 128
            spects = create_spectrograms(X[:-1, ...], self.sfreq)
            batch_spects.append(spects)
        return np.stack(batch_spects, 0)
    
    def forward(self, X, p_factor):
        """
        Forward pass for validation.
        X: Tensor of shape (batch, 129, 200) - raw EEG data
        Returns: Predictions of shape (batch, 1)
        """
        # Create spectrograms
        spects = self.create_spectrograms_from_eeg(X)
        spects_torch = torch.tensor(spects, dtype=torch.float32).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            y_pred = self.vit_model(spects_torch, classify=True)
            
            # Ensure correct shape
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)
            elif y_pred.shape[1] != 1:
                y_pred = y_pred.view(-1, 1)
        
        # score
        # convert tensors to numpy for sklearn rmse
        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        p_factor_np = p_factor.detach().cpu().numpy().reshape(-1)
        p_factor_max = np.max(p_factor_np)
        p_factor_min = np.min(p_factor_np)
        p_factor_std = std(p_factor_np)
        #print(f"y_pred shape: {y_pred_np.shape}, p_factor shape: {p_factor_np.shape}, p_factor values: {p_factor_np}")
        score = rmse(p_factor_np, y_pred_np)
      
        return score



class Submission:
    """Submission class for the EEG challenge"""
    
    def __init__(self, SFREQ, DEVICE, model1_checkpoint=None, model2_checkpoint=None):
        print('='*60)
        print('Submission Pipeline Initialized')
        print('='*60)
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model1_checkpoint = model1_checkpoint
        self.model2_checkpoint = model2_checkpoint
    
    def get_model_challenge_1(self):
        """Get Model 1 for Challenge 1 (Reaction Time Prediction)"""
        print("\nLoading Challenge 1 Model (Reaction Time)...")
        model = Model1(
            self.sfreq, 
            self.device, 
            checkpoint_path=self.model1_checkpoint
        )
        print("Challenge 1 model ready\n")
        return model
    
    def get_model_challenge_2(self):
        """Get Model 2 for Challenge 2 (P-Factor Prediction)"""
        print("\nLoading Challenge 2 Model (P-Factor)...")
        model = Model2(
            self.sfreq, 
            self.device, 
            checkpoint_path=self.model2_checkpoint
        )
        print("Challenge 2 model ready\n")
        return model


# Example usage and testing
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SFREQ = 100
    
    # Paths to your best checkpoints
    MODEL1_CHECKPOINT = "/home/pmi_lab/EEG_challenge/EEG_challenge/finetuning/model1/f0/best_model.pth"
    MODEL2_CHECKPOINT = "/home/pmi_lab/EEG_challenge/EEG_challenge/finetuning/model2/f1/checkpoint_latest.pth"
    
    # Path to data folder with .set files
    DATA_FOLDER = "/home/pmi_lab/EEG_challenge/EEG_challenge/data/R5_mini_L100/sub-NDARAP785CTE"
    PFACTOR_TSV = "/home/pmi_lab/EEG_challenge/EEG_challenge/data/R5_mini_L100/participants.tsv"
    SFREQ = 100

    # Load p-factor data
    pfactor_df = pd.read_csv(PFACTOR_TSV, sep='\t')
    pfactor_dict = dict(zip(pfactor_df['participant_id'], pfactor_df['p_factor']))

    # Get all .set files in DATA_FOLDER (including subfolders)
    set_files = sorted(glob.glob(f"{DATA_FOLDER}/**/*.set", recursive=True))
    if len(set_files) == 0:
        print(f"No .set files found under {DATA_FOLDER}")
    else:
        print(f"Found {len(set_files)} .set files (including subdirectories)")

    # Separate files by task
    model1_files = []  # ContrastChange
    model2_files = []  # Other tasks

    for set_file in set_files:
        filename = Path(set_file).name.lower()
        if 'contrastchange' in filename:
            model1_files.append(set_file)
        else:
            model2_files.append(set_file)

    print(f"Model 1 files (ContrastChange): {len(model1_files)}")
    print(f"Model 2 files (Other tasks): {len(model2_files)}")

    windows = []
    rt_values = []
        
    # Process Model 1 files (with RT extraction)
    for file_idx, set_file in enumerate(model1_files):
        filename = Path(set_file).name
        # print(f"Processing {filename}...")
        
        # Load EEG data
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        
        # Remove reference channel
        # if 'Ch_128' in raw.ch_names:
        #     raw.drop_channels(['Ch_128'])
        
        data = raw.get_data()  # Shape: (128, n_samples)
        n_channels, n_samples = data.shape
        
        # Extract events and trials
        events = mne.events_from_annotations(raw, verbose=False)
        trials = extract_trials_from_events(events, SFREQ)
        
        if len(trials) == 0:
            print(f"  No valid trials found")
            continue
        
        # Extract windows and RT for each trial
        epoch_samples = int(2.0 * SFREQ)  # 2 seconds = 200 samples
        shift_samples = int(0.5 * SFREQ)  # 0.5 second shift after stimulus
        
        for trial in trials:
            if trial['target_sample'] is None or trial['rt_from_stimulus'] is None:
                continue
            
            # Extract 2-second window starting 0.5s after stimulus
            start_idx = trial['target_sample'] + shift_samples
            end_idx = start_idx + epoch_samples
            
            if start_idx < 0 or end_idx > n_samples:
                continue
            
            # Extract window data (128 channels, 200 samples)
            window = data[:, start_idx:end_idx]
            
            windows.append(window)
            
            # Store RT value
            rt_values.append(trial['rt_from_stimulus'])
        #print(f"  Found {len(windows)} windows for file number {file_idx+1}/{len(model1_files)}")
        if len(windows) == 0:
            print(f"  No valid windows extracted")
            continue
        
    # Stack windows and convert to tensor
    windows = np.stack(windows)  # Shape: (n_windows, 129, 200)
    model1_windows_tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    rt_tensor = torch.tensor(rt_values, dtype=torch.float32).to(DEVICE)

    print(f"  Extracted {len(windows)} windows with RT")
    # predictions = model1(model1_windows_tensor)
    # Compare predictions with rt_tensor


    model2_windows = []
    # Process Model 2 files (with p-factor)
    for file_idx, set_file in enumerate(model2_files[:3]):
        filename = Path(set_file).name
        # print(f"Processing {filename}...")
        
        # Extract participant ID
        participant_id = filename.split('_')[0]  # e.g., 'sub-NDARXX123'
        
        # Get p-factor
        pfactor = pfactor_dict.get(participant_id)
        if pfactor is None or np.isnan(pfactor):
            print(f"  Warning: No p-factor found for {participant_id}")
            continue
        
        # Load EEG data
        raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
        
        # # Remove reference channel
        # if 'Ch_128' in raw.ch_names:
        #     raw.drop_channels(['Ch_128'])
        
        data = raw.get_data()  # Shape: (128, n_samples)
        n_channels, n_samples = data.shape
        
        # Extract 2-second windows (200 samples at 100Hz)
        window_size = 200
        stride = 200  # Non-overlapping windows
        
        for start_idx in range(0, n_samples - window_size + 1, stride):
            window = data[:, start_idx:start_idx + window_size]
            model2_windows.append(window)

        if len(model2_windows) == 0:
            print(f"  No windows extracted")
            continue
        
    # Stack windows and convert to tensor
    model2_windows = np.stack(model2_windows)  # Shape: (n_windows, 129, 200)
    model2_windows_tensor = torch.tensor(model2_windows, dtype=torch.float32).to(DEVICE)
    pfactor_tensor = torch.tensor([pfactor] * len(model2_windows), dtype=torch.float32).to(DEVICE)

    print(f"  Extracted {len(model2_windows)} windows with p-factor: {pfactor_tensor}")

    # predictions = model2(windows_tensor)
    # Compare predictions with pfactor_tensor

    print("\n" + "="*60)
    
    # Create submission
    sub = Submission(SFREQ, DEVICE, MODEL1_CHECKPOINT, MODEL2_CHECKPOINT)
    
    # Test Model 1
    print("\nTesting Model 1")
    print("-"*60)
    model1 = sub.get_model_challenge_1()
 
    # test_input = torch.randn(2, 129, 200).to(DEVICE) * 100
    # print(f"Test input shape: {test_input.shape}")

    output1 = model1(model1_windows_tensor, rt_tensor)
    #print(f"Model 1 output shape: {output1.shape}")
    print(f"Model 1 predictions: {output1}")
    
    # Test Model 2
    print("\nTesting Model 2")
    print("-"*60)
    model2 = sub.get_model_challenge_2()

    output2 = model2(model2_windows_tensor, pfactor_tensor)
    #print(f"Model 2 output shape: {output2.shape}")
    print(f"Model 2 predictions: {output2}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
