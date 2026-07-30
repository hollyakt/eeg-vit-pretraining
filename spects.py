
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path
import gc


def create_individual_channel_spectrograms_dynamic(eeg_data, fs, window_size,
                                                   target_size=(224, 224), overlap_percent=0.5, 
                                                   frequency_focus=True, max_freq=50, min_freq=0.5,
                                                   verbose=False):
    """
    
    Returns:
    --------
    dict containing:
        - 'all_spectrograms': array of shape (n_channels, n_windows, 224, 224)
        - 'metadata': processing parameters and information
    """
    
    if verbose:
        print("Dynamic Multi-Window Spectrogram Generation")
        print("=" * 45)
    
    # Ensure correct data orientation
    eeg_data = np.array(eeg_data)
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T  # Transpose to (n_channels, n_samples)
    
    n_channels, n_samples = eeg_data.shape
    duration = n_samples / fs
    
    if verbose:
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Sampling frequency: {fs} Hz")
        print(f"Recording duration: {duration:.2f} seconds")
        print(f"Window overlap: {overlap_percent*100:.0f}%")
    
    # Calculate sliding window parameters
    hop = int(window_size * (1 - overlap_percent))
   
    # # Calculate how many windows we can create
    # pre_event_data_length = event_time
    # post_event_data_length = n_samples - event_time
    
    # Number of complete windows that fit
    n_windows = max(0, (window_size - n_samples) // hop + 1)
    
    # if n_pre_windows == 0 and n_post_windows == 0:
    #     raise ValueError("Not enough data to create any windows with the specified parameters")
    
    # Create spectrograms for each channel
    all_spectrograms = []
    
    for ch_idx in range(n_channels):
        # if verbose and (ch_idx % 10 == 0 or ch_idx < 3):
        #     print(f"\nProcessing Channel {ch_idx + 1}/{n_channels}")
        
        channel_data = eeg_data[ch_idx, :]
        
        # Create post-event spectrograms using sliding windows  
        spectrograms = create_sliding_window_spectrograms_fixed_size(
            channel_data, window_size, hop,
            fs, target_size, frequency_focus, max_freq, min_freq,
            window_type='post-event', verbose=(verbose and ch_idx < 3)
        )
        
        all_spectrograms.append(spectrograms)
        
        # if verbose and ch_idx < 3:
        #     print(f"  Generated {len(pre_spectrograms)} pre-event spectrograms")
        #     print(f"  Generated {len(post_spectrograms)} post-event spectrograms")
    
    # Convert to numpy arrays and pad if necessary
    max_windows = max(len(ch_pre) for ch_pre in all_spectrograms) if all_spectrograms else 0
    
    
    # Pad channels with fewer windows (using zero-padding)
    padded_pre_spectrograms = []
    padded_post_spectrograms = []
    
    for ch_idx in range(n_channels):
        # # Pad pre-event spectrograms
        # pre_specs = all_pre_spectrograms[ch_idx]
        # while len(pre_specs) < max_pre_windows:
        #     pre_specs.append(np.zeros(target_size))
        # padded_pre_spectrograms.append(pre_specs)
        
        # Pad post-event spectrograms
        post_specs = all_spectrograms[ch_idx]
        while len(post_specs) < max_windows:
            post_specs.append(np.zeros(target_size))
        padded_post_spectrograms.append(post_specs)
    
    # Convert to numpy arrays
    # pre_event_spectrograms = np.array(padded_pre_spectrograms) if max_pre_windows > 0 else np.empty((n_channels, 0, *target_size))
    # post_event_spectrograms = np.array(padded_post_spectrograms) if max_post_windows > 0 else np.empty((n_channels, 0, *target_size))
    # all_spectrograms = np.concatenate((pre_event_spectrograms, post_event_spectrograms), axis=1)
    
    # Prepare metadata
    metadata = {
        'n_channels': n_channels,
        'spectrogram_shape': target_size,
        'fs': fs,
        'duration': duration,
        'overlap_percent': overlap_percent,
        'frequency_focus': frequency_focus,
        'frequency_range': (min_freq, max_freq) if frequency_focus else (0, fs/2),
        'total_spectrograms': n_channels * (max_windows),
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    results = {
        'all_spectrograms': all_spectrograms,
        'metadata': metadata
    }
    
    return results

def create_sliding_window_spectrograms_fixed_size(signal_data, window_size, hop_size, fs, 
                                                 target_size, frequency_focus, max_freq, min_freq,
                                                 window_type='', verbose=False):
    """
    Create multiple spectrograms using sliding windows of fixed size
    
    Parameters:
    -----------
    signal_data : array-like
        1D signal data
    window_size : int
        Size of each window in samples (e.g., 1024, 4096)
    hop_size : int  
        Step size between windows in samples
    fs : float
        Sampling frequency
    target_size : tuple
        Target spectrogram size (224, 224)
    frequency_focus : bool
        Whether to focus on EEG frequencies
    max_freq, min_freq : float
        Frequency range when frequency_focus=True
    window_type : str
        'pre-event' or 'post-event' for verbose output
    verbose : bool
        Print detailed info
    
    Returns:
    --------
    list of spectrograms, each of shape target_size
    """
    n_samples = len(signal_data)
    
    if n_samples < window_size:
        if verbose:
            print(f"  Warning: {window_type} data ({n_samples} samples) shorter than window ({window_size})")
        return []
    
    # Calculate window positions
    spectrograms = []
    window_starts = []
    
    start_idx = 0
    while start_idx + window_size <= n_samples:
        window_starts.append(start_idx)
        start_idx += hop_size
    
    if verbose:
        print(f"  {window_type}: Creating {len(window_starts)} windows of {window_size} samples each")
        print(f"  Window positions: {window_starts[:3]}{'...' if len(window_starts) > 3 else ''}")
    
    # Create spectrogram for each window
    for i, start_idx in enumerate(window_starts):
        end_idx = start_idx + window_size
        window_data = signal_data[start_idx:end_idx]
        
        # Create spectrogram for this window
        spectrogram = create_fixed_window_spectrogram(
            window_data, fs, target_size, frequency_focus, max_freq, min_freq
        )
        spectrograms.append(spectrogram)
    
    return spectrograms

def create_fixed_window_spectrogram(window_data, fs, target_size=(224, 224), 
                                   frequency_focus=True, max_freq=50, min_freq=0.5):
    """
    Create a single spectrogram from a fixed-size window of data
    
    Parameters:
    -----------
    window_data : array-like
        1D signal data of fixed length
    fs : float
        Sampling frequency
    target_size : tuple
        Target spectrogram size (224, 224)
    frequency_focus : bool
        Whether to focus on EEG frequencies
    max_freq, min_freq : float
        Frequency range when frequency_focus=True
    
    Returns:
    --------
    ndarray of shape target_size
    """
    window_length = len(window_data)
    
    # Choose STFT parameters based on window length
    if window_length >= 4096:
        nperseg = 512  # Good balance for long windows
    elif window_length >= 2048:
        nperseg = 256  
    elif window_length >= 1024:
        nperseg = 128
    else:
        nperseg = min(64, window_length // 4)
    
    noverlap = nperseg // 2  # 50% overlap for STFT
    nfft = max(512, nperseg)  # Ensure good frequency resolution
    
    # Create spectrogram
    f, t, Sxx = signal.spectrogram(
        window_data,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=True,
        detrend='constant',
        scaling='density'
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
    
    if frequency_focus:
        # Focus on meaningful EEG frequency range
        freq_mask = (f >= min_freq) & (f <= max_freq)
        if np.sum(freq_mask) > 0:
            Sxx_focused = Sxx_db[freq_mask, :]
        else:
            Sxx_focused = Sxx_db
    else:
        Sxx_focused = Sxx_db
    
    # Normalize to [0, 1] range
    p1, p99 = np.percentile(Sxx_focused, [1, 99])
    Sxx_clipped = np.clip(Sxx_focused, p1, p99)
    Sxx_normalized = (Sxx_clipped - Sxx_clipped.min()) / (Sxx_clipped.max() - Sxx_clipped.min() + 1e-8)
    
    # Resize to target size
    zoom_factors = (target_size[0] / Sxx_normalized.shape[0], 
                   target_size[1] / Sxx_normalized.shape[1])
    Sxx_resized = zoom(Sxx_normalized, zoom_factors, order=1, mode='nearest')
    
    return Sxx_resized

def validate_dynamic_spectrograms(results, raw_data, fs, event_time, channels_to_plot=None):
    """
    Validation function for dynamic multi-window spectrograms
    """
    if channels_to_plot is None:
        channels_to_plot = list(range(min(2, results['pre_event_spectrograms'].shape[0])))
    
    specs = results['all_spectrograms']
    metadata = results['metadata']
    
    n_channels_plot = len(channels_to_plot)

    # Show a few windows
    n_show = min(3, specs.shape[1]) if specs.shape[1] > 0 else 0

    total_cols = n_show
    if total_cols == 0:
        print("No spectrograms to display!")
        return
        
    fig, axes = plt.subplots(n_channels_plot, total_cols, figsize=(4*total_cols, 4*n_channels_plot))
    if n_channels_plot == 1:
        axes = axes.reshape(1, -1)
    if total_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, ch_idx in enumerate(channels_to_plot):
        col = 0

        # Plot spectrograms
        for idx in range(n_show):
            if idx < specs.shape[1]:
                im = axes[i, col].imshow(specs[ch_idx, idx], 
                                       aspect='auto', origin='lower', cmap='viridis')
                axes[i, col].set_title(f'Ch{ch_idx}: {idx+1}\n({metadata["pre_event_duration_s"]:.1f}s window)')
                axes[i, col].set_xlabel('Time Frames')
                axes[i, col].set_ylabel('Frequency Bins')
                plt.colorbar(im, ax=axes[i, col], fraction=0.046)
            col += 1
    
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("=== Multi-Window Spectrogram Validation ===")
    print(f"Pre-event: {metadata['n_pre_windows']} windows of {metadata['pre_event_duration_s']:.2f}s each")
    print(f"Post-event: {metadata['n_post_windows']} windows of {metadata['post_event_duration_s']:.2f}s each") 
    print(f"Window overlap: {metadata['overlap_percent']*100:.0f}%")
    print(f"Hop sizes: Pre={metadata['pre_hop_size']} samples, Post={metadata['post_hop_size']} samples")
    print(f"Frequency focus: {metadata['frequency_focus']}")
    if metadata['frequency_focus']:
        print(f"Frequency range: {metadata['frequency_range'][0]}-{metadata['frequency_range'][1]} Hz")
    print(f"Output shapes: Pre={specs.shape}, Post={specs.shape}")
    print(f"Total spectrograms per channel: {metadata['n_pre_windows'] + metadata['n_post_windows']}")
    print(f"Total spectrograms across all channels: {metadata['total_spectrograms']}")

