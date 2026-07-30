import numpy as np
import scipy.io as sio
from scipy import signal
import warnings
# from meegkit.detrend import detrend


def preprocess_eeg_for_spectrograms(raw_data, ch_names=None, sfreq=500, reference_ch='Cz', 
                                  line_freq=50, low_freq=0.5, high_freq=50, 
                                  artifact_threshold=150,
                                  verbose=False, plot_results=False):
    """
    EEG preprocessing optimized for spectrogram generation
    
    Returns:
    --------
    dict containing preprocessed data optimized for spectrograms
    """
    
    if verbose:
        print("EEG Preprocessing Pipeline for Spectrograms")
        print("=" * 55)
    
    # Convert and orient data
    data = np.array(raw_data)
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    
    if data.shape[0] > data.shape[1]:
        data = data.T
        if verbose:
            print(f"Transposed data to shape: {data.shape}")
    
    n_channels, n_samples = data.shape
    
    # Create channel names
    if ch_names is None:
        ch_names = [f'Ch_{i+1}' for i in range(n_channels)]
    elif len(ch_names) != n_channels:
        if verbose:
            print(f"Warning: Channel names length mismatch, using default names")
        ch_names = [f'Ch_{i+1}' for i in range(n_channels)]
    
    preprocessing_steps = []
    
    # Step 1: Remove reference electrode
    original_ch_names = ch_names.copy()
    if reference_ch in ch_names:
        ref_idx = ch_names.index(reference_ch)
        data = np.delete(data, ref_idx, axis=0)
        ch_names.remove(reference_ch)
        preprocessing_steps.append(f"Removed reference electrode: {reference_ch}")
        if verbose:
            print(f"\n1. Removed reference electrode '{reference_ch}'")
    
    # Step 3: Basic detrending (remove slow drifts)
    if verbose:
        print(f"\n3. Removing linear trends")
    
    for ch in range(data.shape[0]):
        data[ch, :] = signal.detrend(data[ch, :], type='linear')
    
    preprocessing_steps.append("Applied linear detrending")
    
    # Step 4: Conservative high-pass filter (remove DC and very slow drifts)
    if verbose:
        print(f"\n4. Applying high-pass filter at {low_freq} Hz")
    
    data_filtered = apply_conservative_highpass(data, sfreq, low_freq, verbose)
    preprocessing_steps.append(f"Applied high-pass filter: {low_freq} Hz")
    
    # Step 5: Optional notch filter. Only apply when the mains line actually
    # sits inside the retained band AND strictly below Nyquist -- scipy's
    # iirnotch requires 0 < w0 < 1, so a 60 Hz notch at 100 Hz sfreq (Nyquist
    # 50 Hz) would crash. In that case the low-pass at high_freq already removes
    # the mains line, so skipping is correct. (Restores the author's intended
    # `line_freq < high_freq` guard.)
    nyq = sfreq / 2.0
    if line_freq and 0 < line_freq < nyq and line_freq < high_freq:
        if verbose:
            print(f"\n5. Applying notch filter at {line_freq} Hz")
        data_filtered = apply_gentle_notch_filter(data_filtered, sfreq, line_freq, verbose)
        preprocessing_steps.append(f"Applied notch filter at {line_freq} Hz")
    else:
        if verbose:
            print(f"\n5. Skipping notch at {line_freq} Hz "
                  f"(>= Nyquist {nyq:g} Hz or >= low-pass {high_freq} Hz; already removed)")
        preprocessing_steps.append(f"Skipped notch filter at {line_freq} Hz")
    
    # Step 6: Conservative low-pass filter
    if high_freq < sfreq/2:
        if verbose:
            print(f"\n6. Applying low-pass filter at {high_freq} Hz")
        data_filtered = apply_conservative_lowpass(data_filtered, sfreq, high_freq, verbose)
        preprocessing_steps.append(f"Applied low-pass filter: {high_freq} Hz")
    
    # Step 7: Artifact detection (mark but don't remove - let spectrogram handle it)
    if verbose:
        print(f"\n7. Detecting extreme artifacts (threshold: {artifact_threshold} μV)")
    
    artifact_mask, n_artifacts = detect_extreme_artifacts(data_filtered, artifact_threshold, sfreq, verbose)
    preprocessing_steps.append(f"Detected {n_artifacts} extreme artifact segments")
    
    data_final = data_filtered
    
    
    result = {
        'data': data_final,  # Shape: (n_channels, n_samples)
        'ch_names': ch_names,
        'sfreq': sfreq,
        'n_artifacts': n_artifacts,
        'artifact_mask': artifact_mask,
        'preprocessing_info': {
            'steps_applied': preprocessing_steps,
            'original_shape': raw_data.shape if hasattr(raw_data, 'shape') else None,
            'final_shape': data_final.shape,
            'channels_removed': len(original_ch_names) - len(ch_names)
        }
    }
    
    return result


def apply_conservative_highpass(data, sfreq, cutoff, verbose=True):
    """Apply gentle high-pass filter to remove DC and slow drifts"""
    
    # Use 2nd order Butterworth (gentler than 4th order)
    sos = signal.butter(2, cutoff, btype='high', fs=sfreq, output='sos')
    
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = signal.sosfiltfilt(sos, data[ch, :])
    
    if verbose:
        print(f"   Applied 2nd order high-pass filter")
    
    return filtered_data

def apply_gentle_notch_filter(data, sfreq, line_freq, verbose=True):
    """Apply gentler notch filter"""
    
    # Lower Q factor = wider notch but less ringing
    quality_factor = 15  # Reduced from 30
    b, a = signal.iirnotch(line_freq, quality_factor, sfreq)
    
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])
    
    return filtered_data

def apply_conservative_lowpass(data, sfreq, cutoff, verbose=True):
    """Apply gentle low-pass filter"""
    
    # Use 2nd order Butterworth
    sos = signal.butter(2, cutoff, btype='low', fs=sfreq, output='sos')
    
    filtered_data = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered_data[ch, :] = signal.sosfiltfilt(sos, data[ch, :])
    
    if verbose:
        print(f"   Applied 2nd order low-pass filter")
    
    return filtered_data

def detect_extreme_artifacts(data, threshold, sfreq, verbose=True):
    """Detect only extreme artifacts - less aggressive than original"""
    
    # Only flag extremely large values
    artifact_mask = np.abs(data) > threshold
    
    # Require artifacts to persist for at least 10ms to avoid flagging spikes
    min_duration = max(1, int(0.01 * sfreq))  # 10ms
    
    artifact_timepoints = np.any(artifact_mask, axis=0)
    
    # Filter out very brief artifacts
    kernel = np.ones(min_duration)
    artifact_smooth = np.convolve(artifact_timepoints.astype(float), kernel, mode='same')
    persistent_artifacts = artifact_smooth >= min_duration * 0.5
    
    # Count segments
    diff = np.diff(np.concatenate(([False], persistent_artifacts, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    n_artifacts = len(starts)
    
    if verbose:
        total_artifact_samples = np.sum(persistent_artifacts)
        artifact_percentage = (total_artifact_samples / len(artifact_timepoints)) * 100
        print(f"   Found {n_artifacts} persistent artifact segments")
        print(f"   Total artifact time: {total_artifact_samples/sfreq:.2f} seconds ({artifact_percentage:.1f}%)")
    
    return persistent_artifacts, n_artifacts

def plot_preprocessing_comparison(raw_data, processed_data, original_ch_names, final_ch_names, sfreq):
    """Plot comparison focusing on spectrogram-relevant aspects"""
    
    try:
        import matplotlib.pyplot as plt
        
        # Ensure proper orientation
        if raw_data.shape[0] < raw_data.shape[1]:
            raw_data = raw_data.T
        if processed_data.shape[0] > processed_data.shape[1]:
            processed_data = processed_data.T
        
        # Select channels for comparison
        n_plot = min(3, len(final_ch_names))
        
        fig, axes = plt.subplots(n_plot, 3, figsize=(15, 10))
        if n_plot == 1:
            axes = axes.reshape(1, -1)
        
        # Time axis (first 10 seconds)
        max_samples = min(int(10 * sfreq), raw_data.shape[0], processed_data.shape[0])
        time_axis = np.arange(max_samples) / sfreq
        
        for i in range(n_plot):
            ch_name = final_ch_names[i]
            if ch_name in original_ch_names:
                orig_idx = original_ch_names.index(ch_name)
                
                # Time domain comparison
                axes[i, 0].plot(time_axis, raw_data[:max_samples, orig_idx], 'b-', linewidth=0.8, alpha=0.7, label='Original')
                axes[i, 0].plot(time_axis, processed_data[:max_samples, i], 'r-', linewidth=0.8, alpha=0.7, label='Processed')
                axes[i, 0].set_title(f'Time Domain: {ch_name}')
                axes[i, 0].set_ylabel('Amplitude')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                # Frequency domain comparison
                f_orig, psd_orig = signal.welch(raw_data[:, orig_idx], sfreq, nperseg=1024)
                f_proc, psd_proc = signal.welch(processed_data[:, i], sfreq, nperseg=1024)
                
                axes[i, 1].loglog(f_orig, psd_orig, 'b-', alpha=0.7, label='Original')
                axes[i, 1].loglog(f_proc, psd_proc, 'r-', alpha=0.7, label='Processed')
                axes[i, 1].set_title(f'Power Spectral Density: {ch_name}')
                axes[i, 1].set_xlabel('Frequency (Hz)')
                axes[i, 1].set_ylabel('PSD')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_xlim(0.5, sfreq/2)
                
                # Histogram comparison
                axes[i, 2].hist(raw_data[:, orig_idx], bins=50, alpha=0.5, label='Original', density=True)
                axes[i, 2].hist(processed_data[:, i], bins=50, alpha=0.5, label='Processed', density=True)
                axes[i, 2].set_title(f'Amplitude Distribution: {ch_name}')
                axes[i, 2].set_xlabel('Amplitude')
                axes[i, 2].set_ylabel('Density')
                axes[i, 2].legend()
                axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('eeg_preprocessing_for_spectrograms.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   Preprocessing comparison plot saved")
        
    except Exception as e:
        print(f"   Could not create plot: {e}")
