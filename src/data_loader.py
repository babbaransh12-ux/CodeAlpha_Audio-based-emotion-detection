# data_loader.py
"""Utilities for loading audio files and extracting features for speech emotion detection.
Assumes dataset structure:
```
Audio Dataset/
    Actor_01/
        calm.wav
        happy.wav
        ...
    Actor_02/ ...
```
Folder (or sub‑folder) names are used as emotion labels.
"""
import os
from pathlib import Path
import librosa
import numpy as np
from typing import List, Tuple

def _load_file(file_path: Path, sr: int = 22050, n_mfcc: int = 40) -> np.ndarray:
    """Load an audio file and compute MFCC features.

    Args:
        file_path: Path to the audio file.
        sr: Target sampling rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        A 2‑D array (time, n_mfcc).
    """
    y, _ = librosa.load(str(file_path), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Transpose to shape (time_steps, n_mfcc)
    return mfcc.T

def load_dataset(root_dir: str, sr: int = 22050, n_mfcc: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Traverse ``root_dir`` (Audio Dataset) and return features + integer labels.

    The immediate sub‑directory name of each audio file is interpreted as the emotion label.

    Returns:
        X: np.ndarray of shape (num_samples, time_steps, n_mfcc)
        y: np.ndarray of integer encoded labels.
    """
    root = Path(root_dir)
    X, y = [], []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = Path(subdir) / f
                # Label is the name of the parent directory that directly contains the file
                label = Path(subdir).name
                mfcc = _load_file(file_path, sr=sr, n_mfcc=n_mfcc)
                X.append(mfcc)
                y.append(label)
    X = np.array(X, dtype=object)  # variable length sequences -> object array
    y = np.array(y)
    return X, y
