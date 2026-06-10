# utils.py
"""Utility functions for the Speech Emotion Detection project.
- label encoding
- train/validation split
- padding helper
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def encode_labels(labels: np.ndarray) -> (np.ndarray, dict):
    """Encode string emotion labels to integer indices.

    Returns the integer array and a mapping dict {index: label}.
    """
    le = LabelEncoder()
    int_labels = le.fit_transform(labels)
    mapping = {i: label for i, label in enumerate(le.classes_)}
    return int_labels, mapping

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """Split features and labels into train and test sets.
    Handles variable‑length sequences by returning list of arrays.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def pad_sequences(batch: np.ndarray, maxlen: int = None):
    """Pad a list/array of variable‑length sequences to uniform length.
    Uses TensorFlow's pad_sequences.
    """
    # Convert object array to list of arrays
    seqs = [seq for seq in batch]
    return tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post', maxlen=maxlen, dtype='float32')
