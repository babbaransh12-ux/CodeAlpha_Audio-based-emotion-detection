# predict_lstm.py
"""Inference script for the Speech Emotion Detection LSTM model.

Usage:
    python predict_lstm.py --audio path/to/file.wav

The script loads the best checkpoint saved during training (`models/best_lstm.h5`),
uses the saved label mapping (`models/label_mapping.json`), extracts MFCC features
with the same parameters used during training, and prints the predicted emotion.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Configuration – must match training settings
SR = 22050          # Sampling rate used during training
N_MFCC = 40         # Number of MFCC coefficients
MODEL_PATH = BASE_DIR / "models" / "best_lstm.h5"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_mapping.json"
# ---------------------------------------------------------------------------

def extract_mfcc(audio_path: Path) -> np.ndarray:
    """Load an audio file and compute MFCC features.

    Returns a 2‑D array of shape (time_steps, N_MFCC).
    """
    y, _ = librosa.load(str(audio_path), sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    # Transpose to (time, n_mfcc) – same orientation as training
    return mfcc.T

def load_label_mapping() -> dict:
    if not LABEL_MAP_PATH.is_file():
        raise FileNotFoundError(f"Label mapping not found at {LABEL_MAP_PATH}")
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Invert mapping: index -> label
    return {int(k): v for k, v in mapping.items()}

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained LSTM model.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the input audio file (wav, mp3, etc.)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Ensure training has finished.")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Load label mapping
    label_map = load_label_mapping()

    # Extract features
    mfcc = extract_mfcc(audio_path)
    # Model expects batch dimension
    mfcc_batch = np.expand_dims(mfcc, axis=0)  # shape (1, time, n_mfcc)

    # Predict
    predictions = model.predict(mfcc_batch, verbose=0)
    predicted_idx = int(np.argmax(predictions, axis=1)[0])
    predicted_label = label_map.get(predicted_idx, "Unknown")

    print(f"Predicted emotion: {predicted_label} (class {predicted_idx})")

if __name__ == "__main__":
    main()
