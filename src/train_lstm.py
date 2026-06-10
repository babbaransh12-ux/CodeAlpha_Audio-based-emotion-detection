# train_lstm.py
"""Training script for Speech Emotion Detection using an LSTM model (TensorFlow).

Prerequisites:
- Install dependencies from requirements.txt (`pip install -r requirements.txt`).
- Ensure the `Audio Dataset` folder follows the structure:
  Audio Dataset/
    emotion_label_1/
      *.wav (or .mp3, .flac)
    emotion_label_2/... etc.
  The immediate sub‑folder name is used as the emotion label.

The script will:
1. Load audio files, extract MFCC features.
2. Encode emotion labels.
3. Split data into train/validation sets.
4. Pad sequences to a uniform length.
5. Build and train an LSTM model.
6. Save the best model checkpoint and label mapping.
7. (Optional) Plot training loss & accuracy.
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_loader import load_dataset
from model import build_lstm_model
from utils import encode_labels, split_data, pad_sequences
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------- Configuration --------------------
DATA_ROOT = BASE_DIR / "Audio Dataset"  # relative to project root
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# MFCC parameters (must match data_loader defaults)
SR = 22050
N_MFCC = 40
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 30

# -------------------- Data Loading --------------------
print("Loading dataset…")
X_raw, y_raw = load_dataset(str(DATA_ROOT), sr=SR, n_mfcc=N_MFCC)
print(f"Loaded {len(X_raw)} audio samples.")

# Encode labels
y_int, label_mapping = encode_labels(y_raw)
num_classes = len(label_mapping)
print(f"Detected {num_classes} emotion classes: {list(label_mapping.values())}")

# Train/validation split
X_train_raw, X_val_raw, y_train, y_val = split_data(X_raw, y_int, test_size=TEST_SIZE)

# Pad sequences – determine max length from training data
max_len = max(seq.shape[0] for seq in X_train_raw)
print(f"Padding sequences to length {max_len} timesteps.")
X_train = pad_sequences(X_train_raw, maxlen=max_len)
X_val = pad_sequences(X_val_raw, maxlen=max_len)

# -------------------- Model Setup --------------------
model = build_lstm_model(input_dim=N_MFCC, num_classes=num_classes)
model.summary()

# Callbacks: save best model & early stopping
checkpoint_path = MODEL_DIR / "best_lstm.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(checkpoint_path),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# -------------------- Training --------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb],
    verbose=2,
)

# Save label mapping for inference
label_map_path = MODEL_DIR / "label_mapping.json"
with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)
print(f"Label mapping saved to {label_map_path}")

# -------------------- Plotting --------------------
if history.history:
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = PLOTS_DIR / "training_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plots saved to {plot_path}")

print("Training complete. Best model saved at", checkpoint_path)
