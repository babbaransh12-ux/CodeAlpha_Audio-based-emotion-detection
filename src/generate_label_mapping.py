# generate_label_mapping.py
"""Utility to create the label mapping JSON required for inference.
It loads the dataset, extracts the string labels, encodes them to integers
using the same logic as in training, and writes the inverse mapping
(index -> label) to `models/label_mapping.json`.
"""
import json
from pathlib import Path

from data_loader import load_dataset
from utils import encode_labels

BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration – must match training settings
DATA_ROOT = BASE_DIR / "Audio Dataset"

def main():
    X, y = load_dataset(str(DATA_ROOT))
    _, mapping = encode_labels(y)  # mapping: index -> label (original order)
    # Invert mapping to index -> label for inference
    inv_mapping = {int(k): v for k, v in mapping.items()}
    out_path = BASE_DIR / "models" / "label_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inv_mapping, f, ensure_ascii=False, indent=2)
    print(f"Label mapping saved to {out_path}")

if __name__ == "__main__":
    main()
