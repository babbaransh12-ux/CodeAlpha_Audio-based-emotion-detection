# Speech Emotion Detection (LSTM)

A machine learning project that detects emotions from speech audio files using a Long Short-Term Memory (LSTM) neural network. The project features a deep learning model trained on audio datasets (such as RAVDESS) and an interactive Flask-based web dashboard for real-time inference and visualization.

## Features

*   **Deep Learning Model:** An LSTM network built with TensorFlow/Keras to analyze sequential audio data.
*   **Audio Processing:** Advanced audio feature extraction (MFCCs, Mel Spectrograms, Waveforms) using `librosa`.
*   **Web Dashboard:** A responsive web application built with Flask allowing users to upload `.wav` files and see instant predictions.
*   **Rich Visualizations:** Dynamically generates Waveform, MFCC, and Spectrogram plots for uploaded audio segments.
*   **Emotion Coverage:** Capable of classifying emotions like Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## Project Structure

*   `src/`: Contains the core machine learning logic.
    *   `data_loader.py`: Scripts for loading and preprocessing raw audio data.
    *   `model.py`: Definition of the LSTM model architecture.
    *   `train_lstm.py`: Script to train the model and save the weights.
    *   `predict_lstm.py`: Command-line script to run predictions.
    *   `utils.py` & `generate_label_mapping.py`: Utility functions and label management.
*   `dashboard/`: Contains the Flask web application.
    *   `app.py`: The main Flask backend server.
    *   `templates/` & `static/`: HTML templates and static assets (CSS/JS) for the UI.
*   `models/`: Stores the trained weights (`.h5` files) and label mappings (`.json`).
*   `Audio Dataset/`: Directory intended for training audio datasets (e.g., RAVDESS).
*   `plots/`: Stores visualizations generated during model training.

## Installation

1.  Clone this repository or download the source code.
2.  Navigate to the project directory:
    ```bash
    cd "Speech Emotion Detection (LSTM)"
    ```
3.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Web Dashboard (Interactive Mode)
To launch the beautiful web interface where you can upload audio files and view predictions:
```bash
python dashboard/app.py
```
Open your web browser and go to `http://localhost:5000`.

### 2. Training the Model
To train the model from scratch on your own dataset, ensure your audio files are correctly structured in the `Audio Dataset/` directory and run:
```bash
python src/train_lstm.py
```

### 3. Command Line Prediction
To predict the emotion of a specific file without the UI:
```bash
python src/predict_lstm.py path/to/your/audio.wav
```

## Technologies Used

*   **Backend & ML:** Python, TensorFlow, Keras, Scikit-learn
*   **Audio Processing:** Librosa
*   **Web Framework:** Flask
*   **Data & Visualization:** NumPy, Pandas, Matplotlib
