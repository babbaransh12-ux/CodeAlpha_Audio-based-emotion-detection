# model.py
"""TensorFlow LSTM model definition for speech emotion detection.
Assumes input shape: (time_steps, n_mfcc)
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_dim: int, num_classes: int, hidden_units: int = 128, dropout: float = 0.3) -> tf.keras.Model:
    """Construct a simple LSTM model.

    Args:
        input_dim: Number of MFCC coefficients (features per time step).
        num_classes: Number of emotion classes.
        hidden_units: LSTM hidden state size.
        dropout: Dropout rate after LSTM.
    Returns:
        Compiled Keras model.
    """
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=(None, input_dim)),
        layers.LSTM(hidden_units, dropout=dropout),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
