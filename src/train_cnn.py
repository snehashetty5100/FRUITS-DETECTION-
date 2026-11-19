"""Module 5: Convolutional Neural Network (CNN) Training

This module handles:
1. Constructing a CNN architecture for fruit ripeness classification.
2. Compiling and fitting CNN models with data augmentation and callbacks.
3. Training history logging and checkpoint saving.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 3
) -> tf.keras.Model:
    """Build a lightweight, beginner-friendly Convolutional Neural Network.

    Args:
        input_shape (tuple): Shape of input images (H, W, C). Defaults to (224, 224, 3).
        num_classes (int): Number of output ripeness classes. Defaults to 3.

    Returns:
        tf.keras.Model: Uncompiled Keras model.
    """
    model = models.Sequential([
        # Data Augmentation layer to improve generalization
        layers.RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        layers.RandomRotation(0.1),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense Classifier Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


def compile_cnn(
    model: tf.keras.Model,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """Compile CNN with Adam optimizer and Sparse Categorical Crossentropy.

    Args:
        model (tf.keras.Model): Model architecture.
        learning_rate (float): Initial optimizer learning rate. Defaults to 0.001.

    Returns:
        tf.keras.Model: Compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_cnn(model: tf.keras.Model, filepath: str) -> None:
    """Save trained Keras model to disk.

    Args:
        model (tf.keras.Model): Trained model.
        filepath (str): Target path (e.g. 'models/fruit_ripeness_cnn.keras').
    """
    model.save(filepath)


def load_cnn(filepath: str) -> tf.keras.Model:
    """Load saved Keras model from disk.

    Args:
        filepath (str): Path to saved model file.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    return tf.keras.models.load_model(filepath)
