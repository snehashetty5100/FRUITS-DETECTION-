"""Module 6: Prediction and Inference Pipeline

This module handles:
1. End-to-end inference combining segmentation, feature extraction, and prediction.
2. Inference time measurement.
3. Prediction using both traditional ML (SVM / Random Forest) and CNN models.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np

from src.preprocessing import load_image, resize_image, normalize_image
from src.feature_extraction import extract_color_features


RIPENESS_CLASSES = ["Unripe", "Ripe", "Overripe"]


def measure_inference_time(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Execute a function and measure its elapsed runtime in milliseconds.

    Args:
        fn: Callable function.
        *args, **kwargs: Function arguments.

    Returns:
        tuple: (function_result, elapsed_time_ms)
    """
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms


def predict_with_ml(
    image_rgb: np.ndarray,
    model: Any,
    scaler: Any = None,
    mask: np.ndarray = None
) -> Dict[str, Any]:
    """Predict ripeness stage using a trained traditional ML classifier.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        model: Trained classifier (SVM or Random Forest).
        scaler: Fitted StandardScaler (optional).
        mask (np.ndarray, optional): Foreground fruit mask.

    Returns:
        dict: Predicted class, probabilities, and confidence score.
    """
    # Extract color features
    features_dict = extract_color_features(image_rgb, mask=mask)
    feature_vector = np.array(list(features_dict.values())).reshape(1, -1)

    if scaler is not None:
        feature_vector = scaler.transform(feature_vector)

    predicted_idx = model.predict(feature_vector)[0]
    probabilities = (
        model.predict_proba(feature_vector)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    return {
        "predicted_label": RIPENESS_CLASSES[predicted_idx],
        "predicted_index": int(predicted_idx),
        "confidence": float(np.max(probabilities)) if probabilities is not None else 1.0,
        "probabilities": probabilities.tolist() if probabilities is not None else []
    }


def predict_with_cnn(
    image_rgb: np.ndarray,
    model: Any,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """Predict ripeness stage using a trained CNN model.

    Args:
        image_rgb (np.ndarray): Input RGB image.
        model: Loaded Keras CNN model.
        target_size (tuple): Target input resolution. Defaults to (224, 224).

    Returns:
        dict: Predicted class, probabilities, and confidence score.
    """
    resized = resize_image(image_rgb, target_size=target_size)
    normalized = normalize_image(resized)
    batch = np.expand_dims(normalized, axis=0)

    preds = model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(preds))

    return {
        "predicted_label": RIPENESS_CLASSES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": float(preds[predicted_idx]),
        "probabilities": preds.tolist()
    }
