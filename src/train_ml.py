"""Module 4: Traditional Machine Learning Training

This module handles:
1. Training Support Vector Machine (SVM) classifiers.
2. Training Random Forest classifiers.
3. Model evaluation (Accuracy, Precision, Recall, Macro F1, Confusion Matrix).
4. Model serialization and deserialization.
"""

from typing import Dict, Any, List
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0
) -> SVC:
    """Train a Support Vector Machine (SVM) classifier.

    Args:
        X_train (np.ndarray): Scaled feature matrix.
        y_train (np.ndarray): Target class labels.
        kernel (str): SVM kernel type ('linear', 'rbf'). Defaults to 'rbf'.
        C (float): Regularization parameter. Defaults to 1.0.

    Returns:
        SVC: Fitted Support Vector Classifier.
    """
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = None
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Args:
        X_train (np.ndarray): Feature matrix.
        y_train (np.ndarray): Target class labels.
        n_estimators (int): Number of decision trees. Defaults to 100.
        max_depth (int, optional): Maximum depth of trees. Defaults to None.

    Returns:
        RandomForestClassifier: Fitted Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str]
) -> Dict[str, Any]:
    """Evaluate classifier performance using standard metrics.

    Args:
        model: Trained classifier (SVM or Random Forest).
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Ground truth labels.
        target_names (list): List of class names.

    Returns:
        dict: Accuracy, Macro F1, confusion matrix, and detailed classification report.
    """
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report
    }


def save_model(model: Any, filepath: str) -> None:
    """Save trained model or scaler to disk.

    Args:
        model: Trained model or preprocessing object.
        filepath (str): Destination file path (.joblib or .pkl).
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """Load saved model or scaler from disk.

    Args:
        filepath (str): Path to saved model file.

    Returns:
        Any: Loaded model object.
    """
    return joblib.load(filepath)
