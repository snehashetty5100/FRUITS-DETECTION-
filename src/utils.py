"""Utility Helper Functions

This module provides helpers for:
1. Plotting and saving confusion matrices.
2. Saving evaluation metrics to CSV.
3. Color visualization helpers.
"""

import os
from typing import List, Dict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None
) -> plt.Figure:
    """Plot and optionally save a styled confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        class_names (list): List of class names.
        title (str): Plot title.
        save_path (str, optional): Destination file path.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel("Actual Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def save_metrics_to_csv(metrics: Dict[str, any], save_path: str) -> None:
    """Export evaluation metrics to a CSV file.

    Args:
        metrics (dict): Key-value pairs of metrics.
        save_path (str): File destination.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
