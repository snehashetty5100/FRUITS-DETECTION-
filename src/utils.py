"""Utility Helper Functions

Pure Python + Matplotlib + Seaborn utilities:
1. Plotting and saving confusion matrices.
2. Saving evaluation metrics to CSV via the csv module.
3. Visualization helpers.
"""

import csv
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
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


def save_metrics_to_csv(metrics: Dict[str, Any], save_path: str) -> None:
    """Export evaluation metrics to a CSV file using standard library csv module.

    Args:
        metrics (dict): Key-value pairs of metrics.
        save_path (str): File destination.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = list(metrics.keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)
