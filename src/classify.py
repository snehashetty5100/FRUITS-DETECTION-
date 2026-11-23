"""Phase 4: Fruit Ripeness Classification Pipeline

Evaluates Machine Learning models on extracted handcrafted color and texture features
to predict fruit ripeness stage (3-class: Unripe, Ripe, Overripe).

Models Evaluated:
1. Majority-Class Baseline (DummyClassifier)
2. k-Nearest Neighbors (kNN, k=3, with StandardScaler)
3. Support Vector Machine (RBF Kernel, C=1.0, with StandardScaler)
4. Random Forest (100 trees, random_state=42)

Evaluation Protocol:
- Stratified 5-Fold Cross Validation.
- Preprocessing (StandardScaler) fitted strictly on training folds to prevent data leakage.
- Metrics: Overall Accuracy, Macro-Averaged F1-Score, Confusion Matrix, and sample size n.
- Ablation Study: Comparison of models trained on Segmented K-Means features vs. Full-Image features
  to quantify the empirical impact of segmentation.
- Explicit warning banner on statistical significance for small datasets (n < 100).
- Pure scikit-learn, NumPy, and standard csv module (zero cv2 / pandas).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RESULTS_DIR = Path("results/phase4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["unripe", "ripe", "overripe"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TARGET_CLASSES)}


def load_feature_dataset(csv_path: Path):
    """Load feature matrix X and target labels y from CSV without pandas."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Features file {csv_path} not found. Run src/features.py first.")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        raise ValueError(f"Feature file {csv_path} is empty.")

    # Identify metadata columns to exclude from feature matrix
    meta_cols = {"filename", "fruit", "stage", "fallback"}
    feature_cols = [col for col in reader[0].keys() if col not in meta_cols]

    X_list = []
    y_list = []
    meta_list = []

    for row in reader:
        # Extract features as float array
        feat_vals = [float(row[col]) for col in feature_cols]
        stage = row["stage"].strip().lower()
        if stage not in CLASS_TO_IDX:
            continue

        X_list.append(feat_vals)
        y_list.append(CLASS_TO_IDX[stage])
        meta_list.append({
            "filename": row.get("filename", ""),
            "fruit": row.get("fruit", ""),
            "stage": stage,
            "fallback": int(row.get("fallback", 0))
        })

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    return X, y, feature_cols, meta_list


def evaluate_model_cv(model, X: np.ndarray, y: np.ndarray, cv, scale_features: bool = False):
    """Perform out-of-fold cross validation with proper preprocessing pipeline."""
    y_true_all = []
    y_pred_all = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Clone and train model on this fold
        clf = model.__class__(**model.get_params())
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        y_true_all.extend(y_val)
        y_pred_all.extend(preds)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return acc, macro_f1, cm


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Format 3x3 confusion matrix as a clear text table."""
    lines = []
    lines.append(f"               {'Pred Unripe':<12} {'Pred Ripe':<12} {'Pred Overripe':<12}")
    for idx, row in enumerate(cm):
        cname = f"True {TARGET_CLASSES[idx].capitalize()}"
        lines.append(f"{cname:<14} {row[0]:<12} {row[1]:<12} {row[2]:<12}")
    return "\n".join(lines)


def run_benchmark():
    print("=" * 80)
    print("PHASE 4: FRUIT RIPENESS CLASSIFICATION & ABLATION BENCHMARK")
    print("=" * 80)
    print("\n********************************************************************************")
    print("WARNING: Dataset contains < 100 images. Performance estimates are indicative only")
    print("and possess wider confidence intervals. No hyperparameters tuned on held-out data.")
    print("********************************************************************************\n")

    masked_csv = RESULTS_DIR / "features.csv"
    full_csv = RESULTS_DIR / "features_full.csv"

    # Load both feature sets
    X_mask, y_mask, feat_names, meta_mask = load_feature_dataset(masked_csv)
    X_full, y_full, _, _ = load_feature_dataset(full_csv)

    n_samples = len(y_mask)
    n_features = len(feat_names)

    class_counts = {c: int(np.sum(y_mask == CLASS_TO_IDX[c])) for c in TARGET_CLASSES}
    print(f"Total Samples (n): {n_samples}")
    print(f"Feature Dimension: {n_features}")
    print(f"Class Distribution: {class_counts}")
    print(f"Cross-Validation: Stratified 5-Fold\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Models configuration
    models = [
        ("Majority Baseline", DummyClassifier(strategy="most_frequent"), False),
        ("k-NN (k=3)", KNeighborsClassifier(n_neighbors=3), True),
        ("SVM (RBF Kernel)", SVC(kernel="rbf", C=1.0, random_state=42), True),
        ("Random Forest (100 trees)", RandomForestClassifier(n_estimators=100, random_state=42), False),
    ]

    results_table = []
    cms = {}

    print("-" * 80)
    print(f"{'Model':<26} | {'Mask Acc':<10} | {'Mask Macro-F1':<14} | {'Full Acc':<10} | {'Full Macro-F1'}")
    print("-" * 80)

    for name, model, scale in models:
        # 1. Segmented K-Means Mask
        acc_m, f1_m, cm_m = evaluate_model_cv(model, X_mask, y_mask, cv, scale_features=scale)
        # 2. Full-Image Ablation
        acc_f, f1_f, cm_f = evaluate_model_cv(model, X_full, y_full, cv, scale_features=scale)

        cms[name] = {"mask": cm_m, "full": cm_f}

        results_table.append({
            "model": name,
            "samples_n": n_samples,
            "mask_accuracy": round(acc_m, 4),
            "mask_macro_f1": round(f1_m, 4),
            "full_accuracy": round(acc_f, 4),
            "full_macro_f1": round(f1_f, 4),
            "segmentation_gain_acc": round(acc_m - acc_f, 4),
            "segmentation_gain_f1": round(f1_m - f1_f, 4),
        })

        print(f"{name:<26} | {acc_m:<10.4f} | {f1_m:<14.4f} | {acc_f:<10.4f} | {f1_f:.4f}")

    print("-" * 80)

    # Export results table to CSV
    res_csv = RESULTS_DIR / "classification_results.csv"
    with open(res_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "samples_n", "mask_accuracy", "mask_macro_f1",
            "full_accuracy", "full_macro_f1", "segmentation_gain_acc", "segmentation_gain_f1"
        ])
        writer.writeheader()
        writer.writerows(results_table)

    print(f"\n[✓] Results table exported to {res_csv}\n")

    # Display Confusion Matrices for Segmented Features
    print("=" * 80)
    print("CONFUSION MATRICES (ON SEGMENTED K-MEANS FEATURES, n=51):")
    print("=" * 80)
    for name, model, _ in models:
        print(f"\n--- Model: {name} ---")
        print(format_confusion_matrix(cms[name]["mask"]))

    # Display Ablation Summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY: SEGMENTED MASK VS FULL-IMAGE CONTENT")
    print("=" * 80)
    for r in results_table:
        gain = r["segmentation_gain_acc"] * 100
        sign = "+" if gain >= 0 else ""
        print(f"  {r['model']:<26}: Mask Acc = {r['mask_accuracy']*100:.1f}%, Full Acc = {r['full_accuracy']*100:.1f}% ({sign}{gain:.1f}% accuracy gain)")

    return results_table


if __name__ == "__main__":
    run_benchmark()
