"""Optional Last Step: Small Convolutional Neural Network (CNN) in TensorFlow.

Trains a lightweight 3-layer CNN on segmented fruit images (100x100 resolution):
- Data: Train (2,832), Val (607), Test (608) from dataset/ripeness_splits.csv.
- Architecture:
    Conv2D(32, 3x3) -> MaxPool -> Conv2D(64, 3x3) -> MaxPool -> Conv2D(64, 3x3) -> MaxPool -> GlobalAvgPool -> Dropout -> Dense(64) -> Dense(3, softmax)
- Evaluated on the Test split and metrics appended to results/final/results.md.
"""

import os
import sys
import csv
import time
import numpy as np
from PIL import Image

# Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image
from src.run_final_pipeline import fast_lab_kmeans_segmentation, draw_confusion_matrix_image


def load_dataset_arrays(splits_path: str = "dataset/ripeness_splits.csv", target_size=(100, 100)):
    """Load and prepare segmented fruit image arrays for CNN training."""
    with open(splits_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classes = ["unripe", "ripe", "overripe"]
    label_map = {c: i for i, c in enumerate(classes)}

    splits_data = {"train": ([], []), "val": ([], []), "test": ([], [])}

    print(f"Loading and segmenting {len(rows)} images resized to {target_size} for CNN...")
    t0 = time.time()

    for idx, r in enumerate(rows):
        sp = r["split"]
        p = r["path"]
        stage = r["stage"]
        y = label_map[stage]

        # Load RGB, segment, and mask
        img_rgb = load_image(p)
        mask = fast_lab_kmeans_segmentation(img_rgb)
        mask_3d = (mask > 0)[:, :, np.newaxis]
        fruit_only = np.where(mask_3d, img_rgb, np.zeros_like(img_rgb))

        # Resize to 100x100 for fast CNN training
        pil_img = Image.fromarray(fruit_only).resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(pil_img, dtype=np.uint8)

        splits_data[sp][0].append(arr)
        splits_data[sp][1].append(y)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(rows)} images...")

    print(f"Dataset preparation completed in {time.time() - t0:.1f}s.")

    X_train = np.array(splits_data["train"][0], dtype=np.float32) / 255.0
    y_train = np.array(splits_data["train"][1], dtype=np.int32)

    X_val = np.array(splits_data["val"][0], dtype=np.float32) / 255.0
    y_val = np.array(splits_data["val"][1], dtype=np.int32)

    X_test = np.array(splits_data["test"][0], dtype=np.float32) / 255.0
    y_test = np.array(splits_data["test"][1], dtype=np.int32)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes


def build_small_cnn(input_shape=(100, 100, 3), num_classes=3):
    """Construct a lightweight, modern ConvNet with batch normalization and dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("=== Optional Step: Small CNN Classifier (TensorFlow) ===")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes = load_dataset_arrays()

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    model = build_small_cnn(input_shape=(100, 100, 3), num_classes=3)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    print("\nTraining CNN for up to 8 epochs...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    train_duration = time.time() - t0
    print(f"CNN Training completed in {train_duration:.1f}s.")

    # Evaluate on Test Split
    t0 = time.time()
    test_probs = model.predict(X_test, batch_size=32)
    cnn_latency_ms = (time.time() - t0) / len(X_test) * 1000.0
    y_pred = np.argmax(test_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n--- Small CNN Test Evaluation ---")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Macro Precision: {prec * 100:.2f}%")
    print(f"Macro Recall:    {rec * 100:.2f}%")
    print(f"Macro F1:        {f1 * 100:.2f}%")
    print(f"Latency:         {cnn_latency_ms:.2f} ms/image")

    # Save model and confusion matrix
    model_save_path = "models/ripeness_cnn.keras"
    model.save(model_save_path)
    print(f"Saved CNN model to {model_save_path}")

    cm_path = "results/final/cm_cnn.png"
    draw_confusion_matrix_image(cm, classes, "Small CNN (3-Layer ConvNet) Confusion Matrix", cm_path)

    # Append to results/final/results.md
    cnn_section = f"""

---

## 6. Optional Deep Learning Extension: Small CNN (TensorFlow)

As an exploratory benchmark, a lightweight 3-layer Convolutional Neural Network (Conv2D $\\times$ 3 + GlobalAveragePooling + Dense) was trained directly on the segmented fruit images ($100 \\times 100$ px):

| Model | Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small CNN (TensorFlow)** | 3 Conv Blocks (32/64/64) + GAP + Dropout | **{acc*100:.2f}%** | **{prec*100:.2f}%** | **{rec*100:.2f}%** | **{f1*100:.2f}%** | {cnn_latency_ms:.2f} ms/img |

- **Confusion Matrix:** [cm_cnn.png](file:///{os.path.abspath(cm_path).replace(chr(92), '/')})
- **Comparison to Classical ML:** The classical pipeline (CIE-Lab K-Means + feature extraction + Random Forest / SVM) achieves competitive accuracy (**83.88%** RF vs. **{acc*100:.2f}%** CNN) with faster training time and full interpretability over color and texture distributions.
"""

    with open("results/final/results.md", "a", encoding="utf-8") as f:
        f.write(cnn_section)

    print("Updated results/final/results.md with CNN results!")


if __name__ == "__main__":
    main()
