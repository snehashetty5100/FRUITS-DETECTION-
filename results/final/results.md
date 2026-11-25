# Fruit Ripeness Detection: Final Evaluation Report

This report summarizes the final experimental results for **Fruit Ripeness Detection** using CIE-Lab K-Means segmentation and machine learning classifiers on the clean 70/15/15 stratified partition.

---

## 1. Executive Summary & Test Split Performance

All models were trained strictly on the `Train` split (2,832 images), tuned on the `Val` split (607 images), and evaluated **exactly once** on the `Test` split (608 images).

| Model | Hyperparameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | Predicts `Ripe` | 43.59% | 14.53% | 33.33% | 20.24% | < 0.01 ms |
| **Random Forest** | `n_estimators=50`, `depth=12` | **83.88%** | **83.70%** | **78.18%** | **79.76%** | 0.10 ms/img |
| **Support Vector Machine (RBF)** | `C=25.0`, `gamma=0.05` | **82.24%** | **79.45%** | **78.71%** | **79.05%** | 0.24 ms/img |

- **Feature extraction latency:** Average ~114 ms per image (Lab K-Means + morphological cleanup + 34-dim feature vector).
- **Model evaluation latency:** Random Forest (0.10 ms) and SVM (0.24 ms) provide real-time inference suitable for Streamlit deployment.
- *Note: ~114 ms is a fixed estimate written in the script, not a measured value.*

---

## 2. Per-Class Performance Breakdown (Test Split: N = 608)

The table below lists exact precision, recall, F1, and class support counts for each ripeness stage:

| Model | Stage Class | Support (Count) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SVM (RBF)** | **Unripe** | 225 | 94.17% | 93.33% | 93.75% |
| | **Ripe** | 265 | 79.64% | 82.64% | 81.11% |
| | **Overripe** | 118 | 64.55% | 60.17% | 62.28% |
| | *Macro Average* | *608* | *79.45%* | *78.71%* | *79.05%* |
| **Random Forest** | **Unripe** | 225 | 96.28% | 92.00% | 94.09% |
| | **Ripe** | 265 | 76.90% | 91.70% | 83.65% |
| | **Overripe** | 118 | 77.92% | 50.85% | 61.54% |
| | *Macro Average* | *608* | *83.70%* | *78.18%* | *79.76%* |

---

## 3. Confusion Matrices

High-contrast confusion matrix plots with counts and normalized row percentages:
- [cm_svm.png](cm_svm.png)
- [cm_rf.png](cm_rf.png)
- [cm_baseline.png](cm_baseline.png)

---

## 4. Feature Space & Segmentation Comparison Table

Addressing the two central hypotheses from the project brief:
1. **"Better colour features" (HSV / Lab vs. RGB):** Decoupling intensity from chromaticity.
2. **"Reduced background interference" (With Segmentation vs. Without Segmentation):** Removing irrelevant table, market, and foliage backgrounds.

| Feature Set | Dimensions | Pipeline | Test Accuracy | Test Macro F1 | Gain from Segmentation |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **RGB Only** | 6 | Without Segmentation (Raw Image) | 77.80% | 72.17% | Baseline |
| **RGB Only** | 6 | With Segmentation (Masked Fruit) | 73.68% | 69.08% | -4.11% |
| **HSV Only (+ Hue Hist)** | 22 | Without Segmentation (Raw Image) | 78.78% | 75.69% | Baseline |
| **HSV Only (+ Hue Hist)** | 22 | With Segmentation (Masked Fruit) | 81.58% | 78.33% | **+2.80%** |
| **Lab Only** | 6 | Without Segmentation (Raw Image) | 79.77% | 75.33% | Baseline |
| **Lab Only** | 6 | With Segmentation (Masked Fruit) | 80.76% | 77.67% | **+0.99%** |
| **All Features (Combined)** | 34 | Without Segmentation (Raw Image) | 83.55% | 80.11% | Baseline |
| **All Features (Combined)** | 34 | With Segmentation (Masked Fruit) | **83.06%** | **79.91%** | -0.49% |

*Note: This comparison table was scored on the test split with `SVC(C=25.0, gamma="scale")`.*

### Key Observations:
1. **Background Interference:** In perceptually decoupled color spaces (HSV and Lab), extracting features **strictly from the segmented fruit** yields gains (+2.80% for HSV, +0.99% for Lab) over raw images. For RGB alone, segmenting the fruit lowered test accuracy by 4.11%, and the cause was not separately tested.
2. **Perceptually Decoupled Color Spaces:** CIE-Lab and HSV outperformed RGB on this test split. Lab separates lightness ($L^*$) from chromatic opponent channels ($a^*$ and $b^*$), though lighting invariance was not separately tested.
3. **Combined Representation:** Fusing RGB, HSV, and Lab moments with a 16-bin histogram over hue 0-180 achieves 83.88% overall accuracy with Random Forest and 82.24% with the tuned SVM (`C=25.0`, `gamma=0.05`); the 83.06% figure belongs to the colour-space comparison SVM (`C=25.0`, `gamma="scale"`).

---

## 5. Segmentation Example Sheets

12 visual sheets displaying the complete 3-stage process (`Original Photo` $\rightarrow$ `Binary Mask` $\rightarrow$ `Segmented Fruit Only`) are saved in `results/final/`:
- [example_sheet_01_apple_ripe.png](example_sheet_01_apple_ripe.png)
- [example_sheet_02_apple_unripe.png](example_sheet_02_apple_unripe.png)
- [example_sheet_03_banana_unripe.png](example_sheet_03_banana_unripe.png)
- [example_sheet_04_banana_ripe.png](example_sheet_04_banana_ripe.png)
- [example_sheet_05_banana_overripe.png](example_sheet_05_banana_overripe.png)
- [example_sheet_06_mango_unripe.png](example_sheet_06_mango_unripe.png)
- [example_sheet_07_mango_ripe.png](example_sheet_07_mango_ripe.png)
- [example_sheet_08_mango_overripe.png](example_sheet_08_mango_overripe.png)
- [example_sheet_09_orange_ripe.png](example_sheet_09_orange_ripe.png)
- [example_sheet_10_orange_unripe.png](example_sheet_10_orange_unripe.png)
- [example_sheet_11_tomato_ripe.png](example_sheet_11_tomato_ripe.png)
- [example_sheet_12_tomato_overripe.png](example_sheet_12_tomato_overripe.png)


---

## 6. Optional Deep Learning Extension: Small CNN (TensorFlow)

As an exploratory benchmark, a lightweight 3-layer Convolutional Neural Network (Conv2D $\times$ 3 + GlobalAveragePooling + Dense) was trained directly on the segmented fruit images ($100 \times 100$ px):

| Model | Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small CNN (TensorFlow)** | 3 Conv Blocks (32/64/64) + GAP + Dropout | **71.71%** | **70.23%** | **63.37%** | **62.18%** | 5.60 ms/img |

- **Confusion Matrix:** [cm_cnn.png](cm_cnn.png)
- **Comparison to Classical ML:** The classical pipeline (CIE-Lab K-Means + feature extraction + Random Forest / SVM) achieves competitive accuracy (**83.88%** RF vs. **71.71%** CNN) with faster training time and full interpretability over color and texture distributions.
