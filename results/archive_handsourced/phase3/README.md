# Phase 3: Colour Segmentation & Benchmarking

This directory contains the evaluation methodology, visual overlays, and benchmark logs for the fruit segmentation pipeline.

---

## 1. Segmentation Pipeline Architecture

The Phase 3 segmentation framework uses classical computer vision algorithms implemented using **Pillow**, **NumPy**, and **SciPy** (zero dependency on OpenCV/cv2 or pandas, ensuring complete compatibility with Windows 11 Smart App Control).

### A. Valid Content Bounding Box Extraction
To prevent symmetric letterbox padding (black `[0, 0, 0]` border areas added during $224 \times 224$ aspect-ratio-preserving resizing) from biasing K-Means clustering, border voting, or IoU computations:
- `get_valid_content_bbox(img)` identifies non-black rows and columns.
- All clustering, color thresholding, and border statistics are computed strictly inside the active content rectangle.

### B. K-Means Clustering ($K=3$) with K-Means++
- **Clustering:** Segments valid content pixels into $K=3$ clusters in RGB space.
- **K-Means++ Initialization:** Standard random sampling collapses when backgrounds are predominantly white or uniform (as occurred with `banana_ripe_001`). K-Means++ probabilistic distance-weighted seeding guarantees diverse initial centroids across fruit and background.

### C. Multi-Factor Foreground Cluster Scoring
Traditional border voting fails when fruit touches an edge or when the background is white. We select the foreground cluster $c^*$ that maximizes:

$$\text{Score}(c) = 0.40 \times \text{CenterBias}(c) + 0.45 \times \text{MeanSat}(c) - 0.55 \times \text{BorderRatio}(c)$$

- **$\text{CenterBias}(c)$**: Fraction of cluster $c$ pixels situated within the central content region ($[0.25, 0.75]$ width and height).
- **$\text{MeanSat}(c)$**: Normalized mean HSV saturation ($[0, 1]$), exploiting the property that fruits typically exhibit higher color saturation than neutral plates or tables.
- **$\text{BorderRatio}(c)$**: Fraction of cluster $c$ pixels lying along the 2-pixel perimeter of the valid content bounding box.

### D. Configured HSV Color Space Thresholding
- Images are converted to HSV space using pure NumPy vectorization ($H \in [0, 179]$, $S \in [0, 255]$, $V \in [0, 255]$).
- A centralized configuration dictionary (`HSV_RANGES` in `src/segmentation.py`) defines specific upper and lower bounds for all 6 fruits across all 3 ripeness stages:
  - **Unripe:** Green, olive, and yellow-green hue bands (e.g., $H \in [28, 85]$).
  - **Ripe:** Yellow, orange, and red hue bands (including two-interval wrap-around handling for red: $[0, 10] \cup [165, 179]$).
  - **Overripe:** Brown, amber, and mottled dark-hue bands with calibrated saturation and value bounds.

### E. Morphological Post-Processing Pipeline
To refine raw binary masks:
1. **Morphological Opening** (disk radius $r=3$, 2 iterations): Eliminates isolated background noise pixels and thin artifacts.
2. **Morphological Closing** (disk radius $r=3$, 2 iterations): Fills small surface pinholes and bridges minor contour discontinuities.
3. **Binary Hole Filling** (`scipy.ndimage.binary_fill_holes`): Recovers interior fruit voids caused by bright specular glares.
4. **Largest Connected Component** (`filter_largest_component`): Rejects any remaining detached background regions, retaining the primary fruit body.

---

## 2. Ground-Truth Annotation & Evaluation Protocol

- **Manifest:** All ground-truth masks are tracked in `dataset/ground_truth_masks/manifest.csv`.
- **Strict Verification Gate:** `src/evaluate_segmentation.py` **only** computes metrics on masks where `verified == "yes"`. Auto-generated masks remain marked `verified == "no"` until visually inspected and hand-corrected.
- **Split Partitions:**
  - **Tune Set (7 samples):** Used to calibrate algorithm parameters, centroid weights, and HSV boundaries. `banana_ripe_001` is frozen in the Tune set because it informed the K-Means++ and scoring enhancements.
  - **Held-Out Set (5 samples):** Strictly isolated for objective validation.
  > [!WARNING]
  > The held-out set must **NOT** be inspected or used to tune segmentation thresholds or algorithm hyperparameters.
- **Evaluation Metrics:**
  - **Intersection over Union (IoU / Jaccard Index):** $\frac{|A \cap B|}{|A \cup B|}$
  - **Sørensen–Dice Coefficient:** $\frac{2 |A \cap B|}{|A| + |B|}$
  - Stored in `results/phase3/eval.csv` via the standard `csv` module.

---

## 3. Known Failure Cases & Edge Cases

1. **Low-Contrast Shaded Papaya (`papaya_ripe_002`)**:
   - Deep shadow gradients on dark olive-orange mottled skin have saturation and value values close to shadowed backgrounds. This can lead to under-segmentation when using conservative color thresholds.
2. **Textured and Wood Grain Backgrounds**:
   - Polished wooden tabletops and cutting boards have warm brown-orange hues ($H \in [10, 25]$) matching ripe bananas, mangoes, and oranges. When the wood grain is visible near the center of the frame, color thresholding alone cannot separate the fruit from the surface without morphological filtering.
3. **High-Gloss Specular Highlights**:
   - Bright direct camera flash creates pure white `[255, 255, 255]` reflections on smooth skins (apples, tomatoes). Hole filling resolves closed fruit outlines, but if a highlight touches the outer edge, the contour may fail to close cleanly.
