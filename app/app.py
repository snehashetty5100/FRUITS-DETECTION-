"""Fruit Ripeness Detection & Colour Segmentation Web Application

Interactive Streamlit interface:
1. Upload any fruit photo (or choose from verified test split samples).
2. Performs automated CIE-Lab K-Means (K=3) segmentation with morphological cleanup.
3. Displays Original Photo, Binary Mask, and Segmented Fruit side-by-side.
4. Extracts 34 color moments and circular Hue histogram features.
5. Predicts ripeness stage (Unripe, Ripe, Overripe) with model confidence and class probabilities.

Zero cv2, zero pandas. Uses Pillow, NumPy, SciPy, scikit-learn, and Streamlit.
"""

import os
import sys
import pickle
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image
import streamlit as st

# Setup project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image, rgb_to_lab, rgb_to_hsv
from src.segmentation import isolate_foreground_cluster, apply_morphological_cleanup


@st.cache_resource
def load_trained_model(model_path: str = "models/ripeness_classifier.pkl"):
    """Load the trained classifier, scaler, and metadata."""
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload


def segment_fruit(img_rgb: np.ndarray) -> np.ndarray:
    """Run CIE-Lab K-Means (K=3) segmentation and morphological cleanup."""
    h, w, _ = img_rgb.shape
    feat_img = rgb_to_lab(img_rgb)

    # Subsample for fast centroid clustering
    sub = feat_img[::3, ::3].reshape(-1, 3).astype(np.float32)
    n_sub = len(sub)

    rng = np.random.RandomState(42)
    centroids = [sub[rng.choice(n_sub)]]
    for _ in range(1, 3):
        dists = np.min([np.sum((sub - c) ** 2, axis=1) for c in centroids], axis=0)
        total_d = dists.sum()
        probs = dists / total_d if total_d > 1e-6 else np.ones(n_sub) / n_sub
        centroids.append(sub[rng.choice(n_sub, p=probs)])
    centroids = np.array(centroids, dtype=np.float32)

    sub_labels = np.full(n_sub, -1, dtype=np.int32)
    for _ in range(12):
        dists = np.stack([np.sum((sub - c) ** 2, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(sub_labels, new_labels):
            break
        sub_labels = new_labels
        for i in range(3):
            m = (sub_labels == i)
            if np.any(m):
                centroids[i] = np.mean(sub[m], axis=0)

    # Assign all pixels
    full_pix = feat_img.reshape(-1, 3).astype(np.float32)
    full_dists = np.stack([np.sum((full_pix - c) ** 2, axis=1) for c in centroids], axis=1)
    full_labels = np.argmin(full_dists, axis=1).reshape(h, w)

    raw_mask = isolate_foreground_cluster(full_labels, img_rgb)
    clean_mask = apply_morphological_cleanup(raw_mask)

    ratio = np.mean(clean_mask > 0)
    if ratio < 0.01 or ratio > 0.99:
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        clean_mask[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = 255

    return clean_mask


def extract_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract 34 color features from the masked fruit area."""
    h, w, _ = img_rgb.shape
    mask_bool = mask > 0
    if np.sum(mask_bool) < 25:
        mask_bool = np.zeros((h, w), dtype=bool)
        mask_bool[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)] = True

    img_hsv = rgb_to_hsv(img_rgb)
    img_lab = rgb_to_lab(img_rgb)

    seg_rgb = img_rgb[mask_bool].astype(np.float32)
    seg_hsv = img_hsv[mask_bool].astype(np.float32)
    seg_lab = img_lab[mask_bool].astype(np.float32)

    seg_r_mean, seg_g_mean, seg_b_mean = np.mean(seg_rgb, axis=0)
    seg_r_std, seg_g_std, seg_b_std = np.std(seg_rgb, axis=0)
    seg_h_mean, seg_s_mean, seg_v_mean = np.mean(seg_hsv, axis=0)
    seg_h_std, seg_s_std, seg_v_std = np.std(seg_hsv, axis=0)
    seg_l_mean, seg_a_mean, seg_b_lab_mean = np.mean(seg_lab, axis=0)
    seg_l_std, seg_a_std, seg_b_lab_std = np.std(seg_lab, axis=0)
    seg_hue_hist, _ = np.histogram(seg_hsv[:, 0], bins=16, range=(0, 180), density=True)

    feats = np.array([
        seg_r_mean, seg_r_std, seg_g_mean, seg_g_std, seg_b_mean, seg_b_std,
        seg_h_mean, seg_h_std, seg_s_mean, seg_s_std, seg_v_mean, seg_v_std,
        seg_l_mean, seg_l_std, seg_a_mean, seg_a_std, seg_b_lab_mean, seg_b_lab_std,
    ] + seg_hue_hist.tolist(), dtype=np.float32)

    return feats


def main():
    st.set_page_config(
        page_title="Fruit Ripeness Detection & Segmentation",
        page_icon="🍎",
        layout="wide"
    )

    st.title("🍎 Fruit Ripeness Detection & Colour Segmentation")
    st.markdown(
        "Automated fruit ripeness stage prediction (`Unripe`, `Ripe`, `Overripe`) using "
        "**CIE-Lab K-Means Clustering**, **Morphological Post-Processing**, and **Machine Learning**."
    )

    model_payload = load_trained_model()
    if model_payload is None:
        st.error("Trained model not found at `models/ripeness_classifier.pkl`. Please run the pipeline script first.")
        st.stop()

    model = model_payload["model"]
    scaler = model_payload["scaler"]
    classes = model_payload["classes"]

    # Sidebar: Input source
    st.sidebar.header("Image Source")
    input_choice = st.sidebar.radio(
        "Choose photo source:",
        options=["Upload Image", "Select Pre-loaded Benchmark Sample"]
    )

    img_rgb = None
    caption_name = ""

    if input_choice == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload fruit image (JPG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            with Image.open(uploaded_file) as pil_img:
                if pil_img.mode in ("RGBA", "LA") or ("A" in pil_img.mode):
                    bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                    bg.paste(pil_img.convert("RGB"), mask=pil_img.split()[-1])
                    img_rgb = np.array(bg, dtype=np.uint8)
                else:
                    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            caption_name = uploaded_file.name
    else:
        sample_catalog = [
            ("apple_ripe_032.jpg", "dataset/annotation_queue/apple_ripe_032.jpg", "Apple - Ripe"),
            ("banana_unripe_184.jpg", "dataset/annotation_queue/banana_unripe_184.jpg", "Banana - Unripe"),
            ("banana_overripe_264.png", "dataset/annotation_queue/banana_overripe_264.png", "Banana - Overripe"),
            ("mango_ripe_436.png", "dataset/annotation_queue/mango_ripe_436.png", "Mango - Ripe"),
            ("orange_ripe_096.jpeg", "dataset/annotation_queue/orange_ripe_096.jpeg", "Orange - Ripe"),
            ("tomato_unripe_071.jpg", "dataset/annotation_queue/tomato_unripe_071.jpg", "Tomato - Unripe"),
            ("tomato_overripe_064.jpeg", "dataset/annotation_queue/tomato_overripe_064.jpeg", "Tomato - Overripe"),
        ]
        sample_labels = [label for _, _, label in sample_catalog]
        chosen_idx = st.sidebar.selectbox("Pick sample image:", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
        fname, fpath, flabel = sample_catalog[chosen_idx]
        if os.path.exists(fpath):
            img_rgb = load_image(fpath)
            caption_name = flabel
        else:
            st.sidebar.warning(f"File not found: {fpath}")

    if img_rgb is None:
        st.info("👈 Please upload a fruit photo or select a sample image from the sidebar to begin.")
        st.stop()

    # Step 1: Run Segmentation
    with st.spinner("Segmenting fruit using CIE-Lab K-Means..."):
        mask = segment_fruit(img_rgb)
        mask_3d = (mask > 0)[:, :, np.newaxis]
        fruit_only = np.where(mask_3d, img_rgb, np.zeros_like(img_rgb))

    # Display 3 visual columns
    st.subheader("1. Visual Segmentation Pipeline")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Photo**")
        st.image(img_rgb, use_container_width=True, caption=caption_name)

    with col2:
        st.markdown("**Segmentation Mask (Lab K=3)**")
        st.image(mask, use_container_width=True, caption="Morphologically Cleaned Mask")

    with col3:
        st.markdown("**Segmented Fruit Only**")
        st.image(fruit_only, use_container_width=True, caption="Background Interference Removed")

    # Step 2: Feature Extraction & Classification
    with st.spinner("Extracting color features and predicting ripeness..."):
        features = extract_features(img_rgb, mask)
        feats_scaled = scaler.transform(features.reshape(1, -1))
        pred_id = int(model.predict(feats_scaled)[0])
        pred_stage = classes[pred_id]
        probs = model.predict_proba(feats_scaled)[0]
        confidence = probs[pred_id] * 100.0

    st.subheader("2. Ripeness Prediction & Classification")

    stage_colors = {
        "unripe": ("#2e7d32", "🍏 UNRIPE", "Fruit is firm, green/acidic, and pre-climacteric."),
        "ripe": ("#f57f17", "🍌 RIPE", "Optimal consumption quality, balanced sugar/acid, peak flavor."),
        "overripe": ("#d32f2f", "🍂 OVERRIPE", "Softened tissue, mottled/darkened skin, fermenting sugars.")
    }

    badge_color, badge_text, badge_desc = stage_colors.get(pred_stage.lower(), ("#333", pred_stage.upper(), ""))

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(
            f"""
            <div style="background-color: {badge_color}18; border-left: 6px solid {badge_color}; padding: 18px; border-radius: 6px; margin-bottom: 12px;">
                <div style="font-size: 26px; font-weight: bold; color: {badge_color};">{badge_text}</div>
                <div style="font-size: 16px; margin-top: 6px; color: #444;">Confidence: <b>{confidence:.1f}%</b></div>
                <div style="font-size: 13px; margin-top: 8px; color: #666;">{badge_desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with res_col2:
        st.markdown("**Class Probability Distribution**")
        for cid, cname in enumerate(classes):
            p = float(probs[cid])
            st.write(f"**{cname.capitalize()}**: {p * 100.0:.1f}%")
            st.progress(p)

    # Step 3: Feature Details
    with st.expander("📊 View Extracted Color Features (Inside Segmented Mask)"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.metric("Mean Lightness (L*)", f"{features[12]:.1f}", help="CIE-Lab L* channel (0 = black, 255 = white)")
            st.metric("Mean Red/Green (a*)", f"{features[14]:.1f}", help="CIE-Lab a* channel (<128 = Green, >128 = Red)")
            st.metric("Mean Yellow/Blue (b*)", f"{features[16]:.1f}", help="CIE-Lab b* channel (>128 = Yellow, <128 = Blue)")
        with f_col2:
            st.metric("Mean Hue (H)", f"{features[6]:.1f} / 180", help="HSV Hue channel (0-179)")
            st.metric("Mean Saturation (S)", f"{features[8]:.1f} / 255", help="HSV Saturation channel")
            st.metric("Mean Value (V)", f"{features[10]:.1f} / 255", help="HSV Brightness channel")
        with f_col3:
            st.metric("Mean Red (R)", f"{features[0]:.1f}")
            st.metric("Mean Green (G)", f"{features[2]:.1f}")
            st.metric("Mean Blue (B)", f"{features[4]:.1f}")
            st.metric("Fruit Area Ratio", f"{np.mean(mask > 0) * 100:.1f}%")


if __name__ == "__main__":
    main()
