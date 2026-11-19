"""Fruit Ripeness Detection - Streamlit Web Application

Interactive user interface for:
1. Uploading fruit images.
2. Visualizing color spaces (RGB, HSV, Lab) and color segmentation.
3. Classifying fruit ripeness (Unripe, Ripe, Overripe).
4. Displaying inference speed and model confidence.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Ripeness Detection System")
st.markdown(
    "Identify fruit ripeness stages (**Unripe**, **Ripe**, **Overripe**) "
    "using **Colour Segmentation** and **Machine Learning**."
)

# Sidebar Configuration
st.sidebar.header("Configuration")
selected_fruit = st.sidebar.selectbox(
    "Select Fruit Type",
    options=["Banana", "Mango", "Papaya", "Apple", "Orange", "Tomato"]
)

selected_model = st.sidebar.selectbox(
    "Select Classifier",
    options=["Support Vector Machine (SVM)", "Random Forest", "CNN (Deep Learning)"]
)

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, caption=f"Uploaded {selected_fruit}", use_container_width=True)

    with col2:
        st.subheader("Color Space Conversions")
        hsv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        lab_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

        color_tab1, color_tab2, color_tab3 = st.tabs(["RGB", "HSV", "Lab"])
        with color_tab1:
            st.image(image_np, caption="RGB Color Space", use_container_width=True)
        with color_tab2:
            st.image(hsv_img, caption="HSV Color Space", use_container_width=True)
        with color_tab3:
            st.image(lab_img, caption="CIE-Lab Color Space", use_container_width=True)

    st.info("ℹ️ Models have not been trained yet. Please complete the dataset and training phases to enable live ripeness inference.")
else:
    st.info("👆 Please upload an image using the sidebar to begin analysis.")
