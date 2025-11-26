import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ============================
# Load Model
# ============================
MODEL_PATH = "sod_model.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 128  # duhet të jetë identik me trajnim

# ============================
# Preprocessing
# ============================
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype("float32") / 255.0
    return img

# ============================
# Predict mask
# ============================
def predict_mask(img):
    inp = np.expand_dims(img, axis=0)
    pred = model.predict(inp)[0]
    pred = (pred > 0.5).astype("float32")
    return pred

# Create overlay
def apply_overlay(image, mask):
    overlay = image.copy()
    overlay = overlay.astype("float32")
    overlay[..., 0] = np.clip(overlay[..., 0] + mask[..., 0] * 0.7, 0, 1)
    return overlay


# STREAMLIT UI

st.title("Salient Object Detection Demo")
st.write("Upload an image and the model will detect the salient object.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    processed = preprocess_image(image)

    # Predict
    pred_mask = predict_mask(processed)
    overlay_img = apply_overlay(processed, pred_mask)

    # Display
    st.subheader("Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Input Image")
        st.image(image)

    with col2:
        st.write("Predicted Mask")
        st.image(pred_mask[..., 0], clamp=True, channels="gray")

    with col3:
        st.write("Overlay")
        st.image(overlay_img, clamp=True)

st.markdown("---")
st.write("Salient Object Detection Demo")
