import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Page config ---
st.set_page_config(page_title="CIFAR10 Image Classifier", page_icon="🤖", layout="centered")

# --- Title ---
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🤖 CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)

# --- About Link ---
st.markdown(
    """
    <div style="text-align:center; margin-bottom:20px;">
        <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank" 
           style="background-color:#4CAF50; color:white; padding:10px 20px; 
                  text-decoration:none; border-radius:8px; font-size:16px;">
           📚 Know more about CIFAR-10 Dataset
        </a>
    </div>
    """, unsafe_allow_html=True
)

st.write("Upload a JPG/PNG image. It will be resized to **32×32** and classified using a trained CNN model.")

# --- Model settings ---
MODEL_PATH = os.path.join("cifar10_model", "cifar10.h5")
INPUT_SIZE = (32, 32)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# --- Load model ---
@st.cache_resource
def load_keras_model(path: str):
    return load_model(path)

model = load_keras_model(MODEL_PATH)

# --- Preprocessing function ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(INPUT_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Classifying... Please wait"):
        x = preprocess_image(image)
        preds = model.predict(x)
        pred_idx = int(np.argmax(preds, axis=-1)[0])
        confidence = float(np.max(preds, axis=-1)[0])

    # --- Result box ---
    st.markdown(
        f"""
        <div style="background-color:#e8f5e9; padding:15px; border-radius:8px; text-align:center;">
            <h3 style="color:#2e7d32;">Prediction: {CLASS_NAMES[pred_idx]}</h3>
            <p style="font-size:18px;">Confidence: <b>{confidence:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True
    )

    # --- Confidence progress bar ---
    st.progress(confidence)

    # --- Probability table ---
    st.subheader("📊 Class Probabilities")
    probs = preds[0]
    order = np.argsort(-probs)
    for idx in order:
        st.write(f"**{CLASS_NAMES[idx]}**: {probs[idx]:.4f}")

st.markdown("---")
st.caption("💡 Tip: Match preprocessing to your training setup for accurate results.")