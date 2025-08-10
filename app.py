import streamlit as st
import numpy as np
import os
import io
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

st.write("Upload a JPG/PNG image or choose a sample image from the CIFAR-10 test set.")

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

# --- Function to capture model summary ---
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()

# ===== Sidebar =====
st.sidebar.header("📂 Image Input")

# Option to upload or choose demo
input_choice = st.sidebar.radio("Choose input type:", ["Upload Your Image", "Use Demo Image"])

uploaded_file = None
demo_image = None

if input_choice == "Upload Your Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

elif input_choice == "Use Demo Image":
    test_folder = "Test"
    available_classes = [f for f in CLASS_NAMES if os.path.exists(os.path.join(test_folder, f"{f}.jpeg"))]
    selected_class = st.sidebar.selectbox("Select a class", available_classes)
    if selected_class:
        demo_image_path = os.path.join(test_folder, f"{selected_class}.jpeg")
        if os.path.exists(demo_image_path):
            demo_image = Image.open(demo_image_path)

# ===== Main Logic =====
image_to_classify = None

if uploaded_file:
    image_to_classify = Image.open(uploaded_file)
elif demo_image:
    image_to_classify = demo_image

if image_to_classify:
    st.image(image_to_classify, caption="🖼 Selected Image", use_container_width=True)

    with st.spinner("🔍 Classifying... Please wait"):
        x = preprocess_image(image_to_classify)
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

with st.expander("🛠 Model Summary"):
    summary_str = get_model_summary(model)
    st.markdown(
        f"""
        <div style="background-color:#1e1e1e; color:#d4d4d4; padding:15px; 
                    border-radius:8px; overflow-x:auto; font-family:monospace;">
        {summary_str.replace('\n', '<br>')}
        </div>
        """, 
        unsafe_allow_html=True
    )


st.markdown("---")
st.caption("💡 Tip: Match preprocessing to your training setup for accurate results.")