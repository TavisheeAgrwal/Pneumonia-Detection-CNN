import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

# --- Page setup ---
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩻", layout="centered")

st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image and let the model predict if it shows signs of **Pneumonia** or is **Normal**.")

# --- Download the model from Google Drive if not present locally ---
MODEL_PATH = "pneumonia_model.keras"

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=10EubXSGpjH7XHvsV0UJO3lS2rIx2Z_MU",
        MODEL_PATH,
        quiet=False
    )

# --- Load model ---
model = load_model(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --- Display uploaded image ---
    st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_column_width=True)
    st.write("🔍 Analyzing image...")

    # --- Preprocess the image ---
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # --- Make prediction ---
    pred = model.predict(img_array)[0][0]
    confidence = float(pred) if pred > 0.5 else 1 - float(pred)
    label = "PNEUMONIA 🫁" if pred > 0.5 else "NORMAL ✅"

    # --- Display results ---
    st.subheader(f"🧠 Prediction: {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # --- Optional styling feedback ---
    if pred > 0.5:
        st.error("⚠️ The model predicts that this X-ray may show signs of Pneumonia.")
    else:
        st.success("✅ The model predicts this X-ray is Normal.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
