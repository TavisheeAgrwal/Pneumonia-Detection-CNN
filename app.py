import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

# Debugging output
st.write("Starting Pneumonia Detector app...")

# Download model from Google Drive
model_path = 'final_train_model.h5'
url = 'https://drive.google.com/uc?id=1jEI3pqNJt0G3N5xlwdWMIVurswK8bd17'
if not os.path.exists(model_path):
    try:
        st.write(f"Downloading model from {url}")
        gdown.download(url, model_path, quiet=False)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        st.write(f"Downloaded model size: {file_size:.2f} MB")
        if file_size < 10:
            st.error(f"Downloaded file is too small ({file_size:.2f} MB). Check the Google Drive link.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Load the model
try:
    st.write("Loading model...")
    model = load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title('Pneumonia Detector')

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=['jpg', 'png'])
if uploaded_file:
    st.write("Processing uploaded image...")
    img = image.load_img(uploaded_file, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img)[0][0]
    result = 'Pneumonia' if pred > 0.5 else 'Normal'
    st.image(uploaded_file, caption=f'Prediction: {result} (Confidence: {pred:.2f})')
