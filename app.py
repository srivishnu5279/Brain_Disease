import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model("Model.h5")

# Define class dictionary
class_dict = {0: "Glioma Tumor", 1: "Meningioma Tumor", 2: "No Tumor", 3: "Pituitary Tumor"}

st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
           color: white; }
    .stApp { background-color: #0b7fcb; }
    .title { text-align: center;
        font-size: 32px;
        color: #00d4ff;
        font-weight: bold; }
    .subtitle { text-align: center;
        font-size: 20px;
        color: #ddd; }
    .uploaded-img { display: flex; 
        justify-content: center; }
    .stButton>button { background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px; }
    .result-box { padding: 15px;
        background-color: #222;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #00ff88; }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to predict the disease
def predict_disease(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # Ensure RGB format
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_label = class_dict[np.argmax(prediction)]  # Get class with highest probability
    return predicted_label

# Streamlit UI
st.title("🧠 Brain Tumor Detection App")
st.write("Upload a brain MRI image to detect the type of tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption="Uploaded Image", use_container_width=400)
    
    if st.button("Predict"):  # Styled button
        result = predict_disease(image)
        st.markdown(f'<div class="result-box">Prediction: {result}</div>', unsafe_allow_html=True)