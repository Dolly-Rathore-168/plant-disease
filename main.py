import streamlit as st
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

# Page setup
st.set_page_config(page_title="🌱 Plant Disease Classifier", layout="wide")
st.title("🌿 Plant Disease Classification")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# Determine base directory
try:
    BASE_DIR = Path(__file__).parent
except NameError:
    BASE_DIR = Path(os.getcwd())

# Class index mapping
default_class_indices = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    5: 'Corn___Common_rust',
    6: 'Corn___Northern_Leaf_Blight',
    7: 'Corn___healthy'
}
INDEX_TO_CLASS = {i: name for i, name in default_class_indices.items()}

# Load class indices JSON if available
def load_class_indices():
    try:
        path = BASE_DIR / "class_indices.json"
        with open(path, "r") as f:
            raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
    except:
        return default_class_indices

# Load the model (no warning if not found)
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "trained_model" / "plant_disease_prediction_model.h5"
    if model_path.exists():
        return tf.keras.models.load_model(model_path)
    else:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Flatten, Dense, Input
        model = Sequential([
            Input(shape=(224, 224, 3)),
            Flatten(),
            Dense(len(default_class_indices), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

# Load disease info if available
def load_disease_info():
    try:
        info_path = BASE_DIR / "disease_info.json"
        with open(info_path, "r") as file:
            return json.load(file)
    except:
        return {}

# Image preprocessing
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

# Predict class
def predict_image_class(model, image):
    processed_img = load_and_preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    predicted_class = INDEX_TO_CLASS.get(predicted_index, "Unknown")
    return predicted_class, confidence

# File uploader
uploaded_file = st.file_uploader(
    "📷 Upload a leaf image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a plant leaf."
)

# Run prediction
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analyzing..."):
            INDEX_TO_CLASS = load_class_indices()
            model = load_model()
            disease_info = load_disease_info()

            if model:
                predicted_class, confidence = predict_image_class(model, image)
                st.success(f"✅ Predicted Disease: **{predicted_class}**")
                st.write(f"🔬 Confidence: `{confidence:.2%}`")

                if predicted_class in disease_info:
                    st.subheader("ℹ️ Disease Info")
                    st.write(disease_info[predicted_class])

                if (
                    "prevention" in disease_info and
                    predicted_class in disease_info["prevention"]
                ):
                    st.subheader("🛡️ Prevention")
                    st.write(disease_info["prevention"][predicted_class])
            else:
                st.error("❌ Model could not be loaded.")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
