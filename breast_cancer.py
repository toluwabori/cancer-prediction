import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("DenseNet121_se.h5", compile=False)

# Helper function for image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit app
st.title("Breast Cancer Classification Web App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make predictions
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)

    # Output result based on threshold 
    threshold = 0.5
    result = "Malignant" if prediction[0][0] > threshold else "Benign"
    
    st.write(f"This image is likely to be {result}")

    st.success("Prediction completed!")


