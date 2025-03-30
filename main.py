# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# Loading the Model
model = load_model('Tea_disease.h5', compile=False)


# Name of Classes
CLASS_NAMES = ['gl', 'rr', 'rsm', 'bb']

# Disease Information
disease_info = {
    'gl': """
        **This is a Non-diseased tea leaf**
    """,
    'rr': """
        **Description :** 
        
        Red rust is a common disease of tea plants...
    """,
    'rsm': """
        **Description :** 
        
        Red spider mites are common pests...
    """,
    'bb': """
        **Description :** 
        
        Brown blight is a common disease of tea plants...
    """
}

# Setting Title of App
st.markdown('<p style="font-size:24px;"><b>Tea Disease Detection (A Plant Disease Detection Tool)</b></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:18px;"><b>Upload an image of the plant leaf</b></p>', unsafe_allow_html=True)

st.markdown('<p style="font-size:18px;">Choose an image...</p>', unsafe_allow_html=True)

# Uploading the Plant image
plant_image = st.file_uploader("", type=["jpeg", "jpg", "png"])
submit = st.button('Predict')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)

        # Resizing and normalizing the image
        opencv_image = cv2.resize(opencv_image, (512, 512))
        opencv_image = opencv_image / 255.0  # Normalization

        # Convert image to 4 Dimensions
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Display disease information
        st.markdown(f'<p style="font-size:22px;"><b>This is a tea leaf with {result}</b></p>', unsafe_allow_html=True)
        st.markdown(disease_info[result])
