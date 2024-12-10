# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained model
model = tf.keras.models.load_model('cifar10_model.h5')

# CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def prepare_image(image):
    # Resize image to 32x32, as required by CIFAR-10 model
    img = load_img(image, target_size=(32, 32))
    
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Add an extra dimension for batch size
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit UI
st.title("CIFAR-10 Image Classifier")

# Upload image using Streamlit's file uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for prediction
    img_array = prepare_image(uploaded_image)
    
    # Predict the image class
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the predicted label
    predicted_label = class_labels[predicted_class_index]
    
    # Show the prediction result
    st.write(f"Prediction: {predicted_label}")
