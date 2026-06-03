import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your custom trained model
model = YOLO('model_bintang_v2.pt')

st.title("🌌 Star & Constellation Detector")
st.write("Upload an image of the night sky to detect your target constellations.")

# Create the file upload button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image OpenCV can read
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Detecting...")
    
    # Run inference
    img_array = np.array(image)
    results = model(img_array, conf=0.01)
    
    # Plot the bounding boxes on the image
    res_plotted = results[0].plot()
    
    # Display the output
    st.image(res_plotted, caption='Detected Constellations', use_column_width=True)