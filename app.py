import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# Fix torch issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Page config
st.set_page_config(page_title="Helmet Detection", layout="wide")

st.title("🪖 Helmet Detection System")

# Load model safely
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Mode selection
option = st.radio("Select Input Type", ["Image", "Video"])

# ---------------- IMAGE ----------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)

        # Run detection
        results = model(img_array)

        # Draw results
        annotated = results[0].plot()

        st.image(annotated, caption="Detected Output", channels="BGR")

# ---------------- VIDEO ----------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.success("Video uploaded successfully!")

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = model(frame)

            # Draw results
            annotated_frame = results[0].plot()

            # Display
            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        st.success("Processing complete!")
