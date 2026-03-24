import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# Fix torch/OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Page settings
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# Title
st.markdown("<h1 style='text-align: center;'>🪖 Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>YOLOv5 Based AI Safety Monitoring</p>", unsafe_allow_html=True)

# Load model safely
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Sidebar
st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Select Input Type", ["Image", "Video"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.info("Upload image or video to detect helmet usage")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Detection Output")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Info")
    status = st.empty()

# ---------------- IMAGE ----------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

        img_array = np.array(image)

        status.info("Running detection...")

        results = model(img_array, conf=confidence)

        annotated = results[0].plot()

        frame_placeholder.image(annotated, channels="BGR")

        status.success("Detection completed!")

# ---------------- VIDEO ----------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        status.info("Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)

            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
        status.success("Video processing completed!")
