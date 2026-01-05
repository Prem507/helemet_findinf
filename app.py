import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0b1c2d;
}
.stApp {
    background-color: #0b1c2d;
}
h1, h2, h3, p, label {
    color: white !important;
}
.stButton button {
    background-color: #38bdf8;
    color: black;
    border-radius: 8px;
}
.stFileUploader {
    background-color: #112d4e;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🪖 AI-Based Helmet Detection System")
st.write("Upload an **image or video** to detect helmet usage")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- CONFIDENCE SLIDER ----------------
confidence_threshold = st.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# ---------------- INPUT TYPE ----------------
option = st.radio(
    "Select input type",
    ("Image Upload", "Video Upload")
)

# ---------------- IMAGE UPLOAD ----------------
if option == "Image Upload":
    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        results = model(img_array)
        annotated = results[0].plot()

        st.image(
            annotated,
            caption="Detection Result",
            use_container_width=True
        )

        helmet_detected = False

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # ⚠️ CHANGE CLASS ID IF YOUR MODEL DIFFERS
                # Assumption:
                # class 0 -> No Helmet
                # class 1 -> Helmet
                if cls == 1 and conf >= confidence_threshold:
                    helmet_detected = True

        if helmet_detected:
            st.success("✅ Helmet detected")
        else:
            st.error("🚨 ALERT: Helmet NOT detected")

# ---------------- VIDEO UPLOAD ----------------
if option == "Video Upload":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        helmet_found = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls == 1 and conf >= confidence_threshold:
                        helmet_found = True

            stframe.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        os.unlink(tfile.name)

        if helmet_found:
            st.success("✅ Helmet detected in video")
        else:
            st.error("🚨 ALERT: Helmet NOT detected in video")
