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
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ---------- SESSION STATE ----------
if "show_popup" not in st.session_state:
    st.session_state.show_popup = False

# ---------- POPUP FUNCTION ----------
def show_alert_popup():
    st.markdown("""
        <style>
        .popup-box {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #111;
            color: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 25px red;
            z-index: 9999;
            text-align: center;
            font-size: 22px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="popup-box">🚨 ALERT!<br>No Helmet Detected<br><br></div>', unsafe_allow_html=True)

    if st.button("❌ Close Alert"):
        st.session_state.show_popup = False

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center;'>🪖 Helmet Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>YOLOv5 Based AI Safety Monitoring</p>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Select Input Type", ["Image", "Video"])
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.info("Upload image or video to detect helmet usage")

# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Detection Output")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Info")
    status = st.empty()

# ---------- IMAGE ----------
if option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_column_width=True)

        img_array = np.array(image)

        status.info("Running detection...")

        results = model(img_array, conf=confidence)
        annotated = results[0].plot()

        # ALERT LOGIC
        no_helmet_detected = False

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 1:  # ⚠️ change if needed
                no_helmet_detected = True

        frame_placeholder.image(annotated, channels="BGR")

        # Trigger popup
        if no_helmet_detected:
            st.session_state.show_popup = True
            st.audio("https://www.soundjay.com/buttons/beep-01a.mp3")
        else:
            st.success("✅ Helmet detected - Safe")

        status.success("Detection completed!")

# ---------- VIDEO ----------
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

            # ALERT LOGIC
            no_helmet_detected = False

            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 1:
                    no_helmet_detected = True

            frame_placeholder.image(annotated_frame, channels="BGR")

            if no_helmet_detected:
                st.session_state.show_popup = True
                status.error("🚨 No Helmet Detected!")
            else:
                status.success("✅ Safe")

        cap.release()
        status.success("Video processing completed!")

# ---------- SHOW POPUP ----------
if st.session_state.show_popup:
    show_alert_popup()
