import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# ================= ADVANCED UI CSS =================
st.markdown("""
<style>

/* -------- GLOBAL -------- */
body, .stApp {
    background: linear-gradient(135deg, #0b1c2d, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* -------- HEADER -------- */
.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #020617);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 48px;
    font-weight: 800;
}
.hero p {
    font-size: 18px;
    opacity: 0.85;
}

/* -------- CARD -------- */
.card {
    background: #020617;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

/* -------- FILE UPLOADER -------- */
.stFileUploader {
    background: #020617;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #1e293b;
}

/* -------- BUTTON -------- */
.stButton button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    padding: 10px 20px;
}

/* -------- ALERT BANNER -------- */
.alert {
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
    margin-top: 20px;
}
.alert-danger {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
}
.alert-success {
    background: linear-gradient(135deg, #064e3b, #22c55e);
}

/* -------- TOAST FIX -------- */
div[data-testid="stToast"] {
    background-color: #38bdf8 !important;
}
div[data-testid="stToast"] * {
    color: black !important;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1>🪖 AI-Based Helmet Detection System</h1>
    <p>
        Real-time computer vision system to detect helmet usage using deep learning (YOLO).
        Designed for smart traffic monitoring and road safety enforcement.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================= INPUT TYPE =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
option = st.radio(
    "Select Input Type",
    ["Image Upload", "Video Upload"],
    horizontal=True
)
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ================= IMAGE UPLOAD ======================
# =====================================================
if option == "Image Upload":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload Traffic Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        results = model(img_array)
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Output", use_container_width=True)

        helmet_detected = False
        no_helmet_detected = False

        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.tolist():
                class_name = model.names[int(cls_id)].lower()

                if "without" in class_name or "no helmet" in class_name:
                    no_helmet_detected = True
                elif "helmet" in class_name:
                    helmet_detected = True

        if no_helmet_detected:
            st.markdown(
                "<div class='alert alert-danger'>🚨 ALERT: Rider without Helmet Detected</div>",
                unsafe_allow_html=True
            )
            st.toast("🚨 Helmet NOT detected!", icon="🚨")

        elif helmet_detected:
            st.markdown(
                "<div class='alert alert-success'>✅ Helmet Detected — Rider is Safe</div>",
                unsafe_allow_html=True
            )
            st.toast("✅ Helmet detected", icon="🪖")

        else:
            st.info("ℹ️ No rider detected")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ================= VIDEO UPLOAD ======================
# =====================================================
if option == "Video Upload":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            stframe.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        os.unlink(tfile.name)

    st.markdown("</div>", unsafe_allow_html=True)
