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

# ================= PREMIUM UI CSS =================
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #0b1c2d, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* HERO */
.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #020617);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 46px;
    font-weight: 800;
}
.hero p {
    font-size: 18px;
    opacity: 0.85;
}

/* CARD */
.card {
    background: #020617;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

/* FILE UPLOADER */
.stFileUploader {
    background: #020617;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #1e293b;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    padding: 10px 20px;
}

/* ALERTS */
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
.alert-info {
    background: linear-gradient(135deg, #0c4a6e, #0284c7);
}

/* TOAST */
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
        Deep Learning–based traffic surveillance system to detect helmet violations
        using YOLO. Designed for Smart City and road safety applications.
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

        st.image(
            annotated,
            caption="Detection Output",
            use_container_width=True
        )

        helmet_count = 0
        no_helmet_count = 0
        rider_detected = False

        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.tolist():
                class_name = model.names[int(cls_id)].lower()

                # rider presence
                if any(x in class_name for x in ["person", "bike", "motorcycle", "rider"]):
                    rider_detected = True

                # helmet logic
                if "without" in class_name or "no helmet" in class_name:
                    no_helmet_count += 1
                    rider_detected = True
                elif "helmet" in class_name:
                    helmet_count += 1
                    rider_detected = True

        # ================= ALERT LOGIC =================
        if no_helmet_count > 0:
            st.markdown(
                f"<div class='alert alert-danger'>🚨 {no_helmet_count} Rider(s) Without Helmet Detected</div>",
                unsafe_allow_html=True
            )
            st.toast("🚨 Helmet violation detected!", icon="🚨")

        elif helmet_count > 0:
            st.markdown(
                f"<div class='alert alert-success'>✅ {helmet_count} Rider(s) Wearing Helmet</div>",
                unsafe_allow_html=True
            )
            st.toast("✅ All riders are safe", icon="🪖")

        elif rider_detected:
            st.markdown(
                "<div class='alert alert-info'>ℹ️ Rider detected, helmet unclear</div>",
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                "<div class='alert alert-info'>ℹ️ No traffic participants detected</div>",
                unsafe_allow_html=True
            )

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
