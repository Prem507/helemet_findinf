import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile
import cv2
import os
import time
from collections import defaultdict

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ENHANCED UI CSS =================
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #0b1c2d, #020617);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* SIDEBAR */
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
    color: #38bdf8;
}

/* HERO */
.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #020617);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    margin-bottom: 30px;
    text-align: center;
}
.hero h1 {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-size: 18px;
    opacity: 0.85;
}

/* CARD */
.card {
    background: rgba(2, 6, 23, 0.8);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(30, 41, 59, 0.5);
}

/* STATS CARD */
.stats-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}
.stats-number {
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
}
.stats-label {
    font-size: 14px;
    opacity: 0.8;
}

/* FILE UPLOADER */
.stFileUploader {
    background: rgba(2, 6, 23, 0.8);
    padding: 18px;
    border-radius: 14px;
    border: 2px dashed #1e293b;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 24px;
    border: none;
    transition: all 0.3s ease;
}
.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(56, 189, 248, 0.4);
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
.alert-warning {
    background: linear-gradient(135deg, #92400e, #f59e0b);
}

/* TOAST */
div[data-testid="stToast"] {
    background-color: #38bdf8 !important;
}
div[data-testid="stToast"] * {
    color: black !important;
    font-weight: 800;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(2, 6, 23, 0.8);
    border-radius: 12px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: white;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9);
    color: black;
}

/* SLIDER */
.stSlider > div > div > div > div {
    background: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🪖 Detection Settings")
    st.markdown("---")

    # Model Info
    st.subheader("Model Information")
    st.info("Using YOLOv5 model trained for helmet detection")

    # Confidence Threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence for detections"
    )

    # Model Classes (assuming standard classes)
    st.subheader("Detection Classes")
    st.markdown("""
    - 🪖 **Helmet**: Rider with helmet
    - 🚫 **No Helmet**: Rider without helmet
    - 🏍️ **Rider**: Motorcycle rider
    - 🚗 **Vehicle**: Other vehicles
    """)

    st.markdown("---")
    st.caption("Built with Streamlit & YOLO")

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
    <h1>🪖 Advanced AI Helmet Detection System</h1>
    <p>
        State-of-the-art deep learning traffic surveillance system using YOLOv5 for real-time helmet violation detection.
        Perfect for Smart City applications and road safety enforcement.
    </p>
</div>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================= MAIN CONTENT =================
tab1, tab2 = st.tabs(["📸 Image Detection", "🎥 Video Detection"])

# =====================================================
# ================= IMAGE DETECTION ===================
# =====================================================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload Traffic Image")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a traffic image for helmet detection"
    )

    if uploaded_image:
        # Load and process image
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        # Progress bar for processing
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Processing image...")
        progress_bar.progress(50)

        # Run detection
        results = model(img_array, conf=conf_threshold)
        annotated = results[0].plot()

        progress_bar.progress(100)
        status_text.text("Detection complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                annotated,
                caption="Detection Results",
                use_container_width=True
            )

        with col2:
            # Statistics
            st.markdown("### 📊 Detection Statistics")

            # Count detections with improved logic
            helmet_count = 0
            no_helmet_count = 0
            rider_count = 0

            if results[0].boxes is not None:
                for cls_id in results[0].boxes.cls.tolist():
                    class_name = model.names[int(cls_id)].lower()

                    # Improved helmet detection logic
                    if "helmet" in class_name and "no" not in class_name and "without" not in class_name:
                        helmet_count += 1
                        rider_count += 1
                    elif "no helmet" in class_name or "without helmet" in class_name or "no-helmet" in class_name:
                        no_helmet_count += 1
                        rider_count += 1
                    elif any(x in class_name for x in ["rider", "person", "motorcycle", "bike"]):
                        rider_count += 1

            # Display stats in cards
            if rider_count > 0 or helmet_count > 0 or no_helmet_count > 0:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-number">{helmet_count}</div>
                        <div class="stats-label">With Helmet</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-number">{no_helmet_count}</div>
                        <div class="stats-label">Without Helmet</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_c:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-number">{rider_count}</div>
                        <div class="stats-label">Total Riders</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Alert logic
                if no_helmet_count > 0:
                    st.markdown(
                        f"<div class='alert alert-danger'>🚨 ALERT: {no_helmet_count} rider(s) detected without helmet!</div>",
                        unsafe_allow_html=True
                    )
                    st.toast("🚨 Helmet violation detected!", icon="🚨")
                elif helmet_count > 0:
                    st.markdown(
                        f"<div class='alert alert-success'>✅ All {helmet_count} rider(s) are wearing helmets safely.</div>",
                        unsafe_allow_html=True
                    )
                    st.toast("✅ Safe riding detected!", icon="🪖")
                else:
                    st.markdown(
                        "<div class='alert alert-info'>ℹ️ Riders detected but helmet status unclear.</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    "<div class='alert alert-info'>ℹ️ No traffic participants detected in the image.</div>",
                    unsafe_allow_html=True
                )

            # Download button
            st.markdown("### 💾 Export Results")
            if st.button("Download Annotated Image"):
                # Save annotated image temporarily
                temp_img = Image.fromarray(annotated)
                temp_path = "annotated_image.png"
                temp_img.save(temp_path)

                with open(temp_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Image",
                        data=file,
                        file_name="helmet_detection_result.png",
                        mime="image/png"
                    )
                os.remove(temp_path)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ================= VIDEO DETECTION ===================
# =====================================================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload Traffic Video")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        help="Upload a traffic video for helmet detection analysis"
    )

    if uploaded_video:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        tfile.close()

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        st.info(f"Video loaded: {total_frames} frames at {fps:.1f} FPS")

        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_every_n_frames = st.slider(
                "Process every Nth frame",
                min_value=1,
                max_value=30,
                value=10,
                help="Process every Nth frame to speed up analysis"
            )
        with col2:
            save_annotated = st.checkbox("Save annotated video", value=False)

        if st.button("🚀 Start Video Analysis", type="primary"):
            # Initialize tracking
            frame_count = 0
            total_helmet = 0
            total_no_helmet = 0
            total_riders = 0

            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_placeholder = st.empty()
            video_placeholder = st.empty()

            cap = cv2.VideoCapture(video_path)

            # For saving annotated video
            if save_annotated:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('annotated_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process every Nth frame
                if frame_count % process_every_n_frames == 0:
                    # Run detection
                    results = model(frame, conf=conf_threshold)

                    # Count detections in this frame with improved logic
                    if results[0].boxes is not None:
                        for cls_id in results[0].boxes.cls.tolist():
                            class_name = model.names[int(cls_id)].lower()
                            if "helmet" in class_name and "no" not in class_name and "without" not in class_name:
                                total_helmet += 1
                            elif "no helmet" in class_name or "without helmet" in class_name or "no-helmet" in class_name:
                                total_no_helmet += 1
                            elif any(x in class_name for x in ["rider", "person", "motorcycle", "bike"]):
                                total_riders += 1

                    # Annotate frame
                    annotated = results[0].plot()

                    # Display frame
                    video_placeholder.image(
                        annotated,
                        channels="BGR",
                        use_container_width=True,
                        caption=f"Frame {frame_count}/{total_frames}"
                    )

                    # Save to video if enabled
                    if save_annotated:
                        out.write(annotated)

                    # Update progress
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                    # Update stats
                    stats_placeholder.markdown(f"""
                    ### 📊 Live Statistics
                    - **Frames Processed**: {frame_count // process_every_n_frames}
                    - **Helmets Detected**: {total_helmet}
                    - **No Helmets Detected**: {total_no_helmet}
                    - **Riders Detected**: {total_riders}
                    """)

            cap.release()
            if save_annotated:
                out.release()

            progress_bar.empty()
            status_text.text("Video analysis complete!")

            # Final statistics
            st.markdown("---")
            st.markdown("### 📊 Final Analysis Results")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", total_frames)
            with col2:
                st.metric("Helmets Detected", total_helmet)
            with col3:
                st.metric("No Helmets", total_no_helmet)
            with col4:
                st.metric("Total Riders", total_riders)

            # Alert based on overall video
            if total_no_helmet > 0:
                st.markdown(
                    f"<div class='alert alert-danger'>🚨 CRITICAL: {total_no_helmet} helmet violations detected in video!</div>",
                    unsafe_allow_html=True
                )
                st.toast("🚨 Multiple helmet violations found!", icon="🚨")
            elif total_helmet > 0:
                st.markdown(
                    f"<div class='alert alert-success'>✅ Excellent: {total_helmet} safe riders detected!</div>",
                    unsafe_allow_html=True
                )
                st.toast("✅ Safe riding observed!", icon="🪖")
            else:
                st.markdown(
                    "<div class='alert alert-info'>ℹ️ No helmet-related detections found in video.</div>",
                    unsafe_allow_html=True
                )

            # Download annotated video
            if save_annotated and os.path.exists('annotated_video.mp4'):
                with open('annotated_video.mp4', 'rb') as f:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=f,
                        file_name="helmet_detection_video.mp4",
                        mime="video/mp4"
                    )

        # Cleanup
        if os.path.exists(video_path):
            os.unlink(video_path)
        if save_annotated and os.path.exists('annotated_video.mp4'):
            os.unlink('annotated_video.mp4')

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7;'>
    <p>🪖 Advanced Helmet Detection System | Powered by YOLOv5 & Streamlit</p>
</div>
""", unsafe_allow_html=True)
