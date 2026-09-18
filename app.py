
import streamlit as st
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np
import tempfile
import os
from textwrap import dedent

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(dedent("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #0b1110;
        color: #f5f7f6;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #101918;
        border-right: 1px solid #263532;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------- HERO ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(25, 88, 64, 0.35),
                rgba(11, 17, 16, 0.95)
            );
        border: 1px solid #2c5144;
        border-radius: 24px;
        padding: 34px 38px;
        margin-bottom: 28px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-title span {
        color: #41d88f;
    }

    .hero-subtitle {
        color: #aabbb5;
        font-size: 17px;
        margin-top: 10px;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 14px;
        border-radius: 20px;
        background: rgba(65,216,143,0.12);
        border: 1px solid rgba(65,216,143,0.35);
        color: #62e5a0;
        font-size: 13px;
        font-weight: 600;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #8fa39c;
        margin-bottom: 22px;
        font-size: 14px;
    }

    /* ---------- CARDS ---------- */

    .metric-card {
        background: #121c1a;
        border: 1px solid #263b35;
        border-radius: 18px;
        padding: 20px 22px;
        min-height: 105px;
    }

    .metric-label {
        color: #8fa39c;
        font-size: 13px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #f5f7f6;
        font-size: 30px;
        font-weight: 750;
    }

    .metric-accent {
        color: #48db92;
    }

    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: #101917;
        border: 1px solid #253a34;
        border-radius: 18px;
        padding: 20px;
        height: 100%;
    }

    .info-card h4 {
        margin-top: 0;
        color: #eaf2ef;
    }

    .info-card p {
        color: #94a7a0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ---------- UPLOAD AREA ---------- */

    [data-testid="stFileUploader"] {
        background: #101917;
        border: 1px dashed #3a6354;
        border-radius: 18px;
        padding: 8px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #32976a;
        background: #1d8b5b;
        color: white;
        font-weight: 700;
        padding: 0.65rem 1rem;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #26a86b;
        border-color: #43c987;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 650;
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #647871;
        font-size: 12px;
        padding-top: 35px;
        border-top: 1px solid #1e2d29;
        margin-top: 40px;
    }

</style>
"""), unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Plastic Bag/Wrapper",
    "Cigarette",
    "Other/Unlabeled Waste",
    "Other Plastic",
    "Plastic Bottle",
    "Plastic Cap/Lid",
    "Other Metal",
    "Metal Can",
    "Glass",
    "Cardboard/Carton",
    "Plastic Straw/Utensil",
    "Paper",
    "Styrofoam",
    "Plastic Cup"
]

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ♻️ WasteVision")

    st.markdown(
        "<p style='color:#7f968e;'>AI Waste Intelligence</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🎯 Detection Settings")

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.caption(
        f"Only detections above **{confidence:.2f}** confidence "
        "will be displayed."
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.markdown(
        """
        **YOLO11s-seg**

        **Task:** Instance Segmentation

        **Classes:** 14

        **Input:** Image / Video
        """
    )

    st.markdown("---")

    st.markdown("### 📊 Test Performance")

    st.markdown(
        """
        **Mask mAP@50:** 29.88%

        **Mask mAP@50–95:** 21.09%

        **Box mAP@50:** 32.21%
        """
    )

# =========================================================
# HERO
# =========================================================

st.markdown(dedent("""
<div class="hero">

    <div class="hero-title">
        ♻️ <span>WasteVision</span> AI
    </div>

    <div class="hero-subtitle">
        Real-Time Waste Instance Segmentation &
        Smart Waste Analytics
    </div>

    <div class="hero-badge">
        ● AI-POWERED WASTE DETECTION
    </div>

</div>
"""), unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================

image_tab, video_tab = st.tabs(
    ["🖼️  IMAGE ANALYSIS", "🎥  VIDEO ANALYSIS"]
)

# =========================================================
# IMAGE TAB
# =========================================================

with image_tab:

    st.markdown(
        '<div class="section-title">Image Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a waste image to identify individual objects, '
        'generate pixel-level masks and analyze waste composition.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload waste image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>🎯 Instance Detection</h4>
                <p>
                Detect individual waste objects even when
                multiple objects appear in the same scene.
                </p>
            </div>
            """), unsafe_allow_html=True)

        with col2:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>✂️ Pixel Masks</h4>
                <p>
                Generate individual segmentation masks instead
                of relying only on rectangular bounding boxes.
                </p>
            </div>
            """), unsafe_allow_html=True)

        with col3:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>📊 Smart Analytics</h4>
                <p>
                Automatically count detected waste categories
                and display confidence scores.
                </p>
            </div>
            """), unsafe_allow_html=True)

    else:

        file_bytes = uploaded_image.read()

        image_array = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        image_rgb = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2RGB
        )

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            st.markdown("#### Original Image")

            st.image(
                image_rgb,
                use_container_width=True
            )

        with col2:

            st.markdown("#### Segmentation Result")

            detect_image = st.button(
                "🔍  ANALYZE IMAGE",
                key="image_detect"
            )

            if detect_image:

                with st.spinner(
                    "Running instance segmentation..."
                ):

                    results = model.predict(
                        source=image_rgb,
                        conf=confidence,
                        imgsz=640,
                        verbose=False
                    )

                    result = results[0]

                    annotated = result.plot()

                    annotated_rgb = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                st.image(
                    annotated_rgb,
                    use_container_width=True
                )

                # -----------------------------------------
                # DETECTION DATA
                # -----------------------------------------

                detection_rows = []

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    classes = (
                        result.boxes.cls
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .numpy()
                    )

                    for cls_id, conf in zip(
                        classes,
                        confidences
                    ):

                        detection_rows.append({
                            "Category": class_names[cls_id],
                            "Confidence": round(
                                float(conf),
                                3
                            )
                        })

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    'Detection Analytics'
                    '</div>',
                    unsafe_allow_html=True
                )

                if not detection_rows:

                    st.info(
                        "No waste objects detected at the "
                        "current confidence threshold."
                    )

                else:

                    category_counts = Counter(
                        row["Category"]
                        for row in detection_rows
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    TOTAL OBJECTS
                                </div>
                                <div class="metric-value metric-accent">
                                    {len(detection_rows)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    CATEGORIES
                                </div>
                                <div class="metric-value">
                                    {len(category_counts)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c3:

                        avg_conf = np.mean([
                            x["Confidence"]
                            for x in detection_rows
                        ])

                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    AVG CONFIDENCE
                                </div>
                                <div class="metric-value">
                                    {avg_conf:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("#### Category Distribution")

                    chart_data = {
                        "Category": list(
                            category_counts.keys()
                        ),
                        "Objects": list(
                            category_counts.values()
                        )
                    }

                    st.bar_chart(
                        chart_data,
                        x="Category",
                        y="Objects"
                    )

                    st.markdown("#### Detected Instances")

                    st.dataframe(
                        detection_rows,
                        use_container_width=True,
                        hide_index=True
                    )

# =========================================================
# VIDEO TAB
# =========================================================

with video_tab:

    st.markdown(
        '<div class="section-title">Video Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a video and WasteVision will perform '
        'frame-by-frame instance segmentation.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload waste video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>🎥 Video Input</h4>
                <p>
                Upload MP4, AVI, MOV or MKV waste footage.
                </p>
            </div>
            """), unsafe_allow_html=True)

        with col2:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>⚡ Frame Analysis</h4>
                <p>
                Each video frame is analyzed using the trained
                YOLO11s segmentation model.
                </p>
            </div>
            """), unsafe_allow_html=True)

        with col3:
            st.markdown(dedent("""
            <div class="info-card">
                <h4>📊 Waste Events</h4>
                <p>
                View detected categories and the number of
                detection events throughout the video.
                </p>
            </div>
            """), unsafe_allow_html=True)

    else:

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        st.markdown("#### Original Video")

        st.video(
            input_file.name
        )

        st.markdown("")

        if st.button(
            "🎯  ANALYZE VIDEO",
            key="video_detect"
        ):

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            cap = cv2.VideoCapture(
                input_file.name
            )

            if not cap.isOpened():

                st.error(
                    "Unable to open the uploaded video."
                )

            else:

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                if fps <= 0:
                    fps = 25

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_file.name,
                    fourcc,
                    fps,
                    (width, height)
                )

                progress = st.progress(0)

                status = st.empty()

                frame_number = 0

                all_detected_classes = []

                with st.spinner(
                    "AI is analyzing your video..."
                ):

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            imgsz=640,
                            verbose=False
                        )

                        result = results[0]

                        annotated_frame = result.plot()

                        writer.write(
                            annotated_frame
                        )

                        if (
                            result.boxes is not None
                            and len(result.boxes) > 0
                        ):

                            classes = (
                                result.boxes.cls
                                .cpu()
                                .numpy()
                                .astype(int)
                            )

                            for cls_id in classes:

                                all_detected_classes.append(
                                    class_names[cls_id]
                                )

                        frame_number += 1

                        if total_frames > 0:

                            progress.progress(
                                min(
                                    frame_number /
                                    total_frames,
                                    1.0
                                )
                            )

                            status.text(
                                f"Analyzing frame "
                                f"{frame_number} / "
                                f"{total_frames}"
                            )

                cap.release()
                writer.release()

                progress.progress(1.0)

                status.success(
                    "Video analysis complete!"
                )

                # -----------------------------------------
                # RESULT VIDEO
                # -----------------------------------------

                st.markdown("#### Segmentation Result")

                st.video(
                    output_file.name
                )

                with open(
                    output_file.name,
                    "rb"
                ) as video_file:

                    st.download_button(
                        label="⬇️  Download Processed Video",
                        data=video_file,
                        file_name="wastevision_result.mp4",
                        mime="video/mp4"
                    )

                # -----------------------------------------
                # VIDEO ANALYTICS
                # -----------------------------------------

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    'Video Analytics'
                    '</div>',
                    unsafe_allow_html=True
                )

                if not all_detected_classes:

                    st.info(
                        "No waste objects were detected."
                    )

                else:

                    category_counts = Counter(
                        all_detected_classes
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    FRAMES PROCESSED
                                </div>
                                <div class="metric-value metric-accent">
                                    {frame_number}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    DETECTION EVENTS
                                </div>
                                <div class="metric-value">
                                    {len(all_detected_classes)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with c3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">
                                    CATEGORIES
                                </div>
                                <div class="metric-value">
                                    {len(category_counts)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("#### Waste Categories")

                    video_chart = {
                        "Category": list(
                            category_counts.keys()
                        ),
                        "Detection Events": list(
                            category_counts.values()
                        )
                    }

                    st.bar_chart(
                        video_chart,
                        x="Category",
                        y="Detection Events"
                    )

                    st.caption(
                        "Video detection events represent detections "
                        "across frames, not unique physical objects."
                    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(dedent("""
<div class="footer">
    ♻️ WasteVision AI &nbsp;•&nbsp;
    YOLO11s Instance Segmentation &nbsp;•&nbsp;
    Smart Waste Analytics
</div>
"""), unsafe_allow_html=True)
