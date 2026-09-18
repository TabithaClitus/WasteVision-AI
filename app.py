
import streamlit as st
from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np
import tempfile
import os

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

st.markdown("""
<style>

.stApp {
    background-color: #0b1110;
}

.main .block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #101918;
    border-right: 1px solid #263b35;
}

/* Main title */

.main-title {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 5px;
}

.green {
    color: #42d993;
}

.subtitle {
    color: #94a8a0;
    font-size: 17px;
    margin-bottom: 25px;
}

/* Feature boxes */

.feature-box {
    background-color: #121c1a;
    border: 1px solid #294139;
    border-radius: 16px;
    padding: 20px;
    min-height: 150px;
}

.feature-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-text {
    color: #91a49d;
    font-size: 14px;
    line-height: 1.6;
}

/* Section */

.section-heading {
    font-size: 26px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 5px;
}

.section-text {
    color: #91a49d;
    margin-bottom: 20px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    background-color: #1d8b5b;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 12px;
}

.stButton > button:hover {
    background-color: #26a86b;
}

/* Metrics */

[data-testid="stMetric"] {
    background-color: #121c1a;
    border: 1px solid #294139;
    border-radius: 14px;
    padding: 15px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background-color: #101917;
    border: 1px dashed #396653;
    border-radius: 14px;
}

/* Footer */

.footer {
    text-align: center;
    color: #63766f;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #22322e;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

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

    st.caption("AI Waste Intelligence Platform")

    st.divider()

    st.markdown("### 🎯 Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.caption(
        f"Detections below {confidence:.2f} confidence "
        "are filtered out."
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**YOLO11s-seg**")
    st.write("Instance Segmentation")
    st.write("14 Waste Categories")

    st.divider()

    st.markdown("### 📊 Test Metrics")

    st.write("Mask mAP@50: **29.88%**")
    st.write("Mask mAP@50–95: **21.09%**")
    st.write("Box mAP@50: **32.21%**")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">♻️ <span class="green">WasteVision</span> AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Waste Instance Segmentation & Smart Waste Analytics'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# INTRO FEATURES
# =========================================================

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">🎯 Individual Detection</div>
        <div class="feature-text">
            Identifies individual waste objects even when
            multiple objects appear in the same scene.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feature2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">✂️ Pixel-Level Masks</div>
        <div class="feature-text">
            Generates individual segmentation masks rather
            than relying only on bounding boxes.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feature3:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">📊 Smart Analytics</div>
        <div class="feature-text">
            Counts detected waste categories and reports
            confidence scores for every instance.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================================
# TABS
# =========================================================

image_tab, video_tab = st.tabs(
    ["🖼️ Image Analysis", "🎥 Video Analysis"]
)

# =========================================================
# IMAGE ANALYSIS
# =========================================================

with image_tab:

    st.markdown(
        '<div class="section-heading">Image Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Upload an image to detect and segment individual waste objects.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        file_bytes = uploaded_image.read()

        image_array = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        image_rgb = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2RGB
        )

        original_col, result_col = st.columns(
            2,
            gap="large"
        )

        with original_col:

            st.markdown("### Original Image")

            st.image(
                image_rgb,
                use_container_width=True
            )

        with result_col:

            st.markdown("### Segmentation Result")

            if st.button(
                "🔍 Analyze Image",
                key="image_detect"
            ):

                with st.spinner(
                    "Running YOLO11s instance segmentation..."
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

                st.divider()

                st.markdown("### 📊 Detection Analytics")

                if not detection_rows:

                    st.info(
                        "No waste objects detected at this "
                        "confidence threshold."
                    )

                else:

                    category_counts = Counter(
                        row["Category"]
                        for row in detection_rows
                    )

                    avg_confidence = np.mean([
                        row["Confidence"]
                        for row in detection_rows
                    ])

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric(
                            "Total Objects",
                            len(detection_rows)
                        )

                    with m2:
                        st.metric(
                            "Categories",
                            len(category_counts)
                        )

                    with m3:
                        st.metric(
                            "Avg Confidence",
                            f"{avg_confidence:.2f}"
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
# VIDEO ANALYSIS
# =========================================================

with video_tab:

    st.markdown(
        '<div class="section-heading">Video Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Upload a video and analyze waste objects frame-by-frame.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Upload Waste Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        st.markdown("### Original Video")

        st.video(
            input_file.name
        )

        if st.button(
            "🎯 Analyze Video",
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
                    "AI is analyzing the video..."
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
                                f"Processing frame "
                                f"{frame_number} / "
                                f"{total_frames}"
                            )

                cap.release()
                writer.release()

                progress.progress(1.0)

                status.success(
                    "Video analysis complete!"
                )

                st.markdown("### 🎯 Segmentation Result")

                st.video(
                    output_file.name
                )

                with open(
                    output_file.name,
                    "rb"
                ) as video_file:

                    st.download_button(
                        "⬇️ Download Processed Video",
                        data=video_file,
                        file_name="wastevision_result.mp4",
                        mime="video/mp4"
                    )

                st.divider()

                st.markdown("### 📊 Video Analytics")

                if not all_detected_classes:

                    st.info(
                        "No waste objects detected."
                    )

                else:

                    category_counts = Counter(
                        all_detected_classes
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric(
                            "Frames Processed",
                            frame_number
                        )

                    with m2:
                        st.metric(
                            "Detection Events",
                            len(all_detected_classes)
                        )

                    with m3:
                        st.metric(
                            "Categories",
                            len(category_counts)
                        )

                    st.markdown(
                        "#### Waste Categories"
                    )

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
                        "Detection events represent detections "
                        "across video frames, not unique physical objects."
                    )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '♻️ WasteVision AI • YOLO11s Instance Segmentation • '
    'Smart Waste Analytics'
    '</div>',
    unsafe_allow_html=True
)
