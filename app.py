
import streamlit as st
from ultralytics import YOLO
from collections import Counter
import cv2
import tempfile
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide"
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

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

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("♻️ WasteVision AI")

st.markdown(
    """
    ### Real-Time Waste Instance Segmentation & Smart Waste Analytics

    Detect individual waste objects using **YOLO11s Instance Segmentation**.
    
    **Supported inputs:** Images 🖼️ and Videos 🎥
    """
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Model:** YOLO11s-seg  
    **Classes:** 14  
    **Task:** Instance Segmentation
    """
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

image_tab, video_tab = st.tabs(
    ["🖼️ Image Detection", "🎥 Video Detection"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================

with image_tab:

    st.header("🖼️ Upload a Waste Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        # Read image
        file_bytes = uploaded_image.read()

        image_array = cv2.imdecode(
            __import__("numpy").frombuffer(
                file_bytes,
                __import__("numpy").uint8
            ),
            cv2.IMREAD_COLOR
        )

        image_rgb = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2RGB
        )

        # Display original
        st.subheader("Original Image")

        st.image(
            image_rgb,
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Waste",
            key="image_detect"
        ):

            with st.spinner("Analyzing image..."):

                results = model.predict(
                    source=image_rgb,
                    conf=confidence,
                    imgsz=640,
                    verbose=False
                )

                result = results[0]

                # Generate annotated image
                annotated = result.plot()

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

            st.subheader("🎯 Instance Segmentation Result")

            st.image(
                annotated_rgb,
                use_container_width=True
            )

            # ------------------------------------------
            # DETECTION DATA
            # ------------------------------------------

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

            # ------------------------------------------
            # ANALYTICS
            # ------------------------------------------

            st.subheader("📊 Waste Analysis")

            if len(detection_rows) == 0:

                st.info(
                    "No waste objects detected."
                )

            else:

                category_counts = Counter(
                    row["Category"]
                    for row in detection_rows
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Objects",
                        len(detection_rows)
                    )

                with col2:
                    st.metric(
                        "Categories Detected",
                        len(category_counts)
                    )

                st.markdown(
                    "### Category Distribution"
                )

                for category, count in category_counts.items():

                    st.write(
                        f"**{category}:** {count}"
                    )

                st.markdown(
                    "### Detected Instances"
                )

                st.dataframe(
                    detection_rows,
                    use_container_width=True,
                    hide_index=True
                )


# ==================================================
# VIDEO DETECTION
# ==================================================

with video_tab:

    st.header("🎥 Upload a Waste Video")

    st.info(
        "Upload a short video to detect and segment waste objects frame-by-frame."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_video.read()
        )

        input_file.close()

        st.subheader("Original Video")

        st.video(
            input_file.name
        )

        if st.button(
            "🎯 Detect Waste in Video",
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
                    "Could not open the uploaded video."
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

                # Prevent invalid FPS
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
                    "Processing video..."
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

                        # Get annotated frame
                        annotated_frame = result.plot()

                        writer.write(
                            annotated_frame
                        )

                        # Collect detected classes
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

                            percent = min(
                                frame_number /
                                total_frames,
                                1.0
                            )

                            progress.progress(
                                percent
                            )

                            status.text(
                                f"Processing frame "
                                f"{frame_number}/"
                                f"{total_frames}"
                            )

                cap.release()
                writer.release()

                progress.progress(1.0)

                status.success(
                    "Video processing complete!"
                )

                # --------------------------------------
                # DISPLAY RESULT
                # --------------------------------------

                st.subheader(
                    "🎯 Segmented Video"
                )

                st.video(
                    output_file.name
                )

                # --------------------------------------
                # VIDEO ANALYTICS
                # --------------------------------------

                st.subheader(
                    "📊 Video Detection Summary"
                )

                if len(all_detected_classes) == 0:

                    st.info(
                        "No waste objects were detected "
                        "in the video."
                    )

                else:

                    category_counts = Counter(
                        all_detected_classes
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Frames Processed",
                            frame_number
                        )

                    with col2:

                        st.metric(
                            "Detection Events",
                            len(all_detected_classes)
                        )

                    st.markdown(
                        "### Categories Detected"
                    )

                    for category, count in category_counts.items():

                        st.write(
                            f"**{category}:** {count} detection events"
                        )

                    st.caption(
                        "Note: Video counts represent detection events "
                        "across frames, not unique physical objects."
                    )

        # Clean up input after processing
        # Output remains available during the session
