from pathlib import Path
import uuid

import streamlit as st

from core.camera.profile_manager import load_profile
from src.detection_pipeline import DetectionPipeline
from src.config import MODELS_DIR , OUTPUTS_DIR


st.title("🔍 Detection Lab")
st.caption("YOLO, Grounding DINO and detection fusion.")
st.divider()

#--------------------Input Validation--------------------

video_path = st.session_state.get("analysis_video_path")
camera_id = st.session_state.get("analysis_camera_id")

if not video_path or not camera_id:
    st.warning(
        "No video selected. Please select a camera and video "
        "in Video Analysis first."
    )

    if st.button("Go to Video Analysis" , type = "primary"):
        st.switch_page("pages/video_analysis.py")

    st.stop()

video_path = Path(video_path)

if not video_path.exists():
    st.error(f"Selected video does not exist: {video_path}")
    st.stop()


profile = load_profile(camera_id)

if not profile:
    st.error(f"Unable to load profile for camera: {camera_id}")
    st.stop()

input_key = f"{camera_id}|{video_path.resolve()}"

if st.session_state.get("detection_input_key") != input_key:
    st.session_state.pop("detection_results" , None)
    st.session_state.pop("detection_output_path" , None)
    st.session_state["detection_input_key"] = input_key


MODEL_PATH = MODELS_DIR/"yolo11s_warehouse_best.pt"

#-------Selected Input---------------

st.subheader("1. Selected Input")

col1 , col2 , col3 = st.columns(3)

col1.metric(
    "Camera",
    profile.get("camera_name" , camera_id)
)

col2.metric(
    "Configured Zones",
    len(profile.get("zones" , {}))
)

col3.metric(
    "Video",
    video_path.name
)

with st.expander("View selected CCTV footage"):
    st.video(str(video_path))


st.divider()

st.subheader("2 . Detection Configuration")

col1 , col2 = st.columns(2)

with col1:
    st.write("**YOLO checkpoint**")
    st.code(str(MODEL_PATH))

with col2:
    st.write("**Detection architecture**")
    st.write("YOLO + Grounding DINO + BoxFusion")

st.write("**Zone filtering:** Enabled")
st.write("**Camera profile:**" ,  camera_id)

if not MODEL_PATH.exists():
    st.error(f"YOLO checkpoint not found: {MODEL_PATH}")
    st.stop()

st.divider()
st.subheader("3. Run Detection")

if st.button(
    "🚀 Start Detection",
    type="primary",
    use_container_width=True
):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    status = st.status(
        "Initializing VisionSentinel detection engine.....",
        expanded = True
    )

    try:
        def update_progress(current , total):
            if total > 0 :
                percentage = min(
                    int((current / total) * 100),
                    100
                )

                progress_bar.progress(percentage)

                progress_bar.text(
                    f"Processing frame {current:,} / {total:,}"
                    f"({percentage}%)"
                )
        status.write("Loading YOLO model...")
        status.write("Loading Grounding DINO...")
        status.write("Initializing BoxFusion and Zone Filter...")

        pipeline = DetectionPipeline(
            camera_profile=profile,
            yolo_model_path = MODEL_PATH,
            progress_callback= update_progress
        )

        status.write("Models loaded successfully...")
        status.write("Starting Video processing...")
        

        output_dir = OUTPUTS_DIR / "processed"
        output_dir.mkdir(parents = True , exist_ok=True)

        output_path = (
            output_dir /
            f"{camera_id}_{video_path.stem}_filtered_"
            f"{uuid.uuid4().hex[:8]}.mp4"
        )

        results = pipeline.process_video(
            video_path = video_path,
            output_path=output_path
        )

        progress_bar.progress(100)

        st.session_state["detection_results"] = results
        st.session_state["detection_output_path"] = str(output_path)

        status.update(
            label = "Detection completed successfully...",
            state="complete",
            expanded = False
        )

        progress_text.success("Video processing completed.")

    except Exception as error:
        status.update(
            label="Detection failed",
            state="error",
            expanded=True
        )

        st.exception(error)

#-------Results-----------

results = st.session_state.get("detection_results")
output_path_value = st.session_state.get("detection_output_path")

if results and output_path_value:
    output_path = Path(output_path_value)

    if output_path.exists():
        st.divider()

        st.subheader("4. Detection Results")

        col1 ,col2 , col3 , col4 = st.columns(4)

        col1.metric(
            "Frame Processed",
            f"{results["frames_processed"]:,}"
        )

        col2.metric(
            "Raw Detections",
            f"{results["raw_detections"]:,}"
        )

        col3.metric(
            "Retained Detections",
            f"{results["retained_detections"]:,}"
        )

        col4.metric(
            "Suppressed Candidates",
            f"{results["suppressed_candidates"]:,}"
        )

        col1 , col2 = st.columns(2)

        col1.metric(
            "Safety Flags",
            f"{results["safety_flags"]:,}"
        )

        col2.metric(
            "Average Processing FPS",
            results["average_fps"]
        )

        st.caption(
            f"Processing time: {results["processing_time"]} seconds"
        )

        st.divider()
        st.subheader("5 . Filtered Output Video")

        if output_path_value:
            output_path = Path(output_path_value)

            if output_path.exists():
                st.video(str(output_path))

            with open(output_path , "rb") as video_file:
                st.download_button(
                    label = "⬇️ Download Processed Video",
                    data=video_file,
                    file_name=output_path.name,
                    mime = "video/mp4",
                    use_container_width=True
                )

        else:
            st.warning("the recorded output video could not be found")

    st.success("VisionSentinel processing completed.")



