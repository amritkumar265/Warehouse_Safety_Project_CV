from pathlib import Path
import uuid

import streamlit as st

from core.camera.profile_manager import load_profile
from src.detection_pipeline import DetectionPipeline
from src.config import MODELS_DIR , OUTPUTS_DIR
from src.video_utils import ensure_browser_compatible_video


st.title("🔍 Detection Lab")
st.caption(
    "Run object detection with YOLO, Grounding DINO, fusion, or RF-DETR. "
    "Track the saved detections separately in Tracking Lab."
)
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

MODEL_PATH = MODELS_DIR/"yolo11s_warehouse_best.pt"

model_options = {
    "YOLO": "YOLO",
    "Grounding DINO": "Grounding DINO",
    "Fusion (YOLO + Grounding DINO)": "Fusion",
    "RF-DETR": "RF-DETR"
}
selected_model_label = st.selectbox(
    "Detection model",
    list(model_options)
)
selected_model = model_options[selected_model_label]

rfdetr_checkpoint_path = None
if selected_model == "RF-DETR":
    st.caption(
        "RF-DETR Small and the base package are Apache-2.0 licensed. "
        "[License details](https://github.com/roboflow/rf-detr/blob/develop/LICENSE)"
    )
    st.caption(
        "RF-DETR Small uses COCO classes by default. "
        "Use a fine-tuned RF-DETR checkpoint to detect forklift or pallet-truck classes."
    )
    checkpoint_input = st.text_input(
        "Optional RF-DETR checkpoint path",
        placeholder="models/rfdetr_warehouse.pth"
    ).strip()
    if checkpoint_input:
        rfdetr_checkpoint_path = Path(checkpoint_input)
        if not rfdetr_checkpoint_path.is_file():
            st.error(f"RF-DETR checkpoint not found: {rfdetr_checkpoint_path}")
            st.stop()
input_key = (
    f"{camera_id}|{video_path.resolve()}|{selected_model}|"
    f"{rfdetr_checkpoint_path or ''}"
)

if st.session_state.get("detection_input_key") != input_key:
    st.session_state.pop("detection_results", None)
    st.session_state.pop("detection_output_path", None)
    st.session_state["detection_input_key"] = input_key

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
    st.write("**Selected model**")
    st.write(selected_model_label)

with col2:
    if selected_model in {"YOLO", "Fusion"}:
        st.write("**YOLO checkpoint**")
        st.code(str(MODEL_PATH))
    elif selected_model == "RF-DETR":
        st.write("**RF-DETR checkpoint**")
        st.write(str(rfdetr_checkpoint_path or "RF-DETR Small (COCO pretrained)"))

if selected_model == "Fusion":
    st.caption(
        "Low-confidence forklift and pallet-truck boxes are kept only when "
        "YOLO and Grounding DINO agree on their location; higher-confidence "
        "vehicle boxes and person detections are unchanged."
    )

st.write("**Zone filtering:** Enabled")
st.write("**Camera profile:**" ,  camera_id)

if (
    selected_model in {"YOLO", "Fusion"}
    and not MODEL_PATH.exists()
):
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
        if selected_model in {"YOLO", "Fusion"}:
            status.write("Loading YOLO model...")
        if selected_model in {"Grounding DINO", "Fusion"}:
            status.write("Loading Grounding DINO...")
        if selected_model == "RF-DETR":
            status.write("Loading RF-DETR...")
        if selected_model == "Fusion":
            status.write("Initializing BoxFusion...")
        status.write("Initializing Zone Filter...")

        pipeline = DetectionPipeline(
            camera_profile=profile,
            yolo_model_path = MODEL_PATH,
            progress_callback=update_progress,
            model_choice=selected_model,
            rfdetr_checkpoint_path=rfdetr_checkpoint_path
        )

        status.write("Models loaded successfully...")
        status.write("Starting Video processing...")
        

        output_dir = OUTPUTS_DIR / "processed"
        output_dir.mkdir(parents = True , exist_ok=True)

        output_path = (
            output_dir /
            f"{camera_id}_{video_path.stem}_{selected_model.lower().replace(' ', '-')}_filtered_"
            f"{uuid.uuid4().hex[:8]}.mp4"
        )

        results = pipeline.process_video(
            video_path = video_path,
            output_path=output_path
        )

        progress_bar.progress(100)

        st.session_state["detection_results"] = results
        st.session_state["detection_output_path"] = str(output_path)
        detection_runs = st.session_state.setdefault("detection_runs", [])
        detection_run = {
            "run_id": uuid.uuid4().hex,
            "camera_id": camera_id,
            "video_path": str(video_path.resolve()),
            "model_choice": selected_model,
            "detection_data_path": results["detection_data_path"],
            "detection_video_path": str(output_path),
            "frames_processed": results["frames_processed"],
            "average_inference_fps": results["average_inference_fps"]
        }
        detection_runs.append(detection_run)
        st.session_state["latest_detection_run_id"] = (
            detection_run["run_id"]
        )

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

        detection_data_value = results.get("detection_data_path")
        if detection_data_value and Path(detection_data_value).is_file():
            detection_data_path = Path(detection_data_value)
            with detection_data_path.open("rb") as detections_file:
                st.download_button(
                    "Download frame-by-frame detection data",
                    data=detections_file,
                    file_name=detection_data_path.name,
                    mime="application/x-ndjson",
                    use_container_width=True
                )
            st.info(
                "Detections are saved frame by frame. Open Tracking Lab to "
                "apply ByteTrack to this or another detector run."
            )
            if st.button("Open Tracking Lab", type="primary"):
                st.switch_page("pages/tracking_lab.py")
        else:
            st.warning(
                "This result has no saved frame-by-frame boxes. Run detection "
                "again to make it available in Tracking Lab."
            )

        st.divider()
        st.subheader("5 . Filtered Output Video")

        try:
            playback_path = ensure_browser_compatible_video(output_path)
        except (RuntimeError, OSError) as error:
            st.error(f"Could not prepare the video for browser playback: {error}")
        else:
            st.video(str(playback_path), format="video/mp4")

            with open(playback_path, "rb") as video_file:
                st.download_button(
                    label = "⬇️ Download Processed Video",
                    data=video_file,
                    file_name=playback_path.name,
                    mime = "video/mp4",
                    use_container_width=True
                )

            st.success("VisionSentinel processing completed.")
    else:
        st.error(f"Processed video was not found: {output_path}")



