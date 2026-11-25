from pathlib import Path
import uuid

import streamlit as st

from src.config import OUTPUTS_DIR
from src.tracking_pipeline import TrackingPipeline
from src.video_utils import ensure_browser_compatible_video


st.title("🎯 Tracking Lab")
st.caption(
    "Apply ByteTrack to frame-by-frame detections saved by any model in "
    "Detection Lab. Tracking runs separately from object detection."
)
st.info(
    "Trajectory points use the bottom-center of each bounding box. "
    "Proximity thresholds are experimental image-space pixels, not physical "
    "distances or validated safety limits."
)

detection_runs = st.session_state.get("detection_runs", [])
if not detection_runs:
    st.warning(
        "No saved detections are available. Run a detector first in "
        "Detection Lab."
    )
    if st.button("Open Detection Lab", type="primary"):
        st.switch_page("pages/detection_lab.py")
    st.stop()

valid_runs = [
    run for run in detection_runs
    if (
        Path(run["video_path"]).is_file()
        and Path(run["detection_data_path"]).is_file()
    )
]
valid_runs.reverse()
if not valid_runs:
    st.error(
        "Saved detection runs were found, but their source videos or "
        "detection data files are unavailable. Run detection again."
    )
    st.stop()


run_by_id = {run["run_id"]: run for run in valid_runs}


def _run_label(run_id):
    run = run_by_id[run_id]
    return (
        f'{run["model_choice"]} | {run["camera_id"]} | '
        f'{Path(run["video_path"]).name} | {run["run_id"][:8]}'
    )


selected_run = st.selectbox(
    "Detection run to track",
    list(run_by_id),
    format_func=_run_label,
    key="tracking_detection_run"
)
selected_run = run_by_id[selected_run]

if st.session_state.get("tracking_selected_run_id") != selected_run["run_id"]:
    st.session_state.pop("tracking_results", None)
    st.session_state.pop("tracking_output_path", None)
    st.session_state["tracking_selected_run_id"] = selected_run["run_id"]

col1, col2, col3 = st.columns(3)
col1.metric("Detector", selected_run["model_choice"])
col2.metric("Camera", selected_run["camera_id"])
col3.metric("Frames in detection run", selected_run["frames_processed"])
st.caption(
    f'Input video: {Path(selected_run["video_path"]).name} · '
    f'detection inference FPS: {selected_run["average_inference_fps"]:.2f}'
)

with st.expander("View source footage"):
    st.video(selected_run["video_path"])

if st.button(
    "▶️ Run ByteTrack",
    type="primary",
    use_container_width=True
):
    progress_bar = st.progress(0)
    progress_text = st.empty()
    status = st.status("Initializing ByteTrack...", expanded=True)

    try:
        def update_progress(current, total):
            if total > 0:
                percentage = min(int(current / total * 100), 100)
                progress_bar.progress(percentage)
                progress_text.text(
                    f"Tracking frame {current:,} / {total:,} "
                    f"({percentage}%)"
                )

        pipeline = TrackingPipeline(
            video_path=selected_run["video_path"],
            detection_data_path=selected_run["detection_data_path"],
            progress_callback=update_progress,
            detection_inference_fps=selected_run["average_inference_fps"]
        )
        output_dir = OUTPUTS_DIR / "tracked"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f'{selected_run["camera_id"]}_'
            f'{Path(selected_run["video_path"]).stem}_'
            f'{selected_run["model_choice"].lower().replace(" ", "-")}_'
            f'{uuid.uuid4().hex[:8]}_tracked.mp4'
        )
        results = pipeline.process_video(output_path)
        st.session_state["tracking_results"] = results
        st.session_state["tracking_output_path"] = str(output_path)
        progress_bar.progress(100)
        status.update(
            label="ByteTrack completed successfully",
            state="complete",
            expanded=False
        )
    except Exception as error:
        status.update(
            label="ByteTrack failed",
            state="error",
            expanded=True
        )
        st.exception(error)

results = st.session_state.get("tracking_results")
output_path_value = st.session_state.get("tracking_output_path")
if results and output_path_value:
    output_path = Path(output_path_value)
    if not output_path.is_file():
        st.error(f"Tracked video was not found: {output_path}")
        st.stop()

    summary = results["tracking_summary"]
    st.divider()
    st.subheader("Tracking Results")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_col1.metric("Frames tracked", results["frames_processed"])
    metrics_col2.metric("Unique people", summary["unique_tracked_people"])
    metrics_col3.metric(
        "Unique forklifts",
        summary["unique_tracked_forklifts"]
    )
    metrics_col4.metric(
        "Unique pallet trucks",
        summary["unique_tracked_pallet_trucks"]
    )
    st.caption(
        f'Average tracking FPS: {results["average_fps"]:.2f} · '
        f'average track duration: '
        f'{summary["average_track_duration_seconds"]:.2f}s · '
        f'maximum track duration: '
        f'{summary["maximum_track_duration_seconds"]:.2f}s · '
        f'detection inference FPS: '
        f'{summary["average_inference_fps"]:.2f}'
    )
    st.caption(
        "Proximity values are image-space pixel measurements only; they "
        "must not be interpreted as physical distances or certified risk."
    )

    try:
        playback_path = ensure_browser_compatible_video(output_path)
    except (RuntimeError, OSError) as error:
        st.error(f"Could not prepare the tracked video for playback: {error}")
    else:
        st.video(str(playback_path), format="video/mp4")
        with playback_path.open("rb") as video_file:
            st.download_button(
                "Download tracked video",
                data=video_file,
                file_name=playback_path.name,
                mime="video/mp4",
                use_container_width=True
            )

    download_col1, download_col2 = st.columns(2)
    events_path = Path(results["events_csv"])
    if events_path.is_file():
        with events_path.open("rb") as events_file:
            download_col1.download_button(
                "Download tracking events CSV",
                data=events_file,
                file_name=events_path.name,
                mime="text/csv",
                use_container_width=True
            )
    summary_path = Path(results["summary_json"])
    if summary_path.is_file():
        with summary_path.open("rb") as summary_file:
            download_col2.download_button(
                "Download tracking summary JSON",
                data=summary_file,
                file_name=summary_path.name,
                mime="application/json",
                use_container_width=True
            )

    with st.expander("Tracking summary details"):
        st.json(summary)
