
import uuid
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image, ImageDraw

from core.camera.profile_manager import (
    load_profile,
    get_registered_cameras
)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_ROOT = PROJECT_ROOT / "videos"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
UPLOAD_ROOT = OUTPUT_ROOT / "uploads"

SUPPORTED_VIDEOS = ["mp4", "avi", "mov", "mkv"]

ZONE_COLORS = {
    "machinery": "#FFA500",
    "shared_movement": "#00BFFF",
    "pedestrian_walkway": "#00C853",
    "person_access": "#AB47BC"
}


# --------------------------------------------------
# VIDEO UTILITIES
# --------------------------------------------------

def get_camera_videos(profile):
    """Find existing videos associated with the camera."""

    video_dirs = []

    reference_video = profile.get("reference_video")

    if reference_video:
        reference_path = Path(reference_video)

        if not reference_path.is_absolute():
            reference_path = PROJECT_ROOT / reference_path

        video_dirs.append(reference_path.parent)

    # Remove duplicate directories
    video_dirs = list(dict.fromkeys(video_dirs))

    video_files = []

    for directory in video_dirs:
        if directory.exists():
            video_files.extend(
                path for path in directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {
                    f".{ext}" for ext in SUPPORTED_VIDEOS
                }
            )

    # Remove duplicate files and sort
    return sorted(set(video_files))


def save_uploaded_video(uploaded_file, camera_id):
    """Save uploaded video under the selected camera."""

    camera_upload_dir = UPLOAD_ROOT / camera_id
    camera_upload_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in {f".{ext}" for ext in SUPPORTED_VIDEOS}:
        raise ValueError("Unsupported video format.")

    filename = f"{uuid.uuid4().hex[:8]}_{Path(uploaded_file.name).name}"

    video_path = camera_upload_dir / filename

    with open(video_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return video_path


def extract_video_frame(video_path):
    """Extract the first frame and video metadata."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Unable to open selected video.")

    success, frame = cap.read()

    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    cap.release()

    if not success:
        raise RuntimeError("Unable to extract video frame.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb), metadata


# --------------------------------------------------
# ZONE PREVIEW
# --------------------------------------------------

def render_zone_overlay(image, profile):
    """Draw saved normalized zone polygons on a video frame."""

    preview = image.convert("RGBA")

    overlay = Image.new(
        "RGBA",
        preview.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    width, height = preview.size

    for zone in profile.get("zones", {}).values():

        zone_type = zone.get("type")
        color = ZONE_COLORS.get(zone_type, "#FFFFFF")

        polygons = zone.get("polygons")

        if polygons is None:
            polygons = [zone.get("polygon_normalized", [])]

        for polygon in polygons:

            if len(polygon) < 3:
                continue

            points = [
                (int(x * width), int(y * height))
                for x, y in polygon
            ]

            rgb_color = tuple(
                int(color[i:i + 2], 16)
                for i in (1, 3, 5)
            )

            draw.polygon(
                points,
                fill=(*rgb_color, 45),
                outline=(*rgb_color, 255),
                width=3
            )

    return Image.alpha_composite(preview, overlay).convert("RGB")


# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.title("🎬 Video Analysis")

st.caption(
    "Select a registered camera and analyze its warehouse CCTV footage."
)

st.divider()


# --------------------------------------------------
# CAMERA SELECTION
# --------------------------------------------------

st.subheader("1. Camera Selection")

registered_cameras = get_registered_cameras()

if not registered_cameras:
    st.warning(
        "No registered cameras found. "
        "Please register a camera first."
    )

    if st.button("Open Camera Registry"):
        st.switch_page("pages/camera_registry.py")

    st.stop()


selected_camera = st.selectbox(
    "Select registered camera",
    registered_cameras
)

profile = load_profile(selected_camera)

if not profile:
    st.error("Unable to load camera profile.")
    st.stop()


camera_name = profile.get("camera_name", selected_camera)
resolution = profile.get("resolution", [None, None])
zones = profile.get("zones", {})

col1, col2, col3 = st.columns(3)

col1.metric("Camera", camera_name)

col2.metric(
    "Resolution",
    f"{resolution[0]} × {resolution[1]}"
    if all(resolution) else "Unknown"
)

col3.metric("Configured Zones", len(zones))


# --------------------------------------------------
# VIDEO SELECTION
# --------------------------------------------------

st.divider()

st.subheader("2. Video Input")

input_method = st.radio(
    "Choose video source",
    ["Existing Project Video", "Upload New Video"],
    horizontal=True
)

selected_video = None

if input_method == "Existing Project Video":

    video_files = get_camera_videos(profile)

    if not video_files:
        st.warning(
            "No existing videos found in the camera's "
            "reference video directory."
        )

    else:
        video_options = {
            video.name: video
            for video in video_files
        }

        selected_name = st.selectbox(
            "Select CCTV video",
            list(video_options.keys())
        )

        selected_video = video_options[selected_name]

else:

    uploaded_file = st.file_uploader(
        "Upload CCTV footage",
        type=SUPPORTED_VIDEOS
    )

    if uploaded_file is not None:

        if st.button("Save Uploaded Video", type="primary"):

            try:
                selected_video = save_uploaded_video(
                    uploaded_file,
                    selected_camera
                )

                st.session_state["selected_video_path"] = str(
                    selected_video
                )

                st.success("Video uploaded successfully.")

            except Exception as error:
                st.error(str(error))

        saved_path = st.session_state.get("selected_video_path")

        if saved_path and Path(saved_path).exists():
            selected_video = Path(saved_path)


# --------------------------------------------------
# VIDEO PREVIEW
# --------------------------------------------------

if selected_video is not None:

    st.divider()

    st.subheader("3. Video Preview")

    try:
        preview_frame, metadata = extract_video_frame(
            selected_video
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Video Resolution",
            f"{metadata['width']} × {metadata['height']}"
        )

        col2.metric(
            "Frame Rate",
            f"{metadata['fps']} FPS"
        )

        col3.metric(
            "Total Frames",
            metadata["frame_count"]
        )

        expected_resolution = tuple(resolution)
        actual_resolution = (
            metadata["width"],
            metadata["height"]
        )

        if all(resolution) and actual_resolution != expected_resolution:
            st.warning(
                "Video resolution differs from the registered "
                "camera profile. Zone alignment needs validation."
            )

        st.markdown("**Original CCTV Frame**")

        st.image(
            preview_frame,
            use_container_width=True
        )

        if zones:

            st.markdown("**Zone Configuration Preview**")

            zone_preview = render_zone_overlay(
                preview_frame,
                profile
            )

            st.image(
                zone_preview,
                use_container_width=True
            )

        else:
            st.warning(
                "No zones are configured for this camera."
            )

        st.markdown("**Selected Video**")

        st.video(str(selected_video))

        st.success("Video loaded and preview generated.")

        st.session_state["analysis_video_path"] = str(
            selected_video
        )

        st.session_state["analysis_camera_id"] = selected_camera

    except Exception as error:
        st.error(f"Video preview failed: {error}")
