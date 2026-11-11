from core.camera.profile_manager import (
    CAMERAS_DIR,
    load_profile,
    save_profile,
    get_registered_cameras
)

from pathlib import Path

import cv2
import streamlit as st
from PIL import Image , ImageDraw
from streamlit_drawable_canvas import st_canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "videos" / "camera001"
CAMERAS_DIR = PROJECT_ROOT / "configs" / "cameras"
CAMERAS_DIR.mkdir(parents = True , exist_ok = True)

SUPPORTED_VIDEOS = {".mp4" , ".avi" , ".mov" , ".mkv"}

ZONE_TYPES = {
    "Machine Area" : "machinery",
    "Shared Movement Area" : "shared_movement",
    "Pedestrian Walkway" : "pedestrian_walkway",
    "Miscellaneous" : "miscellaneous"
}

ZONE_COLORS = {
    "machinery" : "#FFA500",
    "shared_movement" : "#00BFFF",
    "pedestrian_walkway" : "#00C853",
    "miscellaneous" : "#AB47BC"
}

def get_video_files():
    if not VIDEO_DIR.exists():
        return []

    return sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEOS
    )

@st.cache_data(show_spinner = False)
def extract_reference_frame(video_path , percentage):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_index = min(
        int(frame_count * percentage / 100),
        max(frame_count - 1 , 0)
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES , frame_index)
    success , frame = cap.read()
    cap.release()


    if not success:
        raise RuntimeError("Could not extract reference frame.")

    rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

    metadata = {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "reference_frame": frame_index,
    }

    return Image.fromarray(rgb), metadata

def extract_polygon(canvas_data , canvas_width , canvas_height):
    if not canvas_data:
        return None
    objects = canvas_data.get("objects" , [])

    if not objects:
        return None

    obj = objects[-1]

    if obj.get("type") != "Path":
        return None
    

    path_commands = obj.get("path" , [])

    vertices = []

    for command in path_commands:
        if command[0] in ("M" , "L") and len(command) >= 3:
            x = float(command[1])
            y = float(command[2])

            vertices.append([x,y])

    if len(vertices) < 3:
        return None

    normalized = []

    for x , y in vertices:
        normalized.append([
            round(max(0.0 , min(1.0 , x / canvas_width)) , 6),
            round(max(0.0 , min(1.0 , y/canvas_height)) , 6)
        ])

    return normalized

def render_saved_zones(image,profiles):
    preview = image.copy().convert("RGBA")
    overlay = Image.new("RGBA" , preview.size , (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    width , height = preview.size

    for zone in profiles.get("zones" , {}).values():
        polygons = zone.get("polygons")

        color = ZONE_COLORS.get(zone['type'] , "#FFFFFF")

        if polygons is None:
            polygons = [zone["polygon_normalized"]]

        for polygon in polygons:
            points = [
                (int(x * width), int(y * height))
                for x, y in polygon
            ]

            draw.polygon(
                points,
                fill=color + "35",
                outline=color
            )

    return Image.alpha_composite(preview, overlay).convert("RGB")

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.title("Camera Registry")
st.caption("Register CCTV cameras and configure warehouse safety zones.")

st.divider()

# Discover existing camera profiles
existing_profiles = get_registered_cameras()

registered_count = len(existing_profiles)
zone_count = 0

for camera_id in existing_profiles:
    profile = load_profile(camera_id)
    zone_count += len(profile.get("zones", {}))

col1, col2 = st.columns(2)
col1.metric("Registered Cameras", registered_count)
col2.metric("Configured Zones", zone_count)

st.divider()

st.subheader("Camera Configuration")

with st.expander("Register a new camera" , expanded = not existing_profiles):

    camera_id = st.text_input(
        "Camera ID",
        value = "Camera_001",
        help = "Use a unique ID , for example Camera_01."
    )

    camera_name = st.text_input(
        "Camera Name",
        value = "Warehouse Camera 001"
    )

    video_files = get_video_files()

    if video_files:
        video_labels = {
            str(p.relative_to(PROJECT_ROOT)) : p
            for p in video_files
        }

        selected_video_label = st.selectbox(
            "Reference video",
            list(video_labels.keys())
        )

        selected_video = video_labels[selected_video_label]

        frame_percentage = st.slider(
            "reference frame position (%)",
            min_value = 0,
            max_value = 100,
            value = 50
        )

        if st.button("Extract Reference Frame"):
            try:
                image , metadata = extract_reference_frame(
                    selected_video,
                    frame_percentage
                )
                camera_dir = CAMERAS_DIR / camera_id
                camera_dir.mkdir(parents = True , exist_ok = True)

                reference_path = camera_dir / "reference.jpg"
                image.save(str(reference_path), format="JPEG")

                st.success(f"Reference saved: {reference_path}")

                st.session_state["reference_image"] = image
                st.session_state["reference_metadata"] = metadata
                st.session_state["reference_video"] = selected_video_label
                st.success("Reference frame extracted.")

            except Exception as error:
                st.error(str(error))

        if not video_files:
            st.warning(
                "No video found in the project's videos directory"
            )

        if st.button("Register Camera", type="primary"):

            if not camera_id.strip() or not camera_name.strip():
                st.error("Camera ID and name are required.")

            elif not (CAMERAS_DIR / camera_id / "reference.jpg").exists():
                st.error("Extract a reference frame before registering.")

            else:
                metadata = st.session_state.get("reference_metadata", {})
                existing_profile = load_profile(camera_id) or {}

                profile = {
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "resolution": [
                        metadata.get("width"),
                        metadata.get("height")
                    ],
                    "reference_video": st.session_state.get(
                    "reference_video",
                    existing_profile.get("reference_video")
                ),
            # Preserve existing zone configuration
                "zones": existing_profile.get("zones", {})
            }

        # Preserve existing metadata if new metadata is unavailable
            if not metadata:
                profile["resolution"] = existing_profile.get(
                "resolution", [None, None]
            )

            save_profile(profile)

            st.session_state["active_camera"] = camera_id

            st.success(f"{camera_id} registered successfully.")
            st.rerun()


st.divider()


st.subheader("Warehouse Zone Designer")

if not existing_profiles:
    st.info("Register a camera to begin configuration zones.")
    st.stop()


selected_camera = st.selectbox(
    "Select Camera",
    existing_profiles
)

profile = load_profile(selected_camera)

camera_dir = CAMERAS_DIR/selected_camera
reference_path = camera_dir/ "reference.jpg"

if not reference_path.exists():
    st.error("Reference image is missing.")
    st.stop()

reference_image = Image.open(reference_path).convert("RGB")

original_width , original_height = reference_image.size

canvas_width = min(1100 , original_width)
scale = canvas_width / original_width
canvas_height = int(original_height * scale)

display_image = reference_image.resize(
    (canvas_width, canvas_height)
)


if profile.get("zones"):
    st.markdown("**Existing zone configuration**")

    preview = render_saved_zones(
        reference_image,
        profile
    )

    st.image(preview, use_container_width=True)

st.caption(
    f"Camera resolution: {original_width} x {original_height}"
)

select_zone_name = st.selectbox(
    "Select zone to anotate",
    list(ZONE_TYPES.keys())
)

zone_key = ZONE_TYPES[select_zone_name]
zone_color = ZONE_COLORS[zone_key]

st.markdown(f"**Drawing: {select_zone_name}**")

st.info(
    "Click to create polygon vertices. "
    "Complete the polygon by clicking near its starting point. "
    "Use the canvas toolbar to undo or clear."
)

canvas_key = f"zone_canvas_{selected_camera}_{zone_key}"

canvas_result = st_canvas(
    fill_color="rgba(0 , 200 , 83 , 0.20)",
    stroke_width=3,
    stroke_color=zone_color,
    background_image=display_image,
    update_streamlit=True,
    height=canvas_height,
    width=canvas_width,
    drawing_mode="polygon",
    key = canvas_key
)

if st.button("Save Zone Polygon" , type = "primary"):

    polygon = extract_polygon(
        canvas_result.json_data,
        canvas_width,
        canvas_height
    )

    if polygon is None:
        st.error("No valid polygon found. Draw at least three vertices.  ")

    else:
        if zone_key not in profile["zones"]:
            profile["zones"][zone_key] = {
                "display_name" : select_zone_name,
                "type" : zone_key,
                "polygons" : []
            }

        profile["zones"][zone_key]["polygons"].append(polygon)


        save_profile(profile)

        st.success(
            f"Polygon {len(profile['zones'][zone_key]['polygons'])}"
            f"saved for {select_zone_name}"
        )
        st.rerun()

st.divider()
st.subheader("Configured Zones")

if not profile.get("zones"):
    st.info("No zones configured yet.")

else:

    for key, zone in profile["zones"].items():

        polygons = zone.get("polygons")

        if polygons is None:
            polygons = [zone["polygon_normalized"]]

        with st.expander(
            f"{zone['display_name']} ({len(polygons)} polygons)"
        ):
            for index, polygon in enumerate(polygons):

                st.write(f"Polygon {index + 1}")
                st.json(polygon)

                if st.button(
                    f"Delete Polygon {index + 1}",
                    key=f"delete_{selected_camera}_{key}_{index}"
                ):
                    polygons.pop(index)

                    if polygons:
                        zone["polygons"] = polygons
                    else:
                        del profile["zones"][key]

                    save_profile(profile)
                    st.rerun()

    
