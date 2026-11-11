import json
from pathlib import Path

import cv2
import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOS_DIR = PROJECT_ROOT / "videos" / "camera001"

PROFILE_PATH = (
    PROJECT_ROOT
    /"configs"
    /"cameras"
    /"Camera_001"
    /"profile.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "zone_validation"
OUTPUT_DIR.mkdir(parents = True , exist_ok = True)

SUPPORTED_FORMATS = {".mp4" , ".mkv" , ".mov" , ".avi"}

ZONE_COLORS = {
    "machinery": (0, 165, 255),           # Orange
    "shared_movement": (255, 191, 0),     # Blue
    "pedestrian_walkway": (0, 200, 83),   # Green
    "person_access": (180, 70, 170),      # Purple
}

def load_profile():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Camera profile not found: {PROFILE_PATH}"
        )

    with open(PROFILE_PATH , "r") as file:
        return json.load(file)


def extract_frame(video_path , percentage = 50):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()

        raise RuntimeError(f"Invalid frame count : {video_path}")

    frame_index = int((percentage / 100) * (frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES , frame_index)

    success , frame = cap.read()
    cap.release()


    if not success:
        raise RuntimeError(
            f"Could not extract frame from : {video_path}"
        )

    return frame , frame_index

def draw_zones(frame , profile):
    height , width = frame.shape[:2]

    overlay = frame.copy()

    for zone_key , zone_data in profile.get("zones" , {}).items():
        color = ZONE_COLORS.get(zone_key , (255 , 255 , 255))
        polygons = zone_data.get("polygons" , [])

        for polygon_index , polygon in enumerate(polygons):

            points = np.array(
                [
                    [
                        round(x * (width-1)),
                        round(y * (height - 1))
                    ]

                    for x , y in polygon
                ],
                dtype = np.int32
            )


            if len(points) < 3:
                print(
                    f"Skipping invalid polygons:"
                    f"{zone_key} , polygon { polygon_index + 1}"
                )
                continue

            cv2.fillPoly(overlay , [points] , color)

            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color = color,
                thickness = 2
            )

            center_x = int(np.mean(points[: , 0]))
            center_y = int(np.mean(points[: , 1]))

            label = zone_data.get("display_name", zone_key)

            cv2.putText(
                frame,
                label,
                (center_x , center_y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )

    result = cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0
    )

    return result

def main():

    profile = load_profile()

    reference_width , reference_height = profile["resolution"]

    video_files = sorted(
        path for path in VIDEOS_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_FORMATS
    )

    if not video_files:
        raise FileNotFoundError(
            f"No supported videos found in {VIDEOS_DIR}"
        )

    print("\nCamera:", profile["camera_id"])
    print("Reference resolution:",
          reference_width, "x", reference_height)

    print("\nSaved zones:")

    for zone_key, zone_data in profile["zones"].items():
        print(
            f"  {zone_data['display_name']}: "
            f"{len(zone_data.get('polygons', []))} polygons"
        )

    print("\nProcessing videos...\n")

    for video_path in video_files:

        try:
            frame , frame_index = extract_frame(video_path)

            height , width = frame.shape[:2]
            print(f"Video: {video_path.name}")
            print(f"  Frame: {frame_index}")
            print(f"  Resolution: {width} x {height}")

            if (width, height) != (
                reference_width,
                reference_height
            ):
                print(
                    "  WARNING: Resolution differs from "
                    "camera reference resolution."
                )

            annotated_frame = draw_zones(frame , profile)

            output_path = (
                OUTPUT_DIR / f"{video_path.stem}_zones.jpg"
            )

            saved = cv2.imwrite(
                str(output_path),
                annotated_frame
            )

            if not saved:
                raise RuntimeError(
                    f"Could not save image: {output_path}"
                )

            print(f"  Saved: {output_path}\n")

        except Exception as error:
            print(f"ERROR processing {video_path.name}: {error}\n")

    print("Zone validation completed.")
    print(f"Output directory: {OUTPUT_DIR}")

            
if __name__ == "__main__":
    main()