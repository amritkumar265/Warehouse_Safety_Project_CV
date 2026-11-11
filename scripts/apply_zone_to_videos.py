import cv2
from pathlib import Path

from validate_zones import (
    VIDEOS_DIR,
    SUPPORTED_FORMATS,
    load_profile,
    draw_zones
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "zone_applied"
OUTPUTS_DIR.mkdir(parents = True)

def process_video(video_path , profile):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps < 0:
        fps = 25.0

    output_path = (
        OUTPUTS_DIR / f"{video_path.stem}_zones.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width , height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create ourpur video : {output_path}"
        )

    frame_number = 0

    print(f"\nProcessing: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    try:
        while True:

            success , frame = cap.read()

            if not success:
                break

            annotated_frame = draw_zones(frame , profile)

            writer.write(annotated_frame)

            frame_number += 1

            if frame_number % 100 == 0:
                print(
                    f"  Processed {frame_number}/{total_frames} frames"
                )
    finally:
        cap.release()
        writer.release()

    print(f"Completed: {video_path.name}")
    print(f"Frames written: {frame_number}")
    print(f"Output: {output_path}")

    return frame_number

def main():
    profile = load_profile()

    video_files = sorted(
        path for path in VIDEOS_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_FORMATS
    )

    if not video_files:
        raise FileNotFoundError(
            f"No videos found in {VIDEOS_DIR}"
        )

    print("\nCamera: " , profile["camera_id"])
    print("Applying saved zones to all video frames.")

    total_processed = 0

    for video_path in video_files:
        try:
            total_processed += process_video(
                video_path,
                profile
            )

        except Exception as error:
            print(f"ERROR : {video_path.name} : {error}")

    print("\n--------------------------------")
    print("ALL-FRAME ZONE PROCESSING FINISHED")
    print("Total frames processed:", total_processed)
    print("Output directory:", OUTPUTS_DIR)
    print("--------------------------------")

if __name__ == "__main__":
    main()


            


