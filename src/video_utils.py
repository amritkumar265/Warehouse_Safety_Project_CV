from pathlib import Path

import cv2


def _read_fourcc(capture):
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(
        chr((value >> (8 * index)) & 0xFF)
        for index in range(4)
    ).lower()


def ensure_browser_compatible_video(video_path):
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open processed video: {video_path}")

    codec = _read_fourcc(capture)
    if codec in {"h264", "avc1"}:
        capture.release()
        return video_path

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Processed video has invalid dimensions.")

    if fps <= 0:
        fps = 30.0

    output_width = width + width % 2
    output_height = height + height % 2
    browser_path = video_path.with_name(
        f"{video_path.stem}_browser.mp4"
    )

    output_is_current = False
    if (
        browser_path.is_file()
        and browser_path.stat().st_mtime >= video_path.stat().st_mtime
    ):
        existing_output = cv2.VideoCapture(str(browser_path))
        output_is_current = (
            existing_output.isOpened()
            and _read_fourcc(existing_output) in {"h264", "avc1"}
        )
        existing_output.release()

    if output_is_current:
        capture.release()
        return browser_path

    writer = cv2.VideoWriter(
        str(browser_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (output_width, output_height)
    )

    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Unable to create an H.264 playback video.")

    frames_written = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if output_width != width or output_height != height:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    output_height - height,
                    0,
                    output_width - width,
                    cv2.BORDER_REPLICATE
                )

            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        writer.release()

    if (
        frames_written == 0
        or not browser_path.is_file()
        or browser_path.stat().st_size == 0
    ):
        browser_path.unlink(missing_ok=True)
        raise RuntimeError("Unable to transcode the processed video.")

    verification_capture = cv2.VideoCapture(str(browser_path))
    is_playable = (
        verification_capture.isOpened()
        and _read_fourcc(verification_capture) in {"h264", "avc1"}
        and verification_capture.read()[0]
    )
    verification_capture.release()

    if not is_playable:
        browser_path.unlink(missing_ok=True)
        raise RuntimeError("The transcoded video is not playable.")

    return browser_path
