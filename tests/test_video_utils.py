import cv2
import numpy as np

from src.video_utils import ensure_browser_compatible_video


def create_video(path, codec):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        10,
        (64, 48)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {codec} test video.")

    for frame_index in range(3):
        frame = np.full(
            (48, 64, 3),
            frame_index * 40,
            dtype=np.uint8
        )
        writer.write(frame)

    writer.release()


def video_codec(path):
    capture = cv2.VideoCapture(str(path))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(
        chr((fourcc >> (8 * index)) & 0xFF)
        for index in range(4)
    ).lower()
    capture.release()
    return codec


def test_transcodes_legacy_mp4v_for_browser_playback(tmp_path):
    source_path = tmp_path / "legacy.mp4"
    create_video(source_path, "mp4v")

    playback_path = ensure_browser_compatible_video(source_path)

    assert playback_path != source_path
    assert video_codec(playback_path) in {"h264", "avc1"}
    assert ensure_browser_compatible_video(source_path) == playback_path


def test_keeps_existing_h264_video_without_transcoding(tmp_path):
    source_path = tmp_path / "already_compatible.mp4"
    create_video(source_path, "avc1")

    assert ensure_browser_compatible_video(source_path) == source_path
