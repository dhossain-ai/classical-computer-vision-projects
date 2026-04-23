from pathlib import Path

import cv2


def open_video(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return capture


def get_video_properties(
    capture: cv2.VideoCapture,
    fps_override: float | None = None,
) -> tuple[float, int, int]:
    fps = fps_override if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return fps, width, height


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    return writer