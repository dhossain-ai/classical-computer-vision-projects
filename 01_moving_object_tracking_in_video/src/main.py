import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, ensure_output_dir
from .features import detect_harris_points, roi_from_points, to_gray
from .optical_flow import track_points
from .speed_estimation import SpeedEstimator
from .video_io import create_video_writer, get_video_properties, open_video
from .visualization import draw_tracking_overlay


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a moving vehicle using Harris corners and Lucas-Kanade optical flow.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output video",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Use the full frame instead of selecting ROI manually",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video fps if needed",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="Approximate pixel-to-meter scale for real-world speed",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()

    if args.input is not None:
        config.input_video = Path(args.input)

    if args.output is not None:
        config.output_video = Path(args.output)

    if args.no_preview:
        config.show_preview = False

    if args.no_save:
        config.save_output = False

    if args.full_frame:
        config.use_manual_roi = False

    if args.fps is not None:
        config.fps_override = args.fps

    if args.pixels_per_meter is not None:
        config.pixels_per_meter = args.pixels_per_meter

    return config


def select_frame_and_roi(
    capture: cv2.VideoCapture,
    use_manual_roi: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    success, frame = capture.read()
    if not success:
        raise RuntimeError("Could not read the first frame from the video.")

    if not use_manual_roi:
        return frame, None

    paused = True
    current_frame = frame.copy()

    while True:
        display_frame = current_frame.copy()

        cv2.putText(
            display_frame,
            "Space: play/pause  |  s: select ROI  |  q: quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        status_text = "Paused" if paused else "Playing"
        cv2.putText(
            display_frame,
            f"Status: {status_text}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Video Preview", display_frame)

        wait_time = 0 if paused else 30
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord(" "):
            paused = not paused

        elif key == ord("s"):
            roi = cv2.selectROI(
                "Select vehicle ROI and press ENTER",
                current_frame,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyWindow("Select vehicle ROI and press ENTER")
            cv2.destroyWindow("Video Preview")

            if roi[2] == 0 or roi[3] == 0:
                return current_frame, None

            return current_frame, tuple(int(value) for value in roi)

        elif key == ord("q") or key == 27:
            cv2.destroyWindow("Video Preview")
            raise RuntimeError("ROI selection cancelled by user.")

        if not paused:
            success, next_frame = capture.read()
            if not success:
                paused = True
            else:
                current_frame = next_frame.copy()


def main() -> None:
    args = parse_arguments()
    config = build_config(args)

    ensure_output_dir()

    print(f"Input video: {config.input_video}")
    print(f"Output video: {config.output_video}")

    capture = open_video(config.input_video)
    fps, width, height = get_video_properties(capture, config.fps_override)

    first_frame, initial_roi = select_frame_and_roi(
        capture,
        config.use_manual_roi,
    )
    current_roi = initial_roi

    first_gray = to_gray(first_frame)
    tracked_points = detect_harris_points(first_gray, current_roi, config)

    if len(tracked_points) == 0:
        capture.release()
        raise RuntimeError("No Harris corner points were found in the selected region.")

    writer = None
    if config.save_output:
        writer = create_video_writer(
            config.output_video,
            fps,
            (width, height),
        )

    speed_estimator = SpeedEstimator(
        smoothing_window=config.smoothing_window,
        pixels_per_meter=config.pixels_per_meter,
    )

    previous_gray = first_gray

    while True:
        success, frame = capture.read()
        if not success:
            break

        current_gray = to_gray(frame)

        old_points, new_points = track_points(
            previous_gray,
            current_gray,
            tracked_points,
            config,
        )

        speed_info = speed_estimator.update(old_points, new_points, fps)

        updated_roi = roi_from_points(
            new_points,
            current_gray.shape,
            config.roi_padding,
        )
        if updated_roi is not None:
            current_roi = updated_roi

        annotated_frame = draw_tracking_overlay(
            frame=frame,
            old_points=old_points,
            new_points=new_points,
            speed_info=speed_info,
            tracked_points_count=len(new_points),
            roi=current_roi,
        )

        if writer is not None:
            writer.write(annotated_frame)

        if config.show_preview:
            cv2.imshow("Moving Object Tracking", annotated_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if len(new_points) < config.min_tracked_points:
            redetect_roi = current_roi if current_roi is not None else initial_roi
            refreshed_points = detect_harris_points(current_gray, redetect_roi, config)

            if len(refreshed_points) > 0:
                tracked_points = refreshed_points
            else:
                tracked_points = new_points
        else:
            tracked_points = new_points

        previous_gray = current_gray

    capture.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()