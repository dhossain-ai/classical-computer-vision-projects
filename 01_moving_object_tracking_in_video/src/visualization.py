import cv2
import numpy as np


def draw_tracking_overlay(
    frame: np.ndarray,
    old_points: np.ndarray,
    new_points: np.ndarray,
    speed_info: dict[str, float],
    tracked_points_count: int,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    output = frame.copy()

    if old_points is not None and new_points is not None:
        if len(old_points) > 0 and len(new_points) > 0:
            for old_point, new_point in zip(old_points, new_points):
                x_new, y_new = new_point.ravel().astype(int)
                x_old, y_old = old_point.ravel().astype(int)

                cv2.line(output, (x_old, y_old), (x_new, y_new), (0, 255, 0), 2)
                cv2.circle(output, (x_new, y_new), 3, (0, 0, 255), -1)

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 200, 0), 2)

    lines: list[str] = []

    if "km_per_hour" in speed_info:
        lines.append(f"Approx speed: {speed_info['km_per_hour']:.2f} km/h")
        lines.append(f"Motion: {speed_info['pixels_per_second']:.2f} px/s")
    else:
        lines.append(f"Approx speed: {speed_info['pixels_per_second']:.2f} px/s")

    lines.append(f"Tracked points: {tracked_points_count}")

    x0, y0 = 15, 30

    for index, text in enumerate(lines):
        y = y0 + index * 30
        cv2.putText(
            output,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output