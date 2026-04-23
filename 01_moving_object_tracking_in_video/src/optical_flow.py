import cv2
import numpy as np

from .config import AppConfig


def track_points(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if previous_points is None or len(previous_points) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    lk_params = dict(
        winSize=config.lk_win_size,
        maxLevel=config.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            config.lk_max_iterations,
            config.lk_epsilon,
        ),
    )

    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        **lk_params,
    )

    if current_points is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty

    valid_mask = status.reshape(-1) == 1

    old_valid = previous_points[valid_mask].reshape(-1, 1, 2).astype(np.float32)
    new_valid = current_points[valid_mask].reshape(-1, 1, 2).astype(np.float32)

    return old_valid, new_valid