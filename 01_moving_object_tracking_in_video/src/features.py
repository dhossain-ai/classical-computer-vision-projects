import cv2
import numpy as np

from .config import AppConfig


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def create_roi_mask(
    image_shape: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)

    if roi is None:
        mask[:] = 255
        return mask

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        mask[:] = 255
        return mask

    mask[y : y + h, x : x + w] = 255
    return mask


def detect_harris_points(
    gray_frame: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    config: AppConfig,
) -> np.ndarray:
    gray_float = np.float32(gray_frame)

    response = cv2.cornerHarris(
        gray_float,
        blockSize=config.harris_block_size,
        ksize=config.harris_ksize,
        k=config.harris_k,
    )
    response = cv2.dilate(response, None)

    mask = create_roi_mask(gray_frame.shape, roi)
    masked_response = response.copy()
    masked_response[mask == 0] = 0

    max_response = float(masked_response.max())
    if max_response <= 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    threshold = config.harris_threshold_ratio * max_response
    candidate_map = np.zeros_like(gray_frame, dtype=np.uint8)
    candidate_map[masked_response > threshold] = 255

    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(candidate_map)

    if num_labels <= 1:
        return np.empty((0, 1, 2), dtype=np.float32)

    candidates: list[tuple[float, int, int]] = []

    for centroid in centroids[1:]:
        x, y = int(round(centroid[0])), int(round(centroid[1]))

        if x < 0 or x >= gray_frame.shape[1] or y < 0 or y >= gray_frame.shape[0]:
            continue

        score = float(masked_response[y, x])
        candidates.append((score, x, y))

    candidates.sort(reverse=True, key=lambda item: item[0])

    selected_points: list[tuple[float, float]] = []

    for _, x, y in candidates:
        keep_point = True

        for sx, sy in selected_points:
            distance = np.hypot(x - sx, y - sy)
            if distance < config.min_corner_distance:
                keep_point = False
                break

        if keep_point:
            selected_points.append((float(x), float(y)))

        if len(selected_points) >= config.max_corners:
            break

    if not selected_points:
        return np.empty((0, 1, 2), dtype=np.float32)

    return np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)


def roi_from_points(
    points: np.ndarray,
    frame_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    if points is None or len(points) == 0:
        return None

    reshaped = points.reshape(-1, 2)

    min_x = int(np.min(reshaped[:, 0])) - padding
    min_y = int(np.min(reshaped[:, 1])) - padding
    max_x = int(np.max(reshaped[:, 0])) + padding
    max_y = int(np.max(reshaped[:, 1])) + padding

    height, width = frame_shape

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(width - 1, max_x)
    max_y = min(height - 1, max_y)

    roi_width = max_x - min_x
    roi_height = max_y - min_y

    if roi_width <= 0 or roi_height <= 0:
        return None

    return min_x, min_y, roi_width, roi_height