from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    mser_delta: int = 5
    mser_min_area: int = 120
    mser_max_area: int = 25000

    min_box_width: int = 20
    min_box_height: int = 20
    min_box_area: int = 400
    max_box_area: int = 80000

    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 1.8
    min_fill_ratio: float = 0.35

    merge_kernel_width: int = 11
    merge_kernel_height: int = 11

    final_min_width: int = 25
    final_min_height: int = 25
    final_min_area: int = 600

    min_red_ratio: float = 0.10
    min_saturation_mean: float = 60.0


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def get_red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return red_mask


def filter_region_boxes(
    image: np.ndarray,
    regions: list[np.ndarray],
    config: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    filtered_boxes: list[tuple[int, int, int, int]] = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = get_red_mask(image)

    for region in regions:
        region_points = region.reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(region_points)

        area = w * h
        if w < config.min_box_width or h < config.min_box_height:
            continue
        if area < config.min_box_area or area > config.max_box_area:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            continue

        contour_area = cv2.contourArea(region_points)
        fill_ratio = contour_area / float(area)
        if fill_ratio < config.min_fill_ratio:
            continue

        roi_red = red_mask[y:y + h, x:x + w]
        red_ratio = float(np.count_nonzero(roi_red)) / float(area)

        roi_hsv = hsv[y:y + h, x:x + w]
        saturation_mean = float(np.mean(roi_hsv[:, :, 1]))

        if red_ratio < config.min_red_ratio:
            continue
        if saturation_mean < config.min_saturation_mean:
            continue

        filtered_boxes.append((x, y, w, h))

    return filtered_boxes


def remove_nested_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    kept_boxes: list[tuple[int, int, int, int]] = []

    for i, box_a in enumerate(boxes):
        xa, ya, wa, ha = box_a
        is_nested = False

        for j, box_b in enumerate(boxes):
            if i == j:
                continue

            xb, yb, wb, hb = box_b

            inside_box = (
                xa >= xb
                and ya >= yb
                and xa + wa <= xb + wb
                and ya + ha <= yb + hb
            )

            if inside_box and (wb * hb) >= (wa * ha):
                is_nested = True
                break

        if not is_nested:
            kept_boxes.append(box_a)

    return kept_boxes


def build_object_mask(
    image_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    config: DetectorConfig,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)

    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.merge_kernel_width, config.merge_kernel_height),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def extract_final_boxes(
    mask: np.ndarray,
    config: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if w < config.final_min_width or h < config.final_min_height:
            continue
        if area < config.final_min_area:
            continue

        final_boxes.append((x, y, w, h))

    final_boxes.sort(key=lambda box: (box[1], box[0]))
    return final_boxes


def detect_sign_regions(
    image: np.ndarray,
    config: DetectorConfig | None = None,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    if config is None:
        config = DetectorConfig()

    gray = preprocess_image(image)

    mser = cv2.MSER_create(
        config.mser_delta,
        config.mser_min_area,
        config.mser_max_area,
    )

    regions, _ = mser.detectRegions(gray)

    candidate_boxes = filter_region_boxes(image, regions, config)
    candidate_boxes = remove_nested_boxes(candidate_boxes)

    object_mask = build_object_mask(gray.shape, candidate_boxes, config)
    final_boxes = extract_final_boxes(object_mask, config)

    return final_boxes, object_mask