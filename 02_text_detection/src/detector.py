from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    mser_delta: int = 5
    mser_min_area: int = 80
    mser_max_area: int = 15000

    min_box_width: int = 10
    min_box_height: int = 10
    min_box_area: int = 120
    max_box_area: int = 30000

    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 12.0
    min_fill_ratio: float = 0.18

    merge_kernel_width: int = 35
    merge_kernel_height: int = 11

    final_min_width: int = 40
    final_min_height: int = 15
    final_min_area: int = 500


def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def filter_region_boxes(
    regions: list[np.ndarray],
    config: DetectorConfig,
) -> list[tuple[int, int, int, int]]:
    filtered_boxes: list[tuple[int, int, int, int]] = []

    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))

        area = w * h
        if w < config.min_box_width or h < config.min_box_height:
            continue
        if area < config.min_box_area or area > config.max_box_area:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            continue

        contour_area = cv2.contourArea(region.reshape(-1, 1, 2))
        fill_ratio = contour_area / float(area)
        if fill_ratio < config.min_fill_ratio:
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

            if xa >= xb and ya >= yb and xa + wa <= xb + wb and ya + ha <= yb + hb:
                area_a = wa * ha
                area_b = wb * hb

                if area_b >= area_a:
                    is_nested = True
                    break

        if not is_nested:
            kept_boxes.append(box_a)

    return kept_boxes


def build_text_mask(
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
    mask = cv2.dilate(mask, kernel, iterations=1)

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


def detect_text_blocks(
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

    candidate_boxes = filter_region_boxes(regions, config)
    candidate_boxes = remove_nested_boxes(candidate_boxes)

    text_mask = build_text_mask(gray.shape, candidate_boxes, config)
    final_boxes = extract_final_boxes(text_mask, config)

    return final_boxes, text_mask