from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np


class DetectedInstance(NamedTuple):
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    good_match_count: int
    matched_points: np.ndarray


@dataclass
class MatchConfig:
    nfeatures: int = 3000
    ratio_test: float = 0.85

    min_good_matches: int = 6
    min_cluster_size: int = 4
    max_instances: int = 6

    cluster_eps_factor: float = 0.08

    min_bbox_width: int = 20
    min_bbox_height: int = 20
    max_bbox_width: int = 300
    max_bbox_height: int = 300

    contrast_threshold: float = 0.01
    edge_threshold: float = 20.0
    sigma: float = 1.2


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.copy()
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def create_sift(config: MatchConfig) -> cv2.SIFT:
    return cv2.SIFT_create(
        nfeatures=config.nfeatures,
        contrastThreshold=config.contrast_threshold,
        edgeThreshold=config.edge_threshold,
        sigma=config.sigma,
    )


def get_template_mask(template_image: np.ndarray) -> np.ndarray | None:
    if template_image.ndim == 3 and template_image.shape[2] == 4:
        alpha = template_image[:, :, 3]
        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        if np.count_nonzero(mask) > 0:
            return mask

    if template_image.ndim == 3:
        gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = template_image.copy()

    mask = np.where(gray < 245, 255, 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def enhance_template(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    enhanced = gray.copy()

    edges = cv2.Canny(gray, 60, 160)
    enhanced = cv2.addWeighted(enhanced, 0.8, edges, 0.6, 0)

    if mask is not None:
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    return enhanced


def detect_and_describe(
    image: np.ndarray,
    sift: cv2.SIFT,
    mask: np.ndarray | None = None,
    template_mode: bool = False,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = preprocess_gray(image)

    if template_mode:
        gray = enhance_template(gray, mask)

    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors


def get_good_matches(
    template_descriptors: np.ndarray,
    scene_descriptors: np.ndarray,
    config: MatchConfig,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(template_descriptors, scene_descriptors, k=2)

    good_matches: list[cv2.DMatch] = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        first, second = pair
        if first.distance < config.ratio_test * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda match: match.distance)
    return good_matches


def cluster_scene_points(points: np.ndarray, eps: float) -> list[np.ndarray]:
    if len(points) == 0:
        return []

    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []

    for i in range(len(points)):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster_indices = []

        while queue:
            current = queue.pop()
            cluster_indices.append(current)

            distances = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where((distances <= eps) & (~visited))[0]

            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(int(neighbor))

        clusters.append(np.array(cluster_indices, dtype=int))

    clusters.sort(key=len, reverse=True)
    return clusters


def build_instance(
    cluster_points: np.ndarray,
    config: MatchConfig,
) -> DetectedInstance | None:
    if len(cluster_points) < config.min_cluster_size:
        return None

    x, y, w, h = cv2.boundingRect(cluster_points.reshape(-1, 1, 2).astype(np.float32))

    if w < config.min_bbox_width or h < config.min_bbox_height:
        return None
    if w > config.max_bbox_width or h > config.max_bbox_height:
        return None

    center_x = int(np.mean(cluster_points[:, 0]))
    center_y = int(np.mean(cluster_points[:, 1]))

    return DetectedInstance(
        bbox=(x, y, w, h),
        center=(center_x, center_y),
        good_match_count=len(cluster_points),
        matched_points=cluster_points,
    )


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)


def suppress_overlapping_instances(
    instances: list[DetectedInstance],
    iou_threshold: float = 0.3,
) -> list[DetectedInstance]:
    kept: list[DetectedInstance] = []

    sorted_instances = sorted(
        instances,
        key=lambda item: item.good_match_count,
        reverse=True,
    )

    for instance in sorted_instances:
        overlaps = any(
            compute_iou(instance.bbox, kept_instance.bbox) > iou_threshold
            for kept_instance in kept
        )
        if not overlaps:
            kept.append(instance)

    return kept


def detect_template_instances(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    config: MatchConfig | None = None,
) -> tuple[list[DetectedInstance], list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    if config is None:
        config = MatchConfig()

    sift = create_sift(config)

    if template_image.ndim == 3 and template_image.shape[2] == 4:
        template_bgr = cv2.cvtColor(template_image, cv2.COLOR_BGRA2BGR)
    else:
        template_bgr = template_image.copy()

    template_mask = get_template_mask(template_image)

    template_keypoints, template_descriptors = detect_and_describe(
        template_bgr,
        sift,
        template_mask,
        template_mode=True,
    )
    scene_keypoints, scene_descriptors = detect_and_describe(
        scene_image,
        sift,
        None,
        template_mode=False,
    )

    if template_descriptors is None or scene_descriptors is None:
        return [], template_keypoints, scene_keypoints

    good_matches = get_good_matches(template_descriptors, scene_descriptors, config)

    if len(good_matches) < config.min_good_matches:
        return [], template_keypoints, scene_keypoints

    scene_points = np.array(
        [scene_keypoints[m.trainIdx].pt for m in good_matches],
        dtype=np.float32,
    )

    scene_height, scene_width = scene_image.shape[:2]
    scene_diagonal = float(np.hypot(scene_width, scene_height))
    cluster_eps = config.cluster_eps_factor * scene_diagonal

    clusters = cluster_scene_points(scene_points, cluster_eps)

    instances: list[DetectedInstance] = []

    for cluster_indices in clusters:
        cluster_points = scene_points[cluster_indices]
        instance = build_instance(cluster_points, config)
        if instance is not None:
            instances.append(instance)

    instances = suppress_overlapping_instances(instances)
    instances = instances[: config.max_instances]

    return instances, template_keypoints, scene_keypoints