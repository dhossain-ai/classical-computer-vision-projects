import cv2
import numpy as np


def build_sift(
    contrast_threshold: float = 0.01,
    n_octave_layers: int = 5,
    edge_threshold: int = 10,
    sigma: float = 1.6,
) -> cv2.SIFT:
    return cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )


def build_flann() -> cv2.FlannBasedMatcher:
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    return cv2.FlannBasedMatcher(index_params, search_params)


def build_template_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.where(gray < 245, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def enhance_template(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 60, 160)
    enhanced = cv2.addWeighted(gray, 0.8, edges, 0.6, 0)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    return enhanced


def compute_keypoints(
    sift: cv2.SIFT,
    gray: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[list, np.ndarray]:
    kp, des = sift.detectAndCompute(gray, mask)
    return kp, des


def lowe_ratio_match(
    flann: cv2.FlannBasedMatcher,
    des_query: np.ndarray,
    des_scene: np.ndarray,
    ratio: float = 0.78,
) -> list:
    if des_query is None or des_scene is None:
        return []
    if len(des_query) < 2 or len(des_scene) < 2:
        return []

    try:
        raw_matches = flann.knnMatch(des_query, des_scene, k=2)
    except cv2.error:
        return []

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    good.sort(key=lambda match: match.distance)
    return good


def cluster_matches(
    good_matches: list,
    kp_scene: list,
    scene_shape: tuple,
    cluster_dist_ratio: float = 0.10,
    min_match: int = 3,
) -> tuple[int, list]:
    if len(good_matches) < min_match:
        return 0, []

    pts = np.array([kp_scene[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    h, w = scene_shape[:2]
    thresh = np.hypot(h, w) * cluster_dist_ratio

    visited = np.zeros(len(pts), dtype=bool)
    centers = []

    for i in range(len(pts)):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster_indices = []

        while queue:
            current = queue.pop()
            cluster_indices.append(current)

            distances = np.linalg.norm(pts - pts[current], axis=1)
            neighbors = np.where((distances <= thresh) & (~visited))[0]

            for neighbor in neighbors:
                visited[neighbor] = True
                queue.append(int(neighbor))

        cluster = pts[cluster_indices]

        if len(cluster) < min_match:
            continue

        x, y, bw, bh = cv2.boundingRect(cluster.reshape(-1, 1, 2))

        if bw < 12 or bh < 12:
            continue
        if bw > 300 or bh > 300:
            continue

        centers.append(np.mean(cluster, axis=0))

    return len(centers), centers