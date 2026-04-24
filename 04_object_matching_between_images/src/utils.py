"""
utils.py
--------
Image loading, preprocessing, and visualization helpers.
"""

import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def load_bgr(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Image not found: '{path}'\nExpected at: {os.path.abspath(path)}"
        )

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"cv2 could not decode: '{path}'")

    return image


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def enhance_contrast(
    gray: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid: tuple = (4, 4),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def draw_keypoints(bgr: np.ndarray, keypoints: list) -> np.ndarray:
    return cv2.drawKeypoints(
        bgr,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_matches_img(
    img_query: np.ndarray,
    kp_query: list,
    img_scene: np.ndarray,
    kp_scene: list,
    good_matches: list,
    max_draw: int = 25,
) -> np.ndarray:
    return cv2.drawMatches(
        img_query,
        kp_query,
        img_scene,
        kp_scene,
        good_matches[:max_draw],
        None,
        matchColor=(0, 255, 100),
        singlePointColor=(200, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_detections(
    bgr: np.ndarray,
    match_pts: list,
    cluster_centers: list,
    wand_count: int,
) -> np.ndarray:
    result = bgr.copy()
    h, w = result.shape[:2]
    radius = int(min(h, w) * 0.09)

    for pt in match_pts:
        cv2.circle(result, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)

    for center in cluster_centers:
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 3)
        cv2.circle(result, (cx, cy), radius + 4, (255, 255, 0), 1)

    label = f"Wands detected: {wand_count}"
    (text_width, text_height), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )
    cv2.rectangle(result, (0, 0), (text_width + 16, text_height + 16), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (8, text_height + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 80),
        2,
    )

    return result


def build_figure(
    wand_bgr: np.ndarray,
    kp_wand: list,
    scene_results: list,
) -> plt.Figure:
    n = len(scene_results)
    fig = plt.figure(figsize=(22, 6 * (n + 1)))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        n + 1,
        3,
        figure=fig,
        hspace=0.4,
        wspace=0.08,
        top=0.95,
        bottom=0.03,
    )

    fig.suptitle(
        "SIFT Wand Detection - Harry Potter Scenes",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    ax_ref = fig.add_subplot(gs[0, :])
    wand_kp_img = draw_keypoints(wand_bgr, kp_wand)
    ax_ref.imshow(cv2.cvtColor(wand_kp_img, cv2.COLOR_BGR2RGB))
    ax_ref.set_title(
        f"Reference Wand  |  {len(kp_wand)} SIFT keypoints detected",
        color="lime",
        fontsize=12,
        fontweight="bold",
    )
    ax_ref.axis("off")

    for i, result in enumerate(scene_results):
        row = i + 1

        ax0 = fig.add_subplot(gs[row, 0])
        scene_kp_img = cv2.drawKeypoints(
            result["bgr"],
            result["kp_scene"][:300],
            None,
            color=(255, 200, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        ax0.imshow(cv2.cvtColor(scene_kp_img, cv2.COLOR_BGR2RGB))
        ax0.set_title(
            f"{result['title']}\n{len(result['kp_scene'])} keypoints",
            color="gold",
            fontsize=8,
        )
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[row, 1])
        match_img = draw_matches_img(
            wand_bgr,
            kp_wand,
            result["bgr"],
            result["kp_scene"],
            result["good"],
        )
        ax1.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(
            f"{len(result['good'])} good matches",
            color="cyan",
            fontsize=8,
        )
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[row, 2])
        detection_img = draw_detections(
            result["bgr"],
            result["match_pts"],
            result["centers"],
            result["count"],
        )
        ax2.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
        title_color = "lime" if result["count"] > 0 else "#aaaaaa"
        ax2.set_title(
            f"-> {result['count']} wand(s) detected",
            color=title_color,
            fontsize=9,
            fontweight="bold",
        )
        ax2.axis("off")

    return fig


def print_summary(scene_results: list) -> None:
    total = sum(result["count"] for result in scene_results)
    width = 30

    print("\n" + "=" * 64)
    print("SIFT WAND DETECTION - FINAL RESULTS")
    print("=" * 64)
    print(f"{'Scene':<{width}} {'KP':>6} {'Matches':>8} {'Wands':>6}")
    print("-" * 64)

    for result in scene_results:
        print(
            f"{result['title']:<{width}} "
            f"{len(result['kp_scene']):>6} "
            f"{len(result['good']):>8} "
            f"{result['count']:>6}"
        )

    print("-" * 64)
    print(f"{'TOTAL WANDS DETECTED':>{width + 22}} {total:>6}")
    print("=" * 64)