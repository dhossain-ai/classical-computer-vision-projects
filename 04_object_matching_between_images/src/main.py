"""
main.py
-------
Entry point for SIFT-based wand detection.

Usage
-----
    python src/main.py
    python src/main.py --data data
    python src/main.py --ratio 0.78
    python src/main.py --min-match 3

Output
------
    output/wand_detection_result.png
"""

import argparse
import os
import sys

# Allow running as `python src/main.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matcher import (
    build_sift,
    build_flann,
    build_template_mask,
    enhance_template,
    compute_keypoints,
    lowe_ratio_match,
    cluster_matches,
)
from src.utils import (
    load_bgr,
    to_gray,
    enhance_contrast,
    build_figure,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and count wands in Harry Potter scenes using SIFT."
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Folder with images",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Folder for results",
    )
    parser.add_argument(
        "--wand",
        default="hp_wand.png",
        help="Reference wand filename inside --data",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "hp_scene1.jpg",
            "hp_scene2.jpg",
            "hp_scene3.jpg",
            "hp_scene4.jpg",
        ],
        help="Scene filenames inside --data",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.78,
        help="Lowe ratio threshold",
    )
    parser.add_argument(
        "--min-match",
        type=int,
        default=3,
        help="Minimum good matches required to attempt clustering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    sift = build_sift()
    flann = build_flann()

    wand_path = os.path.join(args.data, args.wand)
    wand_bgr = load_bgr(wand_path)
    wand_gray = enhance_contrast(to_gray(wand_bgr))

    wand_mask = build_template_mask(wand_gray)
    wand_gray = enhance_template(wand_gray, wand_mask)

    kp_wand, des_wand = compute_keypoints(sift, wand_gray, mask=wand_mask)
    print(f"[INFO] Reference wand : {wand_path}  ({len(kp_wand)} keypoints)")

    scene_results = []

    for filename in args.scenes:
        path = os.path.join(args.data, filename)
        name = os.path.splitext(filename)[0]

        try:
            scene_bgr = load_bgr(path)
        except FileNotFoundError as error:
            print(f"[WARN] {error} — skipping.")
            continue

        scene_gray = enhance_contrast(to_gray(scene_bgr))
        kp_scene, des_scene = compute_keypoints(sift, scene_gray)

        good = lowe_ratio_match(flann, des_wand, des_scene, ratio=args.ratio)

        count, centers = cluster_matches(
            good,
            kp_scene,
            scene_bgr.shape,
            min_match=args.min_match,
        )

        match_pts = (
            [kp_scene[m.trainIdx].pt for m in good]
            if len(good) >= args.min_match
            else []
        )

        print(
            f"[INFO] {name:<20}  kp={len(kp_scene):<6} "
            f"matches={len(good):<4}  wands={count}"
        )

        scene_results.append(
            {
                "title": name,
                "bgr": scene_bgr,
                "kp_scene": kp_scene,
                "good": good,
                "match_pts": match_pts,
                "centers": centers,
                "count": count,
            }
        )

    if not scene_results:
        print("[ERROR] No scenes processed. Check your --data folder.")
        sys.exit(1)

    print_summary(scene_results)

    fig = build_figure(wand_bgr, kp_wand, scene_results)
    out_path = os.path.join(args.output, "wand_detection_result.png")
    fig.savefig(
        out_path,
        dpi=140,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print(f"\n[INFO] Result saved -> {out_path}")


if __name__ == "__main__":
    main()