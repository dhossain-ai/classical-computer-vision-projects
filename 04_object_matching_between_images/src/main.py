import argparse
from pathlib import Path

import cv2

from .matcher import MatchConfig, detect_template_instances
from .utils import (
    build_match_visualization,
    draw_instances,
    draw_title,
    ensure_output_dir,
    load_image,
    save_image,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match an object between two images using SIFT.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="data/hp_wand.png",
        help="Path to template image",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="data/hp_scene1.jpg",
        help="Path to scene image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/hp_scene1_detected.jpg",
        help="Path to save annotated scene image",
    )
    parser.add_argument(
        "--matches-output",
        type=str,
        default="output/hp_scene1_matches.jpg",
        help="Path to save keypoint match visualization",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show output windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    template_path = Path(args.template)
    scene_path = Path(args.scene)
    output_path = Path(args.output)
    matches_output_path = Path(args.matches_output)

    ensure_output_dir(output_path)
    ensure_output_dir(matches_output_path)

    template_image = load_image(template_path, unchanged=True)
    scene_image = load_image(scene_path)

    config = MatchConfig()
    instances, template_keypoints, scene_keypoints = detect_template_instances(
        template_image,
        scene_image,
        config,
    )

    annotated_scene = draw_instances(scene_image, instances)
    annotated_scene = draw_title(annotated_scene, f"Detected wand count: {len(instances)}")

    best_instance = max(instances, key=lambda item: item.inlier_count, default=None)
    match_visualization = build_match_visualization(
        template_image,
        scene_image,
        template_keypoints,
        scene_keypoints,
        best_instance,
    )

    save_image(output_path, annotated_scene)

    if match_visualization is not None:
        save_image(matches_output_path, match_visualization)

    print(f"Template image: {template_path}")
    print(f"Scene image: {scene_path}")
    print(f"Detected wand count: {len(instances)}")

    if instances:
        for index, instance in enumerate(instances, start=1):
            print(
                f"Instance {index}: good_matches={instance.good_match_count}, "
                f"inliers={instance.inlier_count}, bbox={instance.bbox}"
            )
    else:
        print("No strong wand instance was detected.")

    print(f"Annotated scene saved to: {output_path}")
    if match_visualization is not None:
        print(f"Match visualization saved to: {matches_output_path}")

    if args.show:
        cv2.imshow("Detected Wand Instances", annotated_scene)
        if match_visualization is not None:
            cv2.imshow("Best Match Visualization", match_visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()