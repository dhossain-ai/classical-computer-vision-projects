import argparse
from pathlib import Path

import cv2

from .detector import DetectorConfig, detect_sign_regions
from .utils import draw_boxes, draw_title, ensure_output_dir, load_image, save_image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect logos or road signs in an image using MSER.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/road_signs.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/sign_detected.jpg",
        help="Path to save output image",
    )
    parser.add_argument(
        "--mask-output",
        type=str,
        default="output/sign_mask.jpg",
        help="Path to save intermediate mask image",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)
    mask_output_path = Path(args.mask_output)

    ensure_output_dir(output_path)
    ensure_output_dir(mask_output_path)

    image = load_image(input_path)

    config = DetectorConfig()
    boxes, object_mask = detect_sign_regions(image, config)

    annotated = draw_boxes(image, boxes, color=(0, 255, 0), thickness=2)
    annotated = draw_title(annotated, f"Detected sign regions: {len(boxes)}")

    save_image(output_path, annotated)
    save_image(mask_output_path, object_mask)

    print(f"Input image: {input_path}")
    print(f"Detected sign regions: {len(boxes)}")
    print(f"Annotated output saved to: {output_path}")
    print(f"Mask output saved to: {mask_output_path}")

    if args.show:
        cv2.imshow("Detected Sign Regions", annotated)
        cv2.imshow("Object Mask", object_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()