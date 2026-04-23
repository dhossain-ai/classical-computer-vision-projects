import argparse
from pathlib import Path

import cv2

from .detector import DetectorConfig, detect_text_blocks
from .utils import draw_boxes, draw_title, ensure_output_dir, load_image, save_image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect text blocks in an image using MSER.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/text.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/text_detected.jpg",
        help="Path to save output image",
    )
    parser.add_argument(
        "--mask-output",
        type=str,
        default="output/text_mask.jpg",
        help="Path to save intermediate text mask",
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
    text_boxes, text_mask = detect_text_blocks(image, config)

    annotated = draw_boxes(image, text_boxes, color=(0, 255, 0), thickness=2)
    annotated = draw_title(annotated, f"Detected text regions: {len(text_boxes)}")

    save_image(output_path, annotated)
    save_image(mask_output_path, text_mask)

    print(f"Input image: {input_path}")
    print(f"Detected text regions: {len(text_boxes)}")
    print(f"Annotated output saved to: {output_path}")
    print(f"Mask output saved to: {mask_output_path}")

    if args.show:
        cv2.imshow("Detected Text Regions", annotated)
        cv2.imshow("Text Mask", text_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()