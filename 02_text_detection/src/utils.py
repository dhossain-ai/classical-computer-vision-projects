from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def save_image(image_path: str | Path, image: np.ndarray) -> None:
    output_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Could not save image: {image_path}")


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()

    for x, y, w, h in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

    return output


def draw_title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()

    cv2.putText(
        output,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def ensure_output_dir(output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)