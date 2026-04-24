from pathlib import Path

import cv2
import numpy as np

from .matcher import DetectedInstance


def load_image(image_path: str | Path, unchanged: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


def save_image(image_path: str | Path, image: np.ndarray) -> None:
    output_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Could not save image: {image_path}")


def ensure_output_dir(output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def draw_instances(
    scene_image: np.ndarray,
    instances: list[DetectedInstance],
) -> np.ndarray:
    output = scene_image.copy()

    for index, instance in enumerate(instances, start=1):
        x, y, w, h = instance.bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cx, cy = instance.center
        cv2.circle(output, (cx, cy), 5, (255, 200, 0), -1)

        label = f"Wand {index}"
        cv2.putText(
            output,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_title(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()

    cv2.putText(
        output,
        text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        text,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def build_match_visualization(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: list[cv2.KeyPoint],
    scene_keypoints: list[cv2.KeyPoint],
    instance: DetectedInstance | None,
) -> np.ndarray | None:
    return None