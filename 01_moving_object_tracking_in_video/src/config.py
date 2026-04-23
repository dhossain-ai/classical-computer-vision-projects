from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class AppConfig:
    input_video: Path = DATA_DIR / "moving_car.mp4"
    output_video: Path = OUTPUT_DIR / "moving_car_result.mp4"

    show_preview: bool = True
    save_output: bool = True
    use_manual_roi: bool = True

    start_frame: int = 0

    max_corners: int = 200
    min_corner_distance: int = 10

    harris_block_size: int = 2
    harris_ksize: int = 3
    harris_k: float = 0.04
    harris_threshold_ratio: float = 0.01

    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    lk_epsilon: float = 0.01

    min_tracked_points: int = 20
    smoothing_window: int = 12
    roi_padding: int = 30

    fps_override: float | None = None

    # Use this only if you know an approximate scene scale.
    pixels_per_meter: float | None = None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)