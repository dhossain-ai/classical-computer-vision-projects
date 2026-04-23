from collections import deque

import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        smoothing_window: int,
        pixels_per_meter: float | None = None,
    ) -> None:
        self.history = deque(maxlen=smoothing_window)
        self.pixels_per_meter = pixels_per_meter

    def update(
        self,
        old_points: np.ndarray,
        new_points: np.ndarray,
        fps: float,
    ) -> dict[str, float]:
        if old_points is not None and new_points is not None:
            if len(old_points) > 0 and len(new_points) > 0 and fps > 0:
                old_xy = old_points.reshape(-1, 2)
                new_xy = new_points.reshape(-1, 2)

                displacements = np.linalg.norm(new_xy - old_xy, axis=1)

                if displacements.size > 0:
                    raw_pixels_per_second = float(np.median(displacements) * fps)
                    self.history.append(raw_pixels_per_second)

        return self.current()

    def current(self) -> dict[str, float]:
        if self.history:
            pixels_per_second = float(np.mean(self.history))
        else:
            pixels_per_second = 0.0

        result = {
            "pixels_per_second": pixels_per_second,
        }

        if self.pixels_per_meter is not None and self.pixels_per_meter > 0:
            meters_per_second = pixels_per_second / self.pixels_per_meter
            result["meters_per_second"] = meters_per_second
            result["km_per_hour"] = meters_per_second * 3.6

        return result