from __future__ import annotations

import numpy as np
from typing import Optional

from .settings import IMAGE_SIZE

VALID_STIMULI = {
    "spot", "horizontal", "vertical", "diagonal_left", "diagonal_right", "blank"
}


def generate_stimulus(kind: str, brightness: float = 0.8, noise: float = 0.05,
                      angle_deg: Optional[float] = None, seed: int = 42) -> np.ndarray:
    if kind not in VALID_STIMULI:
        raise ValueError(f"未知刺激类型: {kind}")
    if not 0 <= brightness <= 1 or not 0 <= noise <= 0.5:
        raise ValueError("亮度须在 0–1，噪声须在 0–0.5")

    axis = np.linspace(-1, 1, IMAGE_SIZE, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    if kind == "blank":
        image = np.zeros_like(x)
    elif kind == "spot":
        image = np.exp(-(x * x + y * y) / (2 * 0.16 ** 2))
    else:
        default_angles = {
            "horizontal": 0.0, "vertical": 90.0,
            "diagonal_left": 45.0, "diagonal_right": 135.0,
        }
        theta = np.deg2rad(default_angles[kind] if angle_deg is None else angle_deg)
        # Constant along the bar; varies across the normal direction.
        normal = -x * np.sin(theta) + y * np.cos(theta)
        image = (np.cos(2 * np.pi * 2.1 * normal) > 0.15).astype(np.float32)

    image *= brightness
    if noise:
        image += np.random.default_rng(seed).normal(0, noise, image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def poisson_encode(image: np.ndarray, duration_ms: int, dt_ms: float,
                   max_rate_hz: float, seed: int) -> np.ndarray:
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"刺激必须为 {IMAGE_SIZE}×{IMAGE_SIZE}")
    if duration_ms < 20 or duration_ms > 1000 or dt_ms <= 0:
        raise ValueError("仿真时长须在 20–1000 ms 且 dt > 0")
    steps = int(round(duration_ms / dt_ms))
    probability = np.clip(image.reshape(-1) * max_rate_hz * dt_ms / 1000.0, 0, 1)
    return (np.random.default_rng(seed).random((steps, image.size)) < probability).astype(np.float32)
