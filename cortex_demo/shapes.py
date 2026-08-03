from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .settings import IMAGE_SIZE

DIRECTIONS = ("up", "down", "left", "right")
SHARPNESS = ("sharp", "rounded")
COMPLETENESS = ("complete", "open")


@dataclass(frozen=True)
class ShapeLabels:
    direction: str
    sharpness: str
    completeness: str


def _polygon_mask(vertices: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    x, y = xx + 0.5, yy + 0.5
    inside = np.zeros((size, size), dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi
        )
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def _rotate_from_up(image: np.ndarray, direction: str) -> np.ndarray:
    turns = {"up": 0, "left": 1, "down": 2, "right": 3}[direction]
    return np.rot90(image, turns)


def generate_shape(direction: str = "up", sharpness: str = "sharp",
                   completeness: str = "complete", brightness: float = 0.9,
                   noise: float = 0.04, seed: int = 0) -> np.ndarray:
    if direction not in DIRECTIONS:
        raise ValueError(f"未知方向: {direction}")
    if sharpness not in SHARPNESS:
        raise ValueError(f"未知边缘形态: {sharpness}")
    if completeness not in COMPLETENESS:
        raise ValueError(f"未知轮廓状态: {completeness}")

    rng = np.random.default_rng(seed)
    shift_x, shift_y = rng.integers(-1, 2, size=2)
    if sharpness == "sharp":
        # Up-facing arrow: a narrow shaft and a triangular point.
        vertices = np.array([
            [8, 1], [14, 7], [11, 7], [11, 14],
            [5, 14], [5, 7], [2, 7],
        ], dtype=np.float32)
        mask = _polygon_mask(vertices)
    else:
        # A rounded teardrop has a spatial direction but no acute corners.
        yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
        body = ((xx - 7.5) / 4.8) ** 2 + ((yy - 9.0) / 5.0) ** 2 <= 1
        head = ((xx - 7.5) / 2.6) ** 2 + ((yy - 3.7) / 3.5) ** 2 <= 1
        mask = body | head

    outline = mask & ~_erode(mask)
    if completeness == "open":
        # Remove a reproducibly jittered section from the right side of the contour.
        yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
        gap_y = int(rng.integers(7, 11))
        gap = (xx >= 9) & (np.abs(yy - gap_y) <= 3)
        outline &= ~gap

    outline = _rotate_from_up(outline, direction)
    outline = np.roll(outline, (shift_y, shift_x), axis=(0, 1))
    image = outline.astype(np.float32) * brightness
    if noise:
        image += rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 1).astype(np.float32)


def generate_dataset(samples: int = 1600, seed: int = 2026):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for i in range(samples):
        direction = DIRECTIONS[i % len(DIRECTIONS)]
        sharpness = SHARPNESS[(i // len(DIRECTIONS)) % len(SHARPNESS)]
        completeness = COMPLETENESS[(i // (len(DIRECTIONS) * len(SHARPNESS))) % 2]
        images.append(generate_shape(
            direction, sharpness, completeness,
            brightness=float(rng.uniform(0.65, 1.0)),
            noise=float(rng.uniform(0.0, 0.12)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
        labels.append(ShapeLabels(direction, sharpness, completeness))
    order = rng.permutation(samples)
    return np.stack(images)[order], [labels[i] for i in order]
