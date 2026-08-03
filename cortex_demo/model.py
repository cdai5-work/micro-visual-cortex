from __future__ import annotations

import time
import numpy as np

from . import settings


def _gabor_weights(seed: int = 7) -> np.ndarray:
    axis = np.linspace(-1, 1, settings.IMAGE_SIZE, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    groups = []
    rng = np.random.default_rng(seed)
    for orientation in settings.ORIENTATIONS:
        theta = np.deg2rad(orientation)
        normal = -x * np.sin(theta) + y * np.cos(theta)
        envelope = np.exp(-(x * x + y * y) / (2 * 0.62 ** 2))
        base = envelope * (np.cos(2 * np.pi * 2.1 * normal) + 1.0) / 2.0
        base = base.reshape(-1)
        base /= base.sum() + 1e-8
        members = [np.clip(base * rng.normal(1.0, 0.035, base.shape), 0, None)
                   for _ in range(settings.NEURONS_PER_GROUP)]
        groups.append(np.stack(members))
    return np.concatenate(groups).astype(np.float32)


class VisualCortexModel:
    """Deterministic-weight, stochastic-input LIF reference model.

    The array interface mirrors a JAX implementation, so replacing np with jax.numpy
    and scanning the time step is isolated from the public simulation API.
    """

    def __init__(self) -> None:
        self.weights = _gabor_weights()
        self.initialization_count = 1
        self.backend_name = "NumPy reference (CPU fallback)"

    def run(self, input_spikes: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray, float]:
        started = time.perf_counter()
        steps = input_spikes.shape[0]
        total = len(settings.ORIENTATIONS) * settings.NEURONS_PER_GROUP
        v = np.full(total, settings.V_REST, dtype=np.float32)
        refractory = np.zeros(total, dtype=np.int16)
        spikes = np.zeros((steps, total), dtype=np.float32)
        voltages = np.zeros((steps, total), dtype=np.float32)
        previous_group_rate = np.zeros(len(settings.ORIENTATIONS), dtype=np.float32)

        for t in range(steps):
            drive = self.weights @ input_spikes[t] * settings.GAIN * 100.0
            inhibition = np.repeat(
                previous_group_rate.sum() - previous_group_rate,
                settings.NEURONS_PER_GROUP,
            ) * settings.LATERAL_INHIBITION
            active = refractory <= 0
            dv = ((settings.V_REST - v) + drive - inhibition) * dt_ms / settings.TAU_MEMBRANE_MS
            v[active] += dv[active]
            fired = active & (v >= settings.V_THRESHOLD)
            spikes[t, fired] = 1.0
            v[fired] = settings.V_RESET
            refractory[active] = 0
            refractory[~active] -= 1
            refractory[fired] = settings.REFRACTORY_MS
            voltages[t] = v
            previous_group_rate = spikes[t].reshape(4, settings.NEURONS_PER_GROUP).mean(axis=1)
        return spikes, voltages, (time.perf_counter() - started) * 1000


MODEL = VisualCortexModel()
