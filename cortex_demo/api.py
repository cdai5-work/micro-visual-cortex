from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from . import settings
from .runtime import get_model
from .stimuli import generate_stimulus, poisson_encode


@dataclass(frozen=True)
class StimulusConfig:
    stimulus_type: str = "vertical"
    angle_deg: Optional[float] = None
    brightness: float = 0.8
    noise: float = 0.05
    duration_ms: int = settings.DEFAULT_DURATION_MS
    dt_ms: float = settings.DEFAULT_DT_MS
    max_rate_hz: float = settings.DEFAULT_MAX_RATE_HZ
    seed: int = settings.DEFAULT_SEED


@dataclass
class SimulationResult:
    stimulus: np.ndarray
    times_ms: np.ndarray
    input_spikes: np.ndarray
    v1_spikes: np.ndarray
    membrane_voltage: np.ndarray
    group_rates_hz: dict[int, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def simulate(config: StimulusConfig) -> SimulationResult:
    image = generate_stimulus(config.stimulus_type, config.brightness, config.noise,
                              config.angle_deg, config.seed)
    encoded = poisson_encode(image, config.duration_ms, config.dt_ms,
                             config.max_rate_hz, config.seed + 1)
    model, fallback_reason = get_model()
    spikes, voltage, elapsed = model.run(encoded, config.dt_ms)
    duration_s = config.duration_ms / 1000.0
    grouped = spikes.reshape(spikes.shape[0], 4, settings.NEURONS_PER_GROUP)
    rates = grouped.sum(axis=(0, 2)) / (settings.NEURONS_PER_GROUP * duration_s)
    return SimulationResult(
        stimulus=image,
        times_ms=np.arange(encoded.shape[0]) * config.dt_ms,
        input_spikes=encoded,
        v1_spikes=spikes,
        membrane_voltage=voltage,
        group_rates_hz={o: float(r) for o, r in zip(settings.ORIENTATIONS, rates)},
        metadata={
            "backend": model.backend_name,
            "fallback_reason": fallback_reason,
            "elapsed_ms": round(elapsed, 2),
            "seed": config.seed,
            "model_initializations": model.initialization_count,
            "scientific_scope": "教学性简化模型，不对应具体动物脑区数据",
        },
    )
