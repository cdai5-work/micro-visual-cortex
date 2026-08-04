from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .runtime import get_model
from .stimuli import poisson_encode


@dataclass(frozen=True)
class ModalitySignal:
    name: str
    spikes: np.ndarray

    def __post_init__(self):
        if self.spikes.ndim != 2:
            raise ValueError(f"{self.name}信号必须为二维脉冲矩阵")


@dataclass
class MultimodalSignalBundle:
    signals: dict[str, ModalitySignal] = field(default_factory=dict)

    def add(self, signal: ModalitySignal) -> None:
        self.signals[signal.name] = signal

    def require(self, name: str) -> ModalitySignal:
        if name not in self.signals:
            raise ValueError(f"缺少模态信号: {name}")
        return self.signals[name]


def scalar_poisson_signal(name: str, values, duration_ms: int, seed: int,
                          max_rate_hz: float = 500.0) -> ModalitySignal:
    values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    probability = values * max_rate_hz / 1000.0
    spikes = np.random.default_rng(seed).random((duration_ms, len(values))) < probability
    return ModalitySignal(name, spikes.astype(np.float32))


def encode_multimodal(image: np.ndarray, pain: float, metallic: float, seed: int,
                      duration_ms: int = 200):
    retina_spikes = poisson_encode(image, duration_ms, 1, 180, seed)
    model, fallback = get_model()
    v1_spikes, voltages, elapsed = model.run(retina_spikes, 1.0)
    bundle = MultimodalSignalBundle()
    bundle.add(ModalitySignal("vision_v1", v1_spikes))
    bundle.add(scalar_poisson_signal(
        "touch", [pain] * 8 + [1 - pain] * 8, duration_ms, seed + 1
    ))
    bundle.add(scalar_poisson_signal(
        "odor", [metallic] * 8 + [1 - metallic] * 8, duration_ms, seed + 2
    ))
    metadata = {"backend": model.backend_name, "fallback_reason": fallback, "elapsed_ms": elapsed}
    return bundle, retina_spikes, voltages, metadata
