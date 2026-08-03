from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .shapes import COMPLETENESS, DIRECTIONS, SHARPNESS, ShapeLabels, generate_dataset
from .stimuli import poisson_encode

HEAD_CLASSES = {
    "direction": DIRECTIONS,
    "sharpness": SHARPNESS,
    "completeness": COMPLETENESS,
}


def _pool4(array: np.ndarray) -> np.ndarray:
    return array.reshape(4, 4, 4, 4).mean(axis=(1, 3))


def spike_features(spikes: np.ndarray) -> np.ndarray:
    """Extract spatial and orientation energy from retina spike trains only."""
    rate_map = spikes.mean(axis=0).reshape(16, 16)
    gx = np.zeros_like(rate_map)
    gy = np.zeros_like(rate_map)
    gx[:, 1:-1] = rate_map[:, 2:] - rate_map[:, :-2]
    gy[1:-1, :] = rate_map[2:, :] - rate_map[:-2, :]
    diag_a = (gx + gy) / np.sqrt(2)
    diag_b = (gx - gy) / np.sqrt(2)
    channels = [rate_map, np.abs(gx), np.abs(gy), np.abs(diag_a), np.abs(diag_b)]
    return np.concatenate([_pool4(c).reshape(-1) for c in channels]).astype(np.float32)


def image_features(image: np.ndarray, seed: int) -> np.ndarray:
    spikes = poisson_encode(image, duration_ms=300, dt_ms=1,
                            max_rate_hz=180, seed=seed)
    return spike_features(spikes)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class DecoderPrediction:
    direction: str
    sharpness: str
    completeness: str
    confidence: dict[str, float]


class MultiTaskDecoder:
    """Small trainable MLP with three semantic output heads."""

    def __init__(self, input_size: int = 80, hidden_size: int = 36, seed: int = 9):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.heads = {
            name: [rng.normal(0, .12, (hidden_size, len(classes))).astype(np.float32),
                   np.zeros(len(classes), dtype=np.float32)]
            for name, classes in HEAD_CLASSES.items()
        }
        self.mean = np.zeros(input_size, dtype=np.float32)
        self.std = np.ones(input_size, dtype=np.float32)
        self.trained = False
        self.metrics = {}

    def _indices(self, labels: list[ShapeLabels], name: str) -> np.ndarray:
        classes = HEAD_CLASSES[name]
        return np.asarray([classes.index(getattr(label, name)) for label in labels])

    def fit(self, features: np.ndarray, labels: list[ShapeLabels], epochs: int = 220,
            learning_rate: float = .035) -> None:
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-5
        x = (features - self.mean) / self.std
        n = len(x)
        targets = {name: self._indices(labels, name) for name in HEAD_CLASSES}

        for _ in range(epochs):
            hidden_pre = x @ self.w1 + self.b1
            hidden = np.maximum(hidden_pre, 0)
            grad_hidden = np.zeros_like(hidden)
            for name, (weight, bias) in self.heads.items():
                probabilities = _softmax(hidden @ weight + bias)
                grad = probabilities
                grad[np.arange(n), targets[name]] -= 1
                grad /= n
                grad_hidden += grad @ weight.T
                self.heads[name][0] -= learning_rate * (hidden.T @ grad)
                self.heads[name][1] -= learning_rate * grad.sum(axis=0)
            grad_hidden[hidden_pre <= 0] = 0
            self.w1 -= learning_rate * (x.T @ grad_hidden)
            self.b1 -= learning_rate * grad_hidden.sum(axis=0)
        self.trained = True

    def probabilities(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = (np.atleast_2d(features) - self.mean) / self.std
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        return {name: _softmax(hidden @ w + b) for name, (w, b) in self.heads.items()}

    def predict(self, features: np.ndarray) -> DecoderPrediction:
        probabilities = self.probabilities(features)
        chosen = {name: HEAD_CLASSES[name][int(p[0].argmax())] for name, p in probabilities.items()}
        confidence = {name: float(p[0].max()) for name, p in probabilities.items()}
        return DecoderPrediction(**chosen, confidence=confidence)


def train_default_decoder(samples: int = 1600) -> MultiTaskDecoder:
    images, labels = generate_dataset(samples=samples)
    split = int(samples * .8)
    features = np.stack([image_features(image, 10000 + i) for i, image in enumerate(images)])
    decoder = MultiTaskDecoder()
    decoder.fit(features[:split], labels[:split])
    probabilities = decoder.probabilities(features[split:])
    decoder.metrics = {
        name: float((p.argmax(axis=1) == decoder._indices(labels[split:], name)).mean())
        for name, p in probabilities.items()
    }
    return decoder


DECODER = None


def get_decoder() -> MultiTaskDecoder:
    global DECODER
    if DECODER is None:
        DECODER = train_default_decoder()
    return DECODER


def decode_image(image: np.ndarray, seed: int = 42):
    spikes = poisson_encode(image, 300, 1, 180, seed)
    decoder = get_decoder()
    return decoder.predict(spike_features(spikes)), spikes, decoder.metrics

