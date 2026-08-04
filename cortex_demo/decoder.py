from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .multimodal import MultimodalSignalBundle, encode_multimodal
from .shapes import COMPLETENESS, DIRECTIONS, SHARPNESS, generate_dataset

HEAD_CLASSES = {
    "direction": DIRECTIONS,
    "sharpness": SHARPNESS,
    "completeness": COMPLETENESS,
    "area": ("small", "large"),
    "force": ("gentle", "strong"),
    "pain": ("no_pain", "pain"),
    "hardness": ("soft", "hard"),
    "metallic": ("not_metallic", "metallic"),
    "danger": ("safer", "danger"),
}


def _temporal_features(spikes: np.ndarray) -> np.ndarray:
    parts = np.array_split(spikes, 3, axis=0)
    return np.concatenate([spikes.mean(axis=0)] + [part.mean(axis=0) for part in parts])


def bundle_features(bundle: MultimodalSignalBundle) -> np.ndarray:
    """Fuse actual V1, touch and odor spike signals through a stable modality interface."""
    visual = bundle.require("vision_v1").spikes
    if visual.shape[1] != 64:
        raise ValueError("空间化V1必须输出64个神经元通道")
    touch = bundle.require("touch").spikes.mean(axis=0)
    if touch.shape[0] != 64:
        raise ValueError("触觉模块必须输出64个通道（四类变量各16通道）")
    odor = bundle.require("odor").spikes.mean(axis=0)
    return np.concatenate([_temporal_features(visual), touch, odor]).astype(np.float32)


def spike_activity_maps(spikes: np.ndarray) -> dict[str, np.ndarray]:
    """Expose the four retinotopic orientation maps encoded by V1 spikes."""
    if spikes.ndim != 2 or spikes.shape[1] != 64:
        raise ValueError("V1脉冲矩阵必须具有形状 (时间步, 64)")
    grouped = spikes.mean(axis=0).reshape(4, 4, 4)
    return {
        "horizontal": grouped[0],
        "diagonal_left": grouped[1],
        "vertical": grouped[2],
        "diagonal_right": grouped[3],
        "activity": grouped.mean(axis=0),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class SemanticLabels:
    direction: str
    sharpness: str
    completeness: str
    area: str
    force: str
    pain: str
    hardness: str
    metallic: str
    danger: str


@dataclass
class DecoderPrediction:
    direction: str
    sharpness: str
    completeness: str
    area: str
    force: str
    pain: str
    hardness: str
    metallic: str
    danger: str
    confidence: dict[str, float]


class MultiTaskDecoder:
    """Trainable fusion MLP accepting any modalities exposed through the signal bundle."""

    def __init__(self, input_size: int = 336, hidden_size: int = 80, seed: int = 9):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.heads = {
            name: [rng.normal(0, .1, (hidden_size, len(classes))).astype(np.float32),
                   np.zeros(len(classes), dtype=np.float32)]
            for name, classes in HEAD_CLASSES.items()
        }
        self.mean = np.zeros(input_size, dtype=np.float32)
        self.std = np.ones(input_size, dtype=np.float32)
        self.trained = False
        self.metrics = {}

    def _indices(self, labels: list[SemanticLabels], name: str) -> np.ndarray:
        classes = HEAD_CLASSES[name]
        return np.asarray([classes.index(getattr(label, name)) for label in labels])

    def fit(self, features: np.ndarray, labels: list[SemanticLabels], epochs: int = 380,
            learning_rate: float = .025) -> None:
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


def _danger_label(direction: str, sharpness: str, force: float, pain: float,
                  hardness: float, metallic: float) -> str:
    painful = pain >= .5
    impact_risk = force >= .65 and hardness >= .6
    risky_object_cue = sharpness == "sharp" and direction == "down" and (
        metallic >= .5 or hardness >= .6
    )
    return "danger" if painful or impact_risk or risky_object_cue else "safer"


def train_default_decoder(samples: int = 800) -> MultiTaskDecoder:
    images, shape_labels = generate_dataset(samples=samples, seed=2027)
    rng = np.random.default_rng(2028)
    features, labels = [], []
    for i, (image, shape_label) in enumerate(zip(images, shape_labels)):
        pain = float(rng.uniform(0, 1))
        area = float(rng.uniform(0, 1))
        force = float(rng.uniform(0, 1))
        hardness = float(rng.uniform(0, 1))
        metallic = float(rng.uniform(0, 1))
        bundle, _, _, _ = encode_multimodal(
            image, pain, metallic, 20000 + i, duration_ms=120,
            area=area, force=force, hardness=hardness,
        )
        features.append(bundle_features(bundle))
        labels.append(SemanticLabels(
            shape_label.direction, shape_label.sharpness, shape_label.completeness,
            "large" if area >= .5 else "small",
            "strong" if force >= .5 else "gentle",
            "pain" if pain >= .5 else "no_pain",
            "hard" if hardness >= .5 else "soft",
            "metallic" if metallic >= .5 else "not_metallic",
            _danger_label(shape_label.direction, shape_label.sharpness, force, pain,
                          hardness, metallic),
        ))
    features = np.stack(features)
    split = int(samples * .8)
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


def decode_multimodal(image: np.ndarray, pain: float, metallic: float, seed: int = 42,
                      area: float = .5, force: float = .5, hardness: float = .5):
    bundle, retina_spikes, voltages, metadata = encode_multimodal(
        image, pain, metallic, seed, duration_ms=200,
        area=area, force=force, hardness=hardness,
    )
    decoder = get_decoder()
    prediction = decoder.predict(bundle_features(bundle))
    return prediction, bundle, retina_spikes, voltages, decoder.metrics, metadata


def decode_image(image: np.ndarray, seed: int = 42):
    prediction, bundle, retina, _, metrics, _ = decode_multimodal(image, 0, 0, seed)
    return prediction, bundle.require("vision_v1").spikes, metrics
