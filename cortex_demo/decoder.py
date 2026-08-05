from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .multimodal import ModalitySignal, MultimodalSignalBundle, encode_multimodal
from .shapes import COMPLETENESS, DIRECTIONS, SHARPNESS, generate_dataset

HEAD_CLASSES = {
    "direction": DIRECTIONS, "sharpness": SHARPNESS, "completeness": COMPLETENESS,
    "area": ("small", "large"), "force": ("gentle", "strong"),
    "pain": ("no_pain", "pain"), "hardness": ("soft", "hard"),
    "metallic": ("not_metallic", "metallic"), "danger": ("safer", "danger"),
}
VISUAL_HEADS = ("direction", "sharpness", "completeness")
TOUCH_HEADS = ("area", "force", "pain", "hardness")
ODOR_HEADS = ("metallic",)
MODALITIES = ("vision", "touch", "odor")
FEATURE_SLICES = {"vision": slice(0, 256), "touch": slice(256, 320), "odor": slice(320, 336)}


def _temporal_features(spikes: np.ndarray) -> np.ndarray:
    parts = np.array_split(spikes, 3, axis=0)
    return np.concatenate([spikes.mean(axis=0)] + [part.mean(axis=0) for part in parts])


def bundle_features(bundle: MultimodalSignalBundle) -> np.ndarray:
    visual = bundle.require("vision_v1").spikes
    touch = bundle.require("touch").spikes.mean(axis=0)
    odor = bundle.require("odor").spikes.mean(axis=0)
    if visual.shape[1] != 64:
        raise ValueError("空间化V1必须输出64个神经元通道")
    if touch.shape[0] != 64:
        raise ValueError("触觉模块必须输出64个通道（四类变量各16通道）")
    if odor.shape[0] != 16:
        raise ValueError("嗅觉模块必须输出16个通道")
    return np.concatenate([_temporal_features(visual), touch, odor]).astype(np.float32)


def spike_activity_maps(spikes: np.ndarray) -> dict[str, np.ndarray]:
    if spikes.ndim != 2 or spikes.shape[1] != 64:
        raise ValueError("V1脉冲矩阵必须具有形状 (时间步, 64)")
    grouped = spikes.mean(axis=0).reshape(4, 4, 4)
    return {"horizontal": grouped[0], "diagonal_left": grouped[1],
            "vertical": grouped[2], "diagonal_right": grouped[3],
            "activity": grouped.mean(axis=0)}


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
    modality_weights: dict[str, float]


class MultiTaskDecoder:
    """Three isolated encoders with a learned softmax gate for the danger head."""

    def __init__(self, embedding_size: int = 32, seed: int = 9):
        rng = np.random.default_rng(seed)
        self.embedding_size = embedding_size
        self.encoders = {}
        for name, size in (("vision", 256), ("touch", 64), ("odor", 16)):
            self.encoders[name] = [
                rng.normal(0, np.sqrt(2 / size), (size, embedding_size)).astype(np.float32),
                np.zeros(embedding_size, dtype=np.float32),
            ]
        self.gate = [rng.normal(0, .08, (embedding_size * 3, 3)).astype(np.float32),
                     np.zeros(3, dtype=np.float32)]
        source_heads = {**{h: "vision" for h in VISUAL_HEADS},
                        **{h: "touch" for h in TOUCH_HEADS},
                        **{h: "odor" for h in ODOR_HEADS}}
        self.head_sources = source_heads
        self.heads = {
            name: [rng.normal(0, .1, (embedding_size, len(classes))).astype(np.float32),
                   np.zeros(len(classes), dtype=np.float32)]
            for name, classes in HEAD_CLASSES.items()
        }
        self.mean = np.zeros(336, dtype=np.float32)
        self.std = np.ones(336, dtype=np.float32)
        self.trained = False
        self.metrics = {}

    def _indices(self, labels: list[SemanticLabels], name: str) -> np.ndarray:
        return np.asarray([HEAD_CLASSES[name].index(getattr(label, name)) for label in labels])

    def _forward(self, features: np.ndarray):
        raw = np.atleast_2d(features)
        x = (raw - self.mean) / self.std
        pre, z, presence = {}, {}, {}
        for name in MODALITIES:
            w, b = self.encoders[name]
            present = np.any(np.abs(raw[:, FEATURE_SLICES[name]]) > 1e-8, axis=1, keepdims=True)
            presence[name] = present
            pre[name] = x[:, FEATURE_SLICES[name]] @ w + b
            pre[name] = np.where(present, pre[name], -1.0)
            z[name] = np.maximum(pre[name], 0)
        joined = np.concatenate([z[name] for name in MODALITIES], axis=1)
        gate_logits = joined @ self.gate[0] + self.gate[1]
        availability = np.concatenate([presence[name] for name in MODALITIES], axis=1)
        gate_logits = np.where(availability, gate_logits, -20.0)
        gates = _softmax(gate_logits)
        stacked = np.stack([z[name] for name in MODALITIES], axis=1)
        fused = (stacked * gates[:, :, None]).sum(axis=1)
        return x, pre, z, joined, gates, fused

    def fit(self, features: np.ndarray, labels: list[SemanticLabels], epochs: int = 420,
            learning_rate: float = .018, loss_masks: dict[str, np.ndarray] | None = None) -> None:
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-5
        n = len(features)
        targets = {name: self._indices(labels, name) for name in HEAD_CLASSES}
        masks = loss_masks or {name: np.ones(n, dtype=np.float32) for name in HEAD_CLASSES}
        for _ in range(epochs):
            x, pre, z, joined, gates, fused = self._forward(features)
            grad_z = {name: np.zeros_like(z[name]) for name in MODALITIES}

            for name, source in self.head_sources.items():
                weight, bias = self.heads[name]
                probs = _softmax(z[source] @ weight + bias)
                mask = masks.get(name, np.ones(n, dtype=np.float32))[:, None]
                grad = probs
                grad[np.arange(n), targets[name]] -= 1
                grad *= mask / max(float(mask.sum()), 1.0)
                grad_z[source] += grad @ weight.T
                self.heads[name][0] -= learning_rate * (z[source].T @ grad)
                self.heads[name][1] -= learning_rate * grad.sum(axis=0)

            danger_w, danger_b = self.heads["danger"]
            probs = _softmax(fused @ danger_w + danger_b)
            grad = probs
            grad[np.arange(n), targets["danger"]] -= 1
            grad /= n
            grad_fused = grad @ danger_w.T
            self.heads["danger"][0] -= learning_rate * (fused.T @ grad)
            self.heads["danger"][1] -= learning_rate * grad.sum(axis=0)

            grad_gate_values = np.empty_like(gates)
            for index, name in enumerate(MODALITIES):
                grad_z[name] += grad_fused * gates[:, index, None]
                grad_gate_values[:, index] = (grad_fused * z[name]).sum(axis=1)
            grad_gate_logits = gates * (
                grad_gate_values - (grad_gate_values * gates).sum(axis=1, keepdims=True)
            )
            grad_joined = grad_gate_logits @ self.gate[0].T
            self.gate[0] -= learning_rate * (joined.T @ grad_gate_logits)
            self.gate[1] -= learning_rate * grad_gate_logits.sum(axis=0)
            for index, name in enumerate(MODALITIES):
                grad_z[name] += grad_joined[:, index * self.embedding_size:(index + 1) * self.embedding_size]
                grad_z[name][pre[name] <= 0] = 0
                source_x = x[:, FEATURE_SLICES[name]]
                self.encoders[name][0] -= learning_rate * (source_x.T @ grad_z[name])
                self.encoders[name][1] -= learning_rate * grad_z[name].sum(axis=0)
        self.trained = True

    def probabilities(self, features: np.ndarray) -> dict[str, np.ndarray]:
        _, _, z, _, _, fused = self._forward(features)
        result = {}
        for name, source in self.head_sources.items():
            w, b = self.heads[name]
            result[name] = _softmax(z[source] @ w + b)
        w, b = self.heads["danger"]
        result["danger"] = _softmax(fused @ w + b)
        return result

    def modality_weights(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[4]

    def predict(self, features: np.ndarray) -> DecoderPrediction:
        probabilities = self.probabilities(features)
        chosen = {name: HEAD_CLASSES[name][int(p[0].argmax())] for name, p in probabilities.items()}
        confidence = {name: float(p[0].max()) for name, p in probabilities.items()}
        weights = self.modality_weights(features)[0]
        modality_weights = {name: float(weights[i]) for i, name in enumerate(MODALITIES)}
        return DecoderPrediction(**chosen, confidence=confidence, modality_weights=modality_weights)


def _danger_label(direction: str, sharpness: str, force: float, pain: float,
                  hardness: float, metallic: float) -> str:
    evidence = 1.8 * pain + .7 * force * hardness
    evidence += .8 * (sharpness == "sharp" and direction == "down")
    evidence += .45 * metallic
    return "danger" if evidence >= 1.15 else "safer"


def _make_training_data(samples: int, image_seed: int, signal_seed: int,
                        duration_ms: int = 120, augment: bool = True):
    images, shape_labels = generate_dataset(samples=samples, seed=image_seed)
    rng = np.random.default_rng(signal_seed)
    features, labels = [], []
    masks = {name: np.ones(samples, dtype=np.float32) for name in HEAD_CLASSES}
    scenarios = []
    for i, (image, shape_label) in enumerate(zip(images, shape_labels)):
        area, force, pain, hardness, metallic = rng.uniform(0, 1, 5)
        scenario = ("normal", "conflict", "missing", "noisy")[i % 4] if augment else "normal"
        if scenario == "conflict":
            visually_risky = shape_label.sharpness == "sharp" and shape_label.direction == "down"
            force, pain, hardness, metallic = ((.15, .08, .2, .1) if visually_risky
                                                else (.9, .9, .9, .8))
        bundle, _, _, _ = encode_multimodal(
            image, float(pain), float(metallic), 20000 + signal_seed + i,
            duration_ms=duration_ms, area=float(area), force=float(force), hardness=float(hardness),
        )
        if scenario == "missing":
            missing = MODALITIES[i % 3]
            signal_name = {"vision": "vision_v1", "touch": "touch", "odor": "odor"}[missing]
            signal = bundle.require(signal_name)
            bundle.signals[signal_name] = ModalitySignal(signal_name, np.zeros_like(signal.spikes))
            affected = {"vision": VISUAL_HEADS, "touch": TOUCH_HEADS, "odor": ODOR_HEADS}[missing]
            for head in affected:
                masks[head][i] = 0
        elif scenario == "noisy":
            noisy_name = ("vision_v1", "touch", "odor")[i % 3]
            signal = bundle.require(noisy_name)
            noise = rng.binomial(1, .18, signal.spikes.shape).astype(np.float32)
            bundle.signals[noisy_name] = ModalitySignal(noisy_name, np.maximum(signal.spikes, noise))
        features.append(bundle_features(bundle))
        labels.append(SemanticLabels(
            shape_label.direction, shape_label.sharpness, shape_label.completeness,
            "large" if area >= .5 else "small", "strong" if force >= .5 else "gentle",
            "pain" if pain >= .5 else "no_pain", "hard" if hardness >= .5 else "soft",
            "metallic" if metallic >= .5 else "not_metallic",
            _danger_label(shape_label.direction, shape_label.sharpness, force, pain, hardness, metallic),
        ))
        scenarios.append(scenario)
    return np.stack(features), labels, masks, scenarios


def train_default_decoder(samples: int = 900) -> MultiTaskDecoder:
    train_x, train_labels, masks, _ = _make_training_data(samples, 2027, 2028, augment=True)
    test_x, test_labels, _, _ = _make_training_data(max(200, samples // 4), 3027, 3028, augment=False)
    decoder = MultiTaskDecoder()
    decoder.fit(train_x, train_labels, loss_masks=masks)
    probabilities = decoder.probabilities(test_x)
    decoder.metrics = {name: float((p.argmax(axis=1) == decoder._indices(test_labels, name)).mean())
                       for name, p in probabilities.items()}
    return decoder


DECODER = None
CAUSAL_REPORT = None


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
    prediction, bundle, _, _, metrics, _ = decode_multimodal(image, 0, 0, seed)
    return prediction, bundle.require("vision_v1").spikes, metrics


def _ablate(features: np.ndarray, modality: str, shuffle: bool = False,
             rng: np.random.Generator | None = None) -> np.ndarray:
    changed = features.copy()
    section = changed[:, FEATURE_SLICES[modality]]
    if shuffle:
        rng = rng or np.random.default_rng(1)
        for row in section:
            rng.shuffle(row)
    else:
        section[:] = 0
    return changed


def _isolate(features: np.ndarray, modality: str) -> np.ndarray:
    changed = features.copy()
    for other in MODALITIES:
        if other != modality:
            changed[:, FEATURE_SLICES[other]] = 0
    return changed


def validate_decoder_causality(samples: int = 180) -> dict:
    global CAUSAL_REPORT
    if CAUSAL_REPORT is not None and CAUSAL_REPORT["samples"] == samples:
        return CAUSAL_REPORT
    decoder = get_decoder()
    x, labels, _, _ = _make_training_data(samples, 9102, 9103, augment=False)
    conditions = {"正常": x}
    for modality, label in (("vision", "V1"), ("touch", "触觉"), ("odor", "嗅觉")):
        conditions[f"切断{label}"] = _ablate(x, modality)
        conditions[f"打乱{label}"] = _ablate(x, modality, True, np.random.default_rng(9200))
    accuracies = {}
    for condition, values in conditions.items():
        probabilities = decoder.probabilities(values)
        accuracies[condition] = {
            head: float((probabilities[head].argmax(axis=1) == decoder._indices(labels, head)).mean())
            for head in HEAD_CLASSES
        }

    # Label-permutation control: the same architecture should not generalize arbitrary labels.
    control_x, control_labels, _, _ = _make_training_data(360, 8102, 8103, augment=False)
    control_test_x, control_test_labels, _, _ = _make_training_data(160, 8202, 8203, augment=False)
    rng = np.random.default_rng(8302)
    permuted = [control_labels[i] for i in rng.permutation(len(control_labels))]
    control = MultiTaskDecoder(seed=33)
    control.fit(control_x, permuted, epochs=150)
    control_probs = control.probabilities(control_test_x)
    random_label_accuracy = float(np.mean([
        (control_probs[h].argmax(axis=1) == control._indices(control_test_labels, h)).mean()
        for h in VISUAL_HEADS + TOUCH_HEADS + ODOR_HEADS
    ]))

    probe_x = x[:1]
    first = decoder.predict(probe_x)
    second = decoder.predict(probe_x.copy())
    reproducible = first == second
    normal_danger = accuracies["正常"]["danger"]
    single_danger = {}
    for modality in MODALITIES:
        probabilities = decoder.probabilities(_isolate(x, modality))["danger"]
        single_danger[modality] = float(
            (probabilities.argmax(axis=1) == decoder._indices(labels, "danger")).mean()
        )
    CAUSAL_REPORT = {
        "accuracies": accuracies, "reproducible": reproducible, "samples": samples,
        "random_label_accuracy": random_label_accuracy,
        "normal_mean": float(np.mean([accuracies["正常"][h] for h in VISUAL_HEADS])),
        "silenced_mean": float(np.mean([accuracies["切断V1"][h] for h in VISUAL_HEADS])),
        "causal_gap": float(np.mean([accuracies["正常"][h] - accuracies["切断V1"][h]
                                     for h in VISUAL_HEADS])),
        "normal_danger": normal_danger, "single_danger": single_danger,
    }
    return CAUSAL_REPORT
