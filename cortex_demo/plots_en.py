from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .decoder import DecoderPrediction, spike_activity_maps


def visual_input_figure(image):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    shown = ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set(title="16×16 visual stimulus sent to the retina", xlabel="Horizontal pixel", ylabel="Vertical pixel")
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which="minor", color="#64748b", linewidth=.25, alpha=.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("equal")
    fig.colorbar(shown, ax=ax, fraction=.046, pad=.04, label="Brightness")
    fig.tight_layout()
    return fig


def decoder_raster_figure(spikes):
    fig, ax = plt.subplots(figsize=(7, 3))
    time_index, neuron_index = np.nonzero(spikes)
    ax.scatter(time_index, neuron_index, s=2, color="#67e8f9", alpha=.7)
    ax.set(xlabel="Time (ms)", ylabel="Input neuron", title="Retinal Poisson spikes")
    ax.set(xlim=(0, spikes.shape[0]), ylim=(0, spikes.shape[1]))
    fig.tight_layout()
    return fig


def decoder_v1_raster_figure(spikes):
    fig, ax = plt.subplots(figsize=(7, 3))
    time_index, neuron_index = np.nonzero(spikes)
    ax.scatter(time_index, neuron_index, s=3, color="#a78bfa", alpha=.75)
    ax.set(xlabel="Time (ms)", ylabel="Retinotopic V1 neuron",
           title="V1 output spikes — the Decoder's visual input")
    ax.set(xlim=(0, spikes.shape[0]), ylim=(0, spikes.shape[1]))
    fig.tight_layout()
    return fig


def decoder_activity_figure(spikes):
    activity = spike_activity_maps(spikes)["activity"]
    fig, ax = plt.subplots(figsize=(4, 4))
    shown = ax.imshow(activity, cmap="magma", interpolation="nearest", vmin=0)
    ax.set(title="Spatial activity reconstructed from V1 spikes",
           xlabel="Retinal position X", ylabel="Retinal position Y")
    fig.colorbar(shown, ax=ax, label="Spike probability per timestep")
    fig.tight_layout()
    return fig


def decoder_edge_maps_figure(spikes):
    maps = spike_activity_maps(spikes)
    panels = [("vertical", "Vertical edges"), ("horizontal", "Horizontal edges"),
              ("diagonal_left", "Left-diagonal edges"),
              ("diagonal_right", "Right-diagonal edges")]
    maximum = max(float(maps[key].max()) for key, _ in panels) or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for ax, (key, title) in zip(axes.flat, panels):
        ax.imshow(maps[key], cmap="viridis", interpolation="nearest", vmin=0, vmax=maximum)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Orientation-selective activity read by the Decoder")
    fig.tight_layout()
    return fig


def tactile_activity_figure(spikes):
    population = spikes.mean(axis=0).reshape(4, 16)
    strengths = population[:, :8].mean(axis=1) * 2
    labels = ["Contact area", "Force", "Pain", "Hardness"]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, strengths, color=["#38bdf8", "#f59e0b", "#fb7185", "#94a3b8"])
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in strengths], padding=3)
    ax.set(ylim=(0, 1.1), ylabel="Normalized receptor activity",
           title="Independent tactile-module output")
    fig.tight_layout()
    return fig


def decoder_confidence_figure(prediction: DecoderPrediction):
    names = ("direction", "sharpness", "completeness", "area", "force", "pain",
             "hardness", "metallic", "danger")
    labels = ["Direction", "Sharpness", "Completeness", "Area", "Force", "Pain",
              "Hardness", "Metal odor", "Danger"]
    values = [prediction.confidence[name] for name in names]
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.barh(labels, values, color=plt.cm.viridis(np.linspace(.15, .9, len(labels))))
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set(xlim=(0, 1.08), xlabel="Confidence", title="Nine trainable Decoder heads")
    fig.tight_layout()
    return fig


def modality_weights_figure(prediction: DecoderPrediction):
    values = [prediction.modality_weights[name] for name in ("vision", "touch", "odor")]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(["Vision / V1", "Touch", "Odor"], values,
                  color=["#8b5cf6", "#f59e0b", "#22c55e"])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set(ylim=(0, 1.08), ylabel="Gating contribution",
           title="Sensory contributions to the danger judgment")
    fig.tight_layout()
    return fig


def causal_validation_figure(report):
    groups = [
        ("Visual attributes", ("direction", "sharpness", "completeness"), "切断V1", "打乱V1"),
        ("Tactile attributes", ("area", "force", "pain", "hardness"), "切断触觉", "打乱触觉"),
        ("Odor attribute", ("metallic",), "切断嗅觉", "打乱嗅觉"),
        ("Danger", ("danger",), "切断触觉", "打乱触觉"),
    ]
    x, width = np.arange(len(groups)), .24
    fig, ax = plt.subplots(figsize=(8, 4.5))
    normal = [np.mean([report["accuracies"]["正常"][h] for h in heads]) for _, heads, _, _ in groups]
    cut = [np.mean([report["accuracies"][c][h] for h in heads]) for _, heads, c, _ in groups]
    shuffled = [np.mean([report["accuracies"][c][h] for h in heads]) for _, heads, _, c in groups]
    for offset, values, label, color in (
        (-width, normal, "Normal", "#22c55e"),
        (0, cut, "Corresponding modality silenced", "#ef4444"),
        (width, shuffled, "Corresponding modality shuffled", "#f59e0b"),
    ):
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(report["random_label_accuracy"], color="#64748b", linestyle="--",
               label="Random-label control")
    ax.set_xticks(x, [group[0] for group in groups])
    ax.set(ylim=(0, 1.05), ylabel="Accuracy on unseen samples",
           title="Multimodal causal ablation and random-label control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
