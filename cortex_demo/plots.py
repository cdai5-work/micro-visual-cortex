from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .api import SimulationResult
from .settings import NEURONS_PER_GROUP, ORIENTATIONS
from .decoder import DecoderPrediction, spike_activity_maps


def visual_input_figure(image: np.ndarray):
    """Render a tiny 16×16 stimulus crisply without browser image upscaling artifacts."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    shown = ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("送入视网膜的16×16视觉刺激")
    ax.set_xlabel("水平像素")
    ax.set_ylabel("垂直像素")
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
    ax.grid(which="minor", color="#64748b", linewidth=.25, alpha=.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_aspect("equal")
    fig.colorbar(shown, ax=ax, fraction=.046, pad=.04, label="亮度")
    fig.tight_layout()
    return fig


def stimulus_figure(result: SimulationResult):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(result.stimulus, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("视觉刺激（16×16）")
    ax.axis("off")
    fig.tight_layout()
    return fig


def raster_figure(result: SimulationResult):
    fig, ax = plt.subplots(figsize=(7, 3))
    t, n = np.nonzero(result.input_spikes)
    ax.scatter(result.times_ms[t], n, s=2, color="#67e8f9", alpha=.75)
    ax.set(xlabel="时间 (ms)", ylabel="输入神经元", title="视网膜泊松脉冲")
    ax.set_xlim(0, result.times_ms[-1] if len(result.times_ms) else 1)
    fig.tight_layout()
    return fig


def rates_figure(result: SimulationResult):
    fig, ax = plt.subplots(figsize=(5, 3))
    values = [result.group_rates_hz[o] for o in ORIENTATIONS]
    bars = ax.bar([f"{o}°" for o in ORIENTATIONS], values,
                  color=["#22d3ee", "#818cf8", "#f472b6", "#f59e0b"])
    ax.bar_label(bars, fmt="%.1f")
    ax.set(ylabel="平均发放率 (Hz)", title="V1 方向群体响应")
    fig.tight_layout()
    return fig


def voltage_figure(result: SimulationResult):
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, orientation in enumerate(ORIENTATIONS):
        ax.plot(result.times_ms, result.membrane_voltage[:, i * NEURONS_PER_GROUP],
                label=f"{orientation}°", linewidth=1)
    ax.set(xlabel="时间 (ms)", ylabel="膜电位 (mV)", title="代表性神经元膜电位")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def heatmap_figure(result: SimulationResult):
    binned = result.v1_spikes.reshape(-1, 4, NEURONS_PER_GROUP).mean(axis=2).T
    fig, ax = plt.subplots(figsize=(7, 2.7))
    im = ax.imshow(binned, aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=(0, result.times_ms[-1], 3.5, -0.5))
    ax.set_yticks(range(4), [f"{o}°" for o in ORIENTATIONS])
    ax.set(xlabel="时间 (ms)", ylabel="偏好方向", title="群体活动热图")
    fig.colorbar(im, ax=ax, label="瞬时活动")
    fig.tight_layout()
    return fig


def decoder_confidence_figure(prediction: DecoderPrediction):
    labels = ["方向", "尖锐度", "完整性", "面积", "力度", "疼痛", "坚硬", "金属气味", "危险"]
    values = [prediction.confidence[name] for name in (
        "direction", "sharpness", "completeness", "area", "force", "pain",
        "hardness", "metallic", "danger"
    )]
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.barh(labels, values, color=plt.cm.viridis(np.linspace(.15, .9, len(labels))))
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("置信度")
    ax.set_title("可训练Decoder输出")
    fig.tight_layout()
    return fig


def tactile_activity_figure(spikes: np.ndarray):
    if spikes.shape[1] != 64:
        raise ValueError("触觉脉冲必须包含64个通道")
    population = spikes.mean(axis=0).reshape(4, 16)
    strengths = population[:, :8].mean(axis=1) * 2  # 500 Hz maximum → normalized 0–1
    labels = ["接触面积", "力度", "疼痛", "坚硬程度"]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, strengths, color=["#38bdf8", "#f59e0b", "#fb7185", "#94a3b8"])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in strengths], padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("触觉受体群活动（归一化）")
    ax.set_title("独立触觉模块输出（送入Decoder）")
    fig.tight_layout()
    return fig


def causal_validation_figure(report: dict):
    heads = ["direction", "sharpness", "completeness"]
    labels = ["方向", "尖锐度", "完整性"]
    conditions = ["正常V1", "切断V1", "打乱V1通道"]
    x = np.arange(len(heads))
    width = .24
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, condition, color in zip((-width, 0, width), conditions,
                                         ("#22c55e", "#ef4444", "#f59e0b")):
        values = [report["accuracies"][condition][head] for head in heads]
        ax.bar(x + offset, values, width, label=condition, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("未见样本准确率")
    ax.set_title("V1因果消融：破坏视觉信号后性能是否下降")
    ax.legend()
    fig.tight_layout()
    return fig


def decoder_raster_figure(spikes: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3))
    time_index, neuron_index = np.nonzero(spikes)
    ax.scatter(time_index, neuron_index, s=2, color="#67e8f9", alpha=.7)
    ax.set(xlabel="时间 (ms)", ylabel="输入神经元", title="图形转换后的泊松脉冲")
    ax.set_xlim(0, spikes.shape[0])
    ax.set_ylim(0, spikes.shape[1])
    fig.tight_layout()
    return fig


def decoder_v1_raster_figure(spikes: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3))
    time_index, neuron_index = np.nonzero(spikes)
    ax.scatter(time_index, neuron_index, s=3, color="#a78bfa", alpha=.75)
    ax.set(xlabel="时间 (ms)", ylabel="空间化V1神经元", title="V1输出脉冲（Decoder的视觉输入）")
    ax.set_xlim(0, spikes.shape[0])
    ax.set_ylim(0, spikes.shape[1])
    fig.tight_layout()
    return fig


def decoder_activity_figure(spikes: np.ndarray):
    activity = spike_activity_maps(spikes)["activity"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(activity, cmap="magma", interpolation="nearest", vmin=0)
    ax.set_title("由脉冲重建的16×16空间活动")
    ax.set_xlabel("视网膜位置 X")
    ax.set_ylabel("视网膜位置 Y")
    fig.colorbar(im, ax=ax, label="每时间步发放概率")
    fig.tight_layout()
    return fig


def decoder_edge_maps_figure(spikes: np.ndarray):
    maps = spike_activity_maps(spikes)
    panels = [
        ("vertical", "垂直边缘响应"),
        ("horizontal", "水平边缘响应"),
        ("diagonal_left", "左斜边缘响应"),
        ("diagonal_right", "右斜边缘响应"),
    ]
    maximum = max(float(maps[key].max()) for key, _ in panels) or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for ax, (key, title) in zip(axes.flat, panels):
        ax.imshow(maps[key], cmap="viridis", interpolation="nearest", vmin=0, vmax=maximum)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Decoder读取的方向边缘活动")
    fig.tight_layout()
    return fig
