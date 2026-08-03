from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .api import SimulationResult
from .settings import NEURONS_PER_GROUP, ORIENTATIONS


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

