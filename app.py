from __future__ import annotations

import gradio as gr

from cortex_demo import StimulusConfig, simulate
from cortex_demo.plots import (
    heatmap_figure, raster_figure, rates_figure, stimulus_figure, voltage_figure,
)
from cortex_demo.settings import STIMULUS_LABELS


def run_demo(label, angle, brightness, noise, duration, seed):
    result = simulate(StimulusConfig(
        stimulus_type=STIMULUS_LABELS[label], angle_deg=float(angle),
        brightness=float(brightness), noise=float(noise),
        duration_ms=int(duration), seed=int(seed),
    ))
    best = max(result.group_rates_hz, key=result.group_rates_hz.get)
    explanation = (
        f"### 仿真结论\n偏好 **{best}°** 的神经群响应最强，平均发放率为 "
        f"**{result.group_rates_hz[best]:.1f} Hz**。\n\n"
        f"后端：{result.metadata['backend']} · 计算耗时：{result.metadata['elapsed_ms']} ms  "
        + (f"\n\n⚠️ {result.metadata['fallback_reason']}" if result.metadata.get("fallback_reason") else "") +
        "\n\n> 这是教学性简化模型，结果不代表真实动物视觉皮层测量。"
    )
    return (stimulus_figure(result), raster_figure(result), rates_figure(result),
            voltage_figure(result), heatmap_figure(result), explanation)


with gr.Blocks(title="微型视觉皮层") as demo:
    gr.Markdown("# 微型视觉皮层\n把几何刺激变成脉冲，观察简化 V1 的方向选择性。")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 刺激设置")
            stimulus = gr.Dropdown(list(STIMULUS_LABELS), value="垂直条纹", label="刺激类型")
            angle = gr.Slider(0, 180, value=90, step=1, label="条纹方向（度）")
            brightness = gr.Slider(0, 1, value=.8, step=.05, label="亮度")
            noise = gr.Slider(0, .5, value=.05, step=.01, label="噪声")
            duration = gr.Slider(20, 1000, value=200, step=20, label="仿真时长（ms）")
            seed = gr.Number(value=42, precision=0, label="随机种子")
            with gr.Row():
                run = gr.Button("开始仿真", variant="primary")
                reset = gr.Button("恢复默认")
            stimulus_plot = gr.Plot(label="原始刺激")
        with gr.Column(scale=2):
            gr.Markdown("## 脑活动")
            explanation = gr.Markdown()
            rates_plot = gr.Plot()
            raster_plot = gr.Plot()
            voltage_plot = gr.Plot()
            heatmap_plot = gr.Plot()

    inputs = [stimulus, angle, brightness, noise, duration, seed]
    outputs = [stimulus_plot, raster_plot, rates_plot, voltage_plot, heatmap_plot, explanation]
    run.click(run_demo, inputs, outputs)
    reset.click(lambda: ("垂直条纹", 90, .8, .05, 200, 42), outputs=inputs)
    demo.load(run_demo, inputs, outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
