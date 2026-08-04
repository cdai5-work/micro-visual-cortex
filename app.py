from __future__ import annotations

import gradio as gr

from cortex_demo import StimulusConfig, decode_multimodal, generate_shape, simulate
from cortex_demo.plots import (
    decoder_activity_figure, decoder_confidence_figure, decoder_edge_maps_figure,
    decoder_raster_figure, decoder_v1_raster_figure, heatmap_figure, raster_figure, rates_figure,
    stimulus_figure, voltage_figure,
)
from cortex_demo.settings import STIMULUS_LABELS


def run_v1_demo(label, angle, brightness, noise, duration, seed):
    result = simulate(StimulusConfig(
        stimulus_type=STIMULUS_LABELS[label], angle_deg=float(angle),
        brightness=float(brightness), noise=float(noise),
        duration_ms=int(duration), seed=int(seed),
    ))
    best = max(result.group_rates_hz, key=result.group_rates_hz.get)
    fallback = result.metadata.get("fallback_reason")
    explanation = (
        f"### 仿真结论\n偏好 **{best}°** 的神经群响应最强，平均发放率为 "
        f"**{result.group_rates_hz[best]:.1f} Hz**。\n\n"
        f"后端：{result.metadata['backend']} · 计算耗时：{result.metadata['elapsed_ms']} ms"
        + (f"\n\n⚠️ {fallback}" if fallback else "")
        + "\n\n> 这是教学性简化模型，结果不代表真实动物视觉皮层测量。"
    )
    return (stimulus_figure(result), raster_figure(result), rates_figure(result),
            voltage_figure(result), heatmap_figure(result), explanation)


DIRECTION_MAP = {"上": "up", "下": "down", "左": "left", "右": "right"}
SHARPNESS_MAP = {"尖锐": "sharp", "圆滑": "rounded"}
COMPLETENESS_MAP = {"完整": "complete", "有缺口": "open"}
ZH_DIRECTION = {v: k for k, v in DIRECTION_MAP.items()}
ZH_SHARPNESS = {v: k for k, v in SHARPNESS_MAP.items()}
ZH_COMPLETENESS = {v: k for k, v in COMPLETENESS_MAP.items()}


def run_decoder_demo(direction, sharpness, completeness, brightness, noise,
                     pain, metallic, seed):
    image = generate_shape(
        DIRECTION_MAP[direction], SHARPNESS_MAP[sharpness],
        COMPLETENESS_MAP[completeness], float(brightness), float(noise), int(seed),
    )
    prediction, bundle, retina_spikes, _, metrics, metadata = decode_multimodal(
        image, float(pain), float(metallic), int(seed) + 100
    )
    v1_spikes = bundle.require("vision_v1").spikes
    danger_text = "危险" if prediction.danger == "danger" else "较安全"
    pain_text = "有疼痛" if prediction.pain == "pain" else "无明显疼痛"
    metal_text = "有金属气味" if prediction.metallic == "metallic" else "无明显金属气味"
    conclusion = (
        "### Decoder解读结果\n"
        f"- 方向：**{ZH_DIRECTION[prediction.direction]}** "
        f"（置信度 {prediction.confidence['direction']:.1%}）\n"
        f"- 边缘：**{ZH_SHARPNESS[prediction.sharpness]}** "
        f"（置信度 {prediction.confidence['sharpness']:.1%}）\n"
        f"- 轮廓：**{ZH_COMPLETENESS[prediction.completeness]}** "
        f"（置信度 {prediction.confidence['completeness']:.1%}）\n\n"
        f"- 触觉：**{pain_text}**（置信度 {prediction.confidence['pain']:.1%}）\n"
        f"- 嗅觉：**{metal_text}**（置信度 {prediction.confidence['metallic']:.1%}）\n"
        f"- 综合判断：**{danger_text}**（置信度 {prediction.confidence['danger']:.1%}）\n\n"
        "Decoder的视觉输入来自64个空间化V1神经元的输出脉冲，并与触觉、嗅觉脉冲融合；它不会读取输入标签。\n\n"
        "验证集准确率："
        f"方向 {metrics['direction']:.1%} · 边缘 {metrics['sharpness']:.1%} · "
        f"轮廓 {metrics['completeness']:.1%} · 疼痛 {metrics['pain']:.1%} · "
        f"金属 {metrics['metallic']:.1%} · 危险 {metrics['danger']:.1%}"
    )
    return (
        image,
        decoder_raster_figure(retina_spikes),
        decoder_v1_raster_figure(v1_spikes),
        decoder_activity_figure(v1_spikes),
        decoder_edge_maps_figure(v1_spikes),
        decoder_confidence_figure(prediction),
        conclusion,
    )


with gr.Blocks(title="微型多模态感知回路 v0.3") as demo:
    gr.Markdown(
        "# 微型多模态感知回路 v0.3\n"
        "视觉经过视网膜与V1，再与触觉、嗅觉脉冲汇合，由可训练网络解读属性和简单危险含义。"
    )
    with gr.Tabs():
        with gr.Tab("V1神经活动"):
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

            v1_inputs = [stimulus, angle, brightness, noise, duration, seed]
            v1_outputs = [stimulus_plot, raster_plot, rates_plot, voltage_plot, heatmap_plot, explanation]
            run.click(run_v1_demo, v1_inputs, v1_outputs)
            reset.click(lambda: ("垂直条纹", 90, .8, .05, 200, 42), outputs=v1_inputs)
            demo.load(run_v1_demo, v1_inputs, v1_outputs)

        with gr.Tab("多模态感知解码"):
            gr.Markdown(
                "## 让网络从脉冲中学习含义\n"
                "图形先经过V1；V1输出再和触觉、嗅觉脉冲一起进入可更新的多任务MLP。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    decoder_direction = gr.Radio(list(DIRECTION_MAP), value="上", label="图形方向")
                    decoder_sharpness = gr.Radio(list(SHARPNESS_MAP), value="尖锐", label="边缘形态")
                    decoder_completeness = gr.Radio(list(COMPLETENESS_MAP), value="完整", label="轮廓状态")
                    decoder_brightness = gr.Slider(.4, 1, value=.9, step=.05, label="亮度")
                    decoder_noise = gr.Slider(0, .25, value=.04, step=.01, label="噪声")
                    decoder_pain = gr.Slider(0, 1, value=.75, step=.05, label="触觉疼痛强度")
                    decoder_metallic = gr.Slider(0, 1, value=.75, step=.05, label="金属气味强度")
                    decoder_seed = gr.Number(value=7, precision=0, label="随机种子")
                    decode_button = gr.Button("生成并解码", variant="primary")
                with gr.Column(scale=2):
                    decoded_image = gr.Image(label="Decoder实际接收的图形", type="numpy")
                    decoder_conclusion = gr.Markdown()
                    decoder_confidence = gr.Plot(label="预测置信度")
            gr.Markdown("## 从图形到含义的中间信号")
            decoder_raster = gr.Plot(label="视网膜泊松脉冲")
            decoder_v1_raster = gr.Plot(label="V1输出脉冲")
            with gr.Row():
                decoder_activity = gr.Plot(label="空间神经活动")
                decoder_edges = gr.Plot(label="方向边缘活动")

            decoder_inputs = [
                decoder_direction, decoder_sharpness, decoder_completeness,
                decoder_brightness, decoder_noise, decoder_pain,
                decoder_metallic, decoder_seed,
            ]
            decoder_outputs = [
                decoded_image, decoder_raster, decoder_v1_raster, decoder_activity,
                decoder_edges, decoder_confidence, decoder_conclusion,
            ]
            decode_button.click(run_decoder_demo, decoder_inputs, decoder_outputs)
            demo.load(run_decoder_demo, decoder_inputs, decoder_outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
