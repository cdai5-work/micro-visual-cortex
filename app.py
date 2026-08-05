from __future__ import annotations

import gradio as gr

from cortex_demo import decode_multimodal, generate_shape, validate_decoder_causality
from cortex_demo.plots import (
    causal_validation_figure, decoder_activity_figure, decoder_confidence_figure,
    decoder_edge_maps_figure, decoder_raster_figure, decoder_v1_raster_figure,
    modality_weights_figure, tactile_activity_figure, visual_input_figure,
)

DIRECTION_MAP = {"上": "up", "下": "down", "左": "left", "右": "right"}
SHARPNESS_MAP = {"尖锐": "sharp", "圆滑": "rounded"}
COMPLETENESS_MAP = {"完整": "complete", "有缺口": "open"}
ZH_DIRECTION = {v: k for k, v in DIRECTION_MAP.items()}
ZH_SHARPNESS = {v: k for k, v in SHARPNESS_MAP.items()}
ZH_COMPLETENESS = {v: k for k, v in COMPLETENESS_MAP.items()}

SCENE_PRESETS = {
    "一致危险：尖锐且疼痛": ("下", "尖锐", "完整", .2, .85, .9, .9, .8),
    "冲突A：看似尖锐但柔软": ("下", "尖锐", "完整", .7, .15, .05, .1, .05),
    "冲突B：看似圆滑但高压疼痛": ("右", "圆滑", "完整", .15, .9, .9, .9, .1),
    "一致安全：圆滑且柔软": ("上", "圆滑", "完整", .7, .15, .05, .1, .05),
}


def apply_scene_preset(name):
    return SCENE_PRESETS[name]


def run_full_circuit(direction, sharpness, completeness, brightness, noise,
                     area, force, pain, hardness, metallic, seed):
    image = generate_shape(
        DIRECTION_MAP[direction], SHARPNESS_MAP[sharpness],
        COMPLETENESS_MAP[completeness], float(brightness), float(noise), int(seed),
    )
    prediction, bundle, retina_spikes, _, metrics, metadata = decode_multimodal(
        image, float(pain), float(metallic), int(seed) + 100,
        area=float(area), force=float(force), hardness=float(hardness),
    )
    v1_spikes = bundle.require("vision_v1").spikes
    touch_spikes = bundle.require("touch").spikes
    danger_text = "危险" if prediction.danger == "danger" else "较安全"
    pain_text = "有疼痛" if prediction.pain == "pain" else "无明显疼痛"
    metal_text = "有金属气味" if prediction.metallic == "metallic" else "无明显金属气味"
    area_text = "大面积" if prediction.area == "large" else "小面积"
    force_text = "强力" if prediction.force == "strong" else "轻柔"
    hardness_text = "坚硬" if prediction.hardness == "hard" else "柔软"
    fallback = metadata.get("fallback_reason")
    conclusion = (
        "### 感知理解结果\n"
        "#### 来自V1的视觉属性\n"
        f"- **{ZH_DIRECTION[prediction.direction]}、{ZH_SHARPNESS[prediction.sharpness]}、"
        f"{ZH_COMPLETENESS[prediction.completeness]}**\n"
        f"- 置信度：方向 {prediction.confidence['direction']:.1%} · "
        f"边缘 {prediction.confidence['sharpness']:.1%} · "
        f"轮廓 {prediction.confidence['completeness']:.1%}\n\n"
        "#### 来自触觉模块的属性\n"
        f"- **{area_text}、{force_text}、{hardness_text}、{pain_text}**\n"
        f"- 置信度：面积 {prediction.confidence['area']:.1%} · "
        f"力度 {prediction.confidence['force']:.1%} · 硬度 {prediction.confidence['hardness']:.1%} · "
        f"疼痛 {prediction.confidence['pain']:.1%}\n\n"
        "#### 来自嗅觉模块的属性\n"
        f"- **{metal_text}**（{prediction.confidence['metallic']:.1%}）\n\n"
        "#### 跨模态融合结论\n"
        f"- **{danger_text}**（置信度 {prediction.confidence['danger']:.1%}）\n\n"
        "#### 门控融合权重\n"
        f"- 视觉 {prediction.modality_weights['vision']:.1%} · "
        f"触觉 {prediction.modality_weights['touch']:.1%} · "
        f"嗅觉 {prediction.modality_weights['odor']:.1%}\n\n"
        f"运行后端：{metadata['backend']} · 耗时 {metadata['elapsed_ms']:.2f} ms"
        + (f"\n\n⚠️ {fallback}" if fallback else "")
        + "\n\n> Decoder只接收V1、触觉与嗅觉脉冲，不读取上方控件值或正确标签。"
    )
    return (
        visual_input_figure(image), decoder_raster_figure(retina_spikes),
        decoder_v1_raster_figure(v1_spikes),
        decoder_activity_figure(v1_spikes), decoder_edge_maps_figure(v1_spikes),
        tactile_activity_figure(touch_spikes), decoder_confidence_figure(prediction),
        modality_weights_figure(prediction), conclusion,
    )


def run_causal_test():
    report = validate_decoder_causality()
    normal = report["normal_mean"]
    silenced = report["silenced_mean"]
    gap = report["causal_gap"]
    reproducible = "通过" if report["reproducible"] else "失败"
    modalities_pass = (
        gap >= .20
        and report["accuracies"]["正常"]["pain"] - report["accuracies"]["切断触觉"]["pain"] >= .20
        and report["accuracies"]["正常"]["metallic"] - report["accuracies"]["切断嗅觉"]["metallic"] >= .20
    )
    verdict = ("支持各属性头依赖正确感觉模态" if modalities_pass
               else "证据不足，需要检查数据泄漏或特征设计")
    text = (
        "### 因果验证报告\n"
        f"- 固定随机种子逐脉冲复现：**{reproducible}**\n"
        f"- {report['samples']}个未见样本，正常V1视觉平均准确率：**{normal:.1%}**\n"
        f"- 切断V1后的视觉平均准确率：**{silenced:.1%}**\n"
        f"- 因果性能差：**{gap:.1%}**\n"
        f"- 触觉切断后危险准确率：**{report['accuracies']['切断触觉']['danger']:.1%}**\n"
        f"- 嗅觉切断后危险准确率：**{report['accuracies']['切断嗅觉']['danger']:.1%}**\n"
        f"- 完整多模态危险准确率：**{report['normal_danger']:.1%}**；"
        f"最佳单模态：**{max(report['single_danger'].values()):.1%}**\n"
        f"- 随机标签对照平均准确率：**{report['random_label_accuracy']:.1%}**\n"
        f"- 结论：**{verdict}**\n\n"
        "这里的“学习”仅表示Decoder从训练样本中学会V1脉冲与标签的统计映射；"
        "不代表生物大脑水平的理解。"
    )
    return causal_validation_figure(report), text


with gr.Blocks(title="微型多模态感知回路 v0.6") as demo:
    gr.Markdown(
        "# 微型多模态感知回路 v0.6\n"
        "三个独立感觉编码器通过可训练门控融合，并用模态消融与随机标签对照验证学习。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ① 视觉输入")
            direction = gr.Radio(list(DIRECTION_MAP), value="下", label="图形方向")
            sharpness = gr.Radio(list(SHARPNESS_MAP), value="尖锐", label="边缘形态")
            completeness = gr.Radio(list(COMPLETENESS_MAP), value="完整", label="轮廓状态")
            brightness = gr.Slider(.4, 1, value=.9, step=.05, label="亮度")
            noise = gr.Slider(0, .25, value=.04, step=.01, label="视觉噪声")
        with gr.Column(scale=1):
            gr.Markdown("## ② 触觉输入")
            area = gr.Slider(0, 1, value=.35, step=.05, label="接触面积（小 → 大）")
            force = gr.Slider(0, 1, value=.75, step=.05, label="作用力度（轻 → 强）")
            pain = gr.Slider(0, 1, value=.75, step=.05, label="疼痛程度（无 → 强）")
            hardness = gr.Slider(0, 1, value=.85, step=.05, label="坚硬程度（软 → 硬）")
        with gr.Column(scale=1):
            gr.Markdown("## ③ 嗅觉与运行")
            metallic = gr.Slider(0, 1, value=.75, step=.05, label="金属气味强度")
            seed = gr.Number(value=7, precision=0, label="随机种子（用于复现）")
            preset = gr.Dropdown(list(SCENE_PRESETS), value="一致危险：尖锐且疼痛", label="场景预设")
            apply_preset = gr.Button("载入场景预设")
            run = gr.Button("运行完整感知回路", variant="primary")

    gr.Markdown("## ④ 视觉通路：输入 → 视网膜 → V1输出")
    with gr.Row():
        image = gr.Plot(label="V1接收的16×16视觉图形")
        retina_plot = gr.Plot(label="视网膜泊松脉冲")
    v1_raster = gr.Plot(label="V1输出脉冲（Decoder的真实视觉输入）")
    with gr.Row():
        v1_activity = gr.Plot(label="V1空间活动")
        v1_edges = gr.Plot(label="V1方向选择活动")

    gr.Markdown("## ⑤ 触觉模块输出与多模态感知理解")
    touch_plot = gr.Plot(label="触觉模块64路群体脉冲摘要")
    with gr.Row():
        confidence_plot = gr.Plot(label="九个Decoder输出头")
        weights_plot = gr.Plot(label="三种感觉贡献权重")
        conclusion = gr.Markdown()

    inputs = [direction, sharpness, completeness, brightness, noise,
              area, force, pain, hardness, metallic, seed]
    outputs = [image, retina_plot, v1_raster, v1_activity, v1_edges,
               touch_plot, confidence_plot, weights_plot, conclusion]
    preset_outputs = [direction, sharpness, completeness, area, force, pain, hardness, metallic]
    apply_preset.click(apply_scene_preset, inputs=preset, outputs=preset_outputs)
    run.click(run_full_circuit, inputs, outputs)
    demo.load(run_full_circuit, inputs, outputs)

    gr.Markdown("## ⑥ 它真的在学习吗？")
    gr.Markdown(
        "点击后在未参与训练的样本上分别切断或打乱V1、触觉和嗅觉，"
        "并与完整多模态、单模态和随机标签训练进行对照。"
    )
    verify = gr.Button("运行因果验证")
    with gr.Row():
        causal_plot = gr.Plot(label="消融实验")
        causal_text = gr.Markdown()
    verify.click(run_causal_test, outputs=[causal_plot, causal_text])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
