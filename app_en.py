from __future__ import annotations

import gradio as gr

from cortex_demo import decode_multimodal, generate_shape, validate_decoder_causality
from cortex_demo.plots_en import (
    causal_validation_figure, decoder_activity_figure, decoder_confidence_figure,
    decoder_edge_maps_figure, decoder_raster_figure, decoder_v1_raster_figure,
    modality_weights_figure, tactile_activity_figure, visual_input_figure,
)

DIRECTIONS = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
SHARPNESS = {"Sharp": "sharp", "Rounded": "rounded"}
COMPLETENESS = {"Complete": "complete", "Open contour": "open"}
EN_DIRECTION = {value: key for key, value in DIRECTIONS.items()}
EN_SHARPNESS = {value: key for key, value in SHARPNESS.items()}
EN_COMPLETENESS = {value: key for key, value in COMPLETENESS.items()}

SCENE_PRESETS = {
    "Consistent danger: sharp and painful": ("Down", "Sharp", "Complete", .2, .85, .9, .9, .8),
    "Conflict A: looks sharp but feels soft": ("Down", "Sharp", "Complete", .7, .15, .05, .1, .05),
    "Conflict B: looks round but causes painful pressure": ("Right", "Rounded", "Complete", .15, .9, .9, .9, .1),
    "Consistent safety: round and soft": ("Up", "Rounded", "Complete", .7, .15, .05, .1, .05),
}


def apply_scene_preset(name):
    return SCENE_PRESETS[name]


def run_full_circuit(direction, sharpness, completeness, brightness, noise,
                     area, force, pain, hardness, metallic, seed):
    image = generate_shape(DIRECTIONS[direction], SHARPNESS[sharpness],
                           COMPLETENESS[completeness], float(brightness), float(noise), int(seed))
    prediction, bundle, retinal_spikes, _, _, metadata = decode_multimodal(
        image, float(pain), float(metallic), int(seed) + 100,
        area=float(area), force=float(force), hardness=float(hardness),
    )
    v1_spikes = bundle.require("vision_v1").spikes
    tactile_spikes = bundle.require("touch").spikes
    danger = "Danger" if prediction.danger == "danger" else "Relatively safe"
    pain_text = "painful" if prediction.pain == "pain" else "no clear pain"
    metal = "metallic odor" if prediction.metallic == "metallic" else "no clear metallic odor"
    area_text = "large contact area" if prediction.area == "large" else "small contact area"
    force_text = "strong force" if prediction.force == "strong" else "gentle force"
    hardness_text = "hard" if prediction.hardness == "hard" else "soft"
    fallback = metadata.get("fallback_reason")
    if fallback and "JAX 不可用" in fallback:
        fallback = "JAX is unavailable; the simulation fell back to the NumPy CPU backend."
    explanation = (
        "### Perceptual interpretation\n"
        "#### Visual attributes decoded only from V1\n"
        f"- **{EN_DIRECTION[prediction.direction]}, {EN_SHARPNESS[prediction.sharpness]}, "
        f"{EN_COMPLETENESS[prediction.completeness]}**\n"
        f"- Confidence: direction {prediction.confidence['direction']:.1%} · "
        f"edge {prediction.confidence['sharpness']:.1%} · "
        f"contour {prediction.confidence['completeness']:.1%}\n\n"
        "#### Tactile attributes decoded only from the tactile module\n"
        f"- **{area_text}, {force_text}, {hardness_text}, {pain_text}**\n"
        f"- Confidence: area {prediction.confidence['area']:.1%} · "
        f"force {prediction.confidence['force']:.1%} · hardness {prediction.confidence['hardness']:.1%} · "
        f"pain {prediction.confidence['pain']:.1%}\n\n"
        "#### Olfactory attribute\n"
        f"- **{metal}** ({prediction.confidence['metallic']:.1%})\n\n"
        "#### Cross-modal fusion\n"
        f"- **{danger}** ({prediction.confidence['danger']:.1%} confidence)\n"
        f"- Gate: vision {prediction.modality_weights['vision']:.1%} · "
        f"touch {prediction.modality_weights['touch']:.1%} · "
        f"odor {prediction.modality_weights['odor']:.1%}\n\n"
        f"Backend: {metadata['backend']} · compute time {metadata['elapsed_ms']:.2f} ms"
        + (f"\n\n⚠️ {fallback}" if fallback else "")
        + "\n\n> The Decoder receives only V1, tactile, and olfactory spikes. It cannot read the controls or labels."
    )
    return (
        visual_input_figure(image), decoder_raster_figure(retinal_spikes),
        decoder_v1_raster_figure(v1_spikes), decoder_activity_figure(v1_spikes),
        decoder_edge_maps_figure(v1_spikes), tactile_activity_figure(tactile_spikes),
        decoder_confidence_figure(prediction), modality_weights_figure(prediction), explanation,
    )


def run_causal_test():
    report = validate_decoder_causality()
    gate_pass = (report["causal_gap"] >= .20
                 and report["accuracies"]["正常"]["pain"] - report["accuracies"]["切断触觉"]["pain"] >= .20
                 and report["accuracies"]["正常"]["metallic"] - report["accuracies"]["切断嗅觉"]["metallic"] >= .20)
    verdict = "The attribute heads depend on the correct sensory modalities" if gate_pass else "Evidence is insufficient"
    text = (
        "### Causal validation report\n"
        f"- Exact spike-level reproducibility with a fixed seed: **{'Pass' if report['reproducible'] else 'Fail'}**\n"
        f"- Normal visual accuracy on {report['samples']} unseen samples: **{report['normal_mean']:.1%}**\n"
        f"- Visual accuracy after silencing V1: **{report['silenced_mean']:.1%}**\n"
        f"- Danger accuracy after silencing touch: **{report['accuracies']['切断触觉']['danger']:.1%}**\n"
        f"- Full multimodal danger accuracy: **{report['normal_danger']:.1%}**; "
        f"best single modality: **{max(report['single_danger'].values()):.1%}**\n"
        f"- Random-label control: **{report['random_label_accuracy']:.1%}**\n"
        f"- Conclusion: **{verdict}**\n\n"
        "Learning here means fitting statistical mappings in a synthetic environment; it is not human-level understanding."
    )
    return causal_validation_figure(report), text


with gr.Blocks(title="Mini Multimodal Perception Circuit v0.6 — English") as demo:
    gr.Markdown(
        "# Mini Multimodal Perception Circuit v0.6\n"
        "**English showcase** · [中文说明](https://github.com/cdai5-work/micro-visual-cortex/blob/main/README.md) · "
        "Three independent sensory encoders are combined by a trainable gate."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("## ① Visual input")
            direction = gr.Radio(list(DIRECTIONS), value="Down", label="Direction")
            sharpness = gr.Radio(list(SHARPNESS), value="Sharp", label="Edge shape")
            completeness = gr.Radio(list(COMPLETENESS), value="Complete", label="Contour")
            brightness = gr.Slider(.4, 1, .9, step=.05, label="Brightness")
            noise = gr.Slider(0, .25, .04, step=.01, label="Visual noise")
        with gr.Column():
            gr.Markdown("## ② Tactile input")
            area = gr.Slider(0, 1, .35, step=.05, label="Contact area")
            force = gr.Slider(0, 1, .75, step=.05, label="Applied force")
            pain = gr.Slider(0, 1, .75, step=.05, label="Pain intensity")
            hardness = gr.Slider(0, 1, .85, step=.05, label="Hardness")
        with gr.Column():
            gr.Markdown("## ③ Olfactory input and run")
            metallic = gr.Slider(0, 1, .75, step=.05, label="Metallic odor")
            seed = gr.Number(7, precision=0, label="Random seed")
            preset = gr.Dropdown(list(SCENE_PRESETS), value=list(SCENE_PRESETS)[0], label="Scenario preset")
            apply_button = gr.Button("Load preset")
            run = gr.Button("Run the complete circuit", variant="primary")

    gr.Markdown("## ④ Visual pathway: input → retina → V1")
    with gr.Row():
        image = gr.Plot(label="16×16 visual input")
        retinal_plot = gr.Plot(label="Retinal Poisson spikes")
    v1_raster = gr.Plot(label="V1 spikes — the Decoder's actual visual input")
    with gr.Row():
        v1_activity = gr.Plot(label="Retinotopic V1 activity")
        v1_edges = gr.Plot(label="Orientation-selective V1 activity")

    gr.Markdown("## ⑤ Tactile output and multimodal interpretation")
    tactile_plot = gr.Plot(label="Tactile population activity")
    with gr.Row():
        confidence_plot = gr.Plot(label="Nine Decoder heads")
        weights_plot = gr.Plot(label="Sensory contribution weights")
        explanation = gr.Markdown()

    inputs = [direction, sharpness, completeness, brightness, noise,
              area, force, pain, hardness, metallic, seed]
    outputs = [image, retinal_plot, v1_raster, v1_activity, v1_edges,
               tactile_plot, confidence_plot, weights_plot, explanation]
    apply_button.click(apply_scene_preset, preset,
                       [direction, sharpness, completeness, area, force, pain, hardness, metallic])
    run.click(run_full_circuit, inputs, outputs)
    demo.load(run_full_circuit, inputs, outputs)

    gr.Markdown("## ⑥ Is it actually learning?")
    gr.Markdown("Compare normal operation with modality silencing, channel shuffling, single-modality baselines, and random labels.")
    verify = gr.Button("Run causal validation")
    with gr.Row():
        causal_plot = gr.Plot(label="Ablation experiment")
        causal_text = gr.Markdown()
    verify.click(run_causal_test, outputs=[causal_plot, causal_text])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(server_port=7861, theme=gr.themes.Soft())
