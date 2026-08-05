# Mini Multimodal Perception Circuit

[中文](README.md) | [English](README_EN.md)

An interactive educational spiking-neural-network project that connects a simplified retina and V1 pathway with tactile and olfactory signals, then uses a trainable gated Decoder to infer basic attributes and danger.

```text
Image → retinal Poisson spikes → retinotopic V1 LIF spikes ┐
Area / force / pain / hardness → tactile spikes            ├→ gated Decoder → attributes + danger
Metallic odor → olfactory spikes                            ┘
```

The Decoder cannot read the image controls or ground-truth labels. Its visual input is the actual 64-channel V1 spike output. This is a synthetic teaching model, not a biological reproduction or a safety system.

## What v0.6 demonstrates

- Independent 32-dimensional visual, tactile, and olfactory encoders.
- Visual heads can only access the visual representation; tactile heads can only access touch; the odor head can only access smell.
- A trainable softmax gate combines the three representations for the danger head and exposes their contribution weights.
- Training includes aligned, conflicting, missing-modality, and noisy examples.
- Scenario presets include “looks sharp but feels soft” and “looks round but causes painful pressure.”
- Causal tests silence or shuffle each modality and compare full multimodal performance, single-modality baselines, and a random-label control.

Current deterministic synthetic benchmark results are approximately:

- Visual attributes on unseen examples: 98.3% mean accuracy.
- Full multimodal danger judgment: 90.0%.
- Best single-modality danger judgment: 85.6%.
- Visual accuracy with V1 silenced: 41.7%, near the mixed task baselines.
- Random-label control: 40.2% average accuracy.

These results support genuine learning within the generated data distribution; they do not establish real-world generalization.

## Run the English showcase

Python 3.12 is recommended; Python 3.11–3.13 is supported.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app_en.py
```

Open <http://127.0.0.1:7861>. The first page load trains the synthetic Decoder once; later requests reuse it.

To run the Chinese version instead:

```powershell
python app.py
```

Then open <http://127.0.0.1:7860>.

## Optional CUDA 12 / JAX backend

The current scale runs comfortably on CPU. For CUDA 12:

```bash
pip install -r requirements-cuda12.txt
python -c "import jax; print(jax.devices())"
```

The runtime uses JAX/JIT when available and otherwise reports a NumPy CPU fallback.

## Public interface

```python
from cortex_demo import decode_multimodal, generate_shape

image = generate_shape("down", "sharp", "complete", seed=7)
prediction, bundle, retina, voltages, metrics, metadata = decode_multimodal(
    image, area=.2, force=.8, pain=.8, hardness=.9,
    metallic=.9, seed=42,
)
print(prediction.danger)
print(prediction.modality_weights)
print(bundle.require("vision_v1").spikes.shape)  # (200, 64)
```

Every modality is represented by `ModalitySignal(name, spikes)`, where `spikes` has the shape `timesteps × channels`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers reproducibility, V1 orientation selectivity, modality isolation, gate normalization, conflicting inputs, causal ablations, random-label controls, and held-out accuracy.

## Known limitations

- V1 contains only 64 simplified retinotopic neurons.
- Tactile and olfactory signals are abstract population encodings, not biological receptor models.
- All concepts and labels come from a generated closed-world distribution.
- There is no continual learning, memory, attention feedback, or open-world concept formation yet.
