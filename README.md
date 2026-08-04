# 微型视觉皮层

一个面向作品展示的“视网膜 → V1 → 可训练Decoder”脉冲神经网络。页面既能展示泊松输入、方向选择性群体发放、膜电位与活动热图，也能从脉冲信号中学习解读基础视觉属性。

> 本项目是教学性简化模型，不对应具体动物、脑区或临床结论。Decoder只学习方向、尖锐度和轮廓完整性，不代表高级语义理解。

## v0.2：可学习的基础视觉Decoder

新增的“基础视觉解码”页面构成完整的最小学习闭环：

```text
程序化图形 → 泊松脉冲 → 空间/方向活动特征 → 多任务MLP → 基础视觉属性
```

Decoder包含一个共享隐藏层和三个分类头：

- 方向：上、下、左、右；
- 边缘形态：尖锐、圆滑；
- 轮廓状态：完整、有缺口。

训练数据由程序自动生成，包含位置、亮度、噪声和缺口位置变化。模型输入来自300 ms泊松脉冲，不会读取生成图形时使用的正确标签。默认固定训练集的验证结果约为：方向100%、边缘98%、轮廓92%。这些数字只描述合成数据分布，不能代表现实图片性能。

MLP使用纯NumPy实现反向传播，首次打开Decoder时自动训练一次并在进程内复用；修改数据或学习参数后可以重新训练，因而它是可更新的解码网络，而不是固定判断规则。

Decoder页面同时展示完整的信号路径：原始16×16图形、300 ms泊松脉冲栅格图、由脉冲重建的空间活动、水平/垂直/两种斜向边缘响应，以及三个语义输出头的置信度。这样可以直接观察“图形 → 电信号 → 边缘特征 → 基础含义”的转换过程。

## 快速启动

主应用目标环境为 Python 3.11–3.13。建议使用 Python 3.12；单张 NVIDIA GPU 为可选加速，普通 CPU 也能完成默认规模仿真。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Python 3.13 必须使用提供预编译 wheel 的 NumPy 2.x。若 pip 显示正在生成或编译 NumPy，请先升级 pip，并确认使用的是本项目最新版 `requirements.txt`，不要继续等待本地源码编译。

浏览器访问 Gradio 输出的本地地址。首次打开会运行一次仿真，后续请求复用同一个模型实例。

## CUDA 12 / JAX 环境

```bash
pip install -r requirements-cuda12.txt
python -c "import jax; print(jax.devices())"
```

`requirements-cuda12.txt` 使用 JAX 的 CUDA 12 wheel。安装后应确认 `jax.devices()` 输出的是 GPU，而不是 CPU。

当前提交包含 NumPy 参考计算核和 JAX JIT 计算核，两者共享公开接口。运行结果的 metadata 会明确显示实际后端。

后端默认采用 `auto`：能初始化 JAX 就使用 JIT 后端，否则回退 NumPy 并在页面提示。正式 GPU 演示可强制要求 JAX，避免静默回退：

```bash
export CORTEX_BACKEND=jax
python app.py
```

## NeurAI 兼容性说明

Python 3.9 已于 2025-10-31 结束官方支持，不应作为新部署的主运行时。NeurAI 当前公开安装文档仍只列出 GPU/CPU 支持 Python 3.8/3.9，因此本项目不会把 NeurAI 安装进主应用环境，也不会误报为已使用 NeurAI 后端。

如必须验证 NeurAI API，应使用南湖提供的官方镜像或独立的、无公网暴露的旧版兼容容器，并仅通过文件或明确的数据接口与主应用交换结果。不要在该容器中运行公开 Gradio 服务；待 NeurAI 发布支持受维护 Python 版本的包后，再增加原生适配器。

## 演示流程

1. 选择“垂直条纹”，点击“开始仿真”。
2. 查看输入栅格图，说明亮像素被转换成更密集的泊松脉冲。
3. 查看方向群体柱状图；90°群体应最活跃。
4. 调低亮度或增加噪声，观察响应强度和稳定性变化。
5. 切换其他条纹，比较最活跃群体。

“角度”滑杆会覆盖条纹预设方向；亮点和空白刺激不使用该参数。

## 公开接口

```python
from cortex_demo import StimulusConfig, simulate

result = simulate(StimulusConfig(
    stimulus_type="vertical",
    angle_deg=90,
    brightness=0.8,
    noise=0.05,
    duration_ms=200,
    seed=42,
))
print(result.group_rates_hz, result.metadata)
```

`SimulationResult` 包含 `stimulus`、`times_ms`、`input_spikes`、`v1_spikes`、`membrane_voltage`、`group_rates_hz` 和 `metadata`。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖刺激范围、泊松随机种子、亮度单调性、空白基线、垂直方向选择性、无重复初始化、形状生成、Decoder输出和验证集准确率。

## 项目结构

- `app.py`：Gradio 双栏展示页。
- `cortex_demo/api.py`：稳定的输入/输出数据接口。
- `cortex_demo/stimuli.py`：几何刺激与泊松编码。
- `cortex_demo/model.py`：固定 Gabor 权重和 LIF 动力学。
- `cortex_demo/plots.py`：五类展示图表。
- `cortex_demo/shapes.py`：带标签的方向、尖锐度和轮廓图形生成器。
- `cortex_demo/decoder.py`：脉冲特征提取与可训练多任务MLP。
- `tests/`：无需 Gradio/JAX 的核心验收测试。

## 已知边界

- 固定 Gabor 风格连接用于可解释的方向选择性，不代表真实皮层连接组。
- 无 STDP、反向传播、数字识别、多机分布式或 KA200 部署。
- 默认网络很小；GPU 的价值主要是后续扩展和 JIT，而不是当前规模的必要条件。
- GPU 性能目标需在实际目标机器完成预热后验收；开发机测试不替代该项测试。
