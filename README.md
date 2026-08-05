# 微型多模态感知回路

一个面向教学与作品展示的脉冲神经网络项目。当前版本把视觉、触觉和嗅觉信号串联/汇合成下面这条最小感知回路：

```text
图形 → 视网膜泊松脉冲 → 空间化 V1 LIF 脉冲 ┐
面积/力度/疼痛/硬度 → 触觉模块脉冲          ├→ 可训练多任务 Decoder → 属性与危险判断
金属气味 → 嗅觉脉冲                         ┘
```

视觉 Decoder 不再直接读取原图、生成参数或正确标签，而是读取 64 个 V1 神经元的输出脉冲。独立触觉模块把面积、力度、疼痛、硬度编码为 64 路 ON/OFF 群体脉冲，嗅觉模块提供 16 路气味脉冲，再通过通用 `MultimodalSignalBundle` 汇入 Decoder。因此以后可以按同样接口添加听觉、温度等模块。

> 这是教学性简化模型，不对应具体动物、脑区或临床结论。“危险”只是合成训练数据里的基础关联，不等于意识、真实语义理解或可靠安全判断。

## v0.6：独立编码器与可信门控融合

视觉、触觉和嗅觉现在分别进入自己的32维可训练编码器。方向、边缘和轮廓头只能读取视觉表征；四个触觉头只能读取触觉表征；气味头只能读取嗅觉表征。只有危险判断头读取三种表征经过softmax门控加权后的融合结果，因此可以显示当前判断对三种感觉的贡献权重。

训练集包含正常、感觉冲突、单模态缺失和高噪声场景，独立测试集使用不同随机种子。页面提供“看似尖锐但柔软”“看似圆滑但高压疼痛”等冲突预设。验证区分别切断或打乱V1、触觉和嗅觉，并加入随机标签训练对照；这可以检查对应属性是否只依赖正确模态，以及网络是否只是记忆固定规律。

## v0.5：单页真实串联与因果验证

界面不再把 V1 和 Decoder 分成两个各自运行的页面。点击一次“运行完整感知回路”，同一次计算产生的 V1 脉冲会直接成为 Decoder 的视觉输入；随后才与触觉、嗅觉模块汇合。

页面底部提供可重复的因果消融实验：在未参与训练的合成样本上比较正常 V1、切断 V1 和随机打乱 V1 通道。固定种子会逐脉冲复现；当前基准中正常 V1 的三个视觉任务平均准确率约 99%，切断后约 42%（对应四分类随机基线 25%、二分类随机基线 50%）。这证明当前 Decoder 的视觉预测确实依赖 V1 信号，而不是读取生成标签；它仍只证明在当前合成分布上的学习。

## 当前能展示什么

- 单页视觉通路：16×16 刺激、视网膜脉冲、四类方向选择性 V1 群体和活动热图。
- 多模态输入区：视觉输入、触觉输入模块（面积、力度、疼痛、硬度）及嗅觉输入模块。
- 各感觉模块输出保持分区：V1 只产生视觉信号；触觉模块产生四类触觉群体活动；嗅觉模块产生气味信号。
- 融合 Decoder 输出九项结果：方向、尖锐度、完整性、面积、力度、疼痛、硬度、金属气味和危险程度。
- 示例关联：疼痛会提高危险标签；“向下 + 尖锐 + 金属气味”也被训练为危险线索。

Decoder 使用纯 NumPy 反向传播实现三个独立编码器、可训练门控和九个输出头。首次进入页面时使用合成样本训练，之后在同一进程中复用。修改数据与训练参数后可重新训练，所以它是可更新网络，不是写死在界面里的答案。

## 快速启动

推荐 Python 3.12；支持 Python 3.11–3.13。Python 3.9 已结束官方支持，不作为新部署环境。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Ubuntu / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

然后打开终端显示的本地地址，通常是 <http://127.0.0.1:7860>。首次打开 Decoder 会生成训练数据并训练数秒，后续操作会复用模型。

如果 pip 显示正在从源码生成 NumPy，请先升级 pip，并使用本项目最新的 `requirements.txt`；正常情况下应下载预编译 wheel，不需要本地编译 NumPy。

## CUDA 12 / JAX（可选）

普通 CPU 足以运行当前规模。CUDA 12 环境可安装：

```bash
pip install -r requirements-cuda12.txt
python -c "import jax; print(jax.devices())"
```

后端默认是 `auto`：JAX 可用时使用 JIT，否则明确回退 NumPy。可用 `CORTEX_BACKEND=jax` 强制要求 JAX，避免演示时静默回退。

NeurAI 当前公开安装文档仍主要面向旧版 Python，因此没有放进现代 Python 主环境，也不会误报为当前运行后端。未来可通过独立兼容容器增加适配器。

## 公开接口

```python
from cortex_demo import decode_multimodal, generate_shape

image = generate_shape("down", "sharp", "complete", seed=7)
prediction, bundle, retina, voltages, metrics, metadata = decode_multimodal(
    image, area=0.2, force=0.8, pain=0.8, hardness=0.9,
    metallic=0.9, seed=42
)
print(prediction.danger)
print(prediction.modality_weights)  # vision / touch / odor，权重和为1
print(bundle.require("vision_v1").spikes.shape)  # (200, 64)
```

每种输入都用 `ModalitySignal(name, spikes)` 表示，其中 `spikes` 的形状为“时间步 × 通道”。`MultimodalSignalBundle` 是解码器与各感觉模块之间的稳定接口。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖刺激范围、随机种子复现、亮度单调性、V1方向选择性、模型复用、独立模态属性头、门控权重、冲突场景、三模态消融、随机标签对照及验证集准确率。

## 项目结构

- `app.py`：Gradio 双页面界面。
- `cortex_demo/model.py`：空间化方向选择性 LIF V1。
- `cortex_demo/multimodal.py`：视觉、触觉、嗅觉编码与通用信号容器。
- `cortex_demo/decoder.py`：信号融合、六头可训练 MLP 与危险关联。
- `cortex_demo/stimuli.py`、`shapes.py`：刺激和合成图形。
- `cortex_demo/plots.py`：神经活动与预测可视化。
- `tests/`：核心验收测试。

## 已知边界

- V1 只有 64 个教学性空间神经元，不是生物精度皮层模型。
- 触觉四变量与嗅觉目前仍是抽象的群体脉冲编码，不是皮肤、脊髓或嗅球的生物精度模型。
- 危险概念来自人为定义的合成标签；模型只会学习当前训练分布中的关联。
- 当前没有持续在线学习、记忆、注意力、反馈连接或开放世界概念形成。
- 新模态虽然有统一接口，仍需增加对应特征和训练数据后才能被 Decoder 利用。
