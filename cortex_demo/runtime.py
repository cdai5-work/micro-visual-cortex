from __future__ import annotations

import os

from .model import MODEL as NUMPY_MODEL

_MODEL = None
_FALLBACK_REASON = None


def get_model():
    global _MODEL, _FALLBACK_REASON
    if _MODEL is not None:
        return _MODEL, _FALLBACK_REASON

    requested = os.getenv("CORTEX_BACKEND", "auto").lower()
    if requested not in {"auto", "jax", "numpy"}:
        raise ValueError("CORTEX_BACKEND 只能是 auto、jax 或 numpy")
    if requested != "numpy":
        try:
            from .jax_model import JaxVisualCortexModel
            _MODEL = JaxVisualCortexModel()
            return _MODEL, None
        except Exception as exc:
            if requested == "jax":
                raise RuntimeError(
                    "JAX GPU 后端初始化失败；请检查 CUDA/JAX 安装，或设置 CORTEX_BACKEND=numpy 回退"
                ) from exc
            _FALLBACK_REASON = f"JAX 不可用，已回退 CPU：{type(exc).__name__}"
    _MODEL = NUMPY_MODEL
    return _MODEL, _FALLBACK_REASON

