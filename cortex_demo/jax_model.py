from __future__ import annotations

import time
import numpy as np

from . import settings
class JaxVisualCortexModel:
    """JIT-compiled single-device LIF kernel for the presentation environment."""

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self.initialization_count = 1
        devices = jax.devices()
        self.backend_name = f"JAX JIT ({devices[0].platform}: {devices[0].device_kind})"

        total = len(settings.ORIENTATIONS) * settings.NEURONS_PER_GROUP
        group_size = settings.NEURONS_PER_GROUP

        def simulate_kernel(input_spikes, dt_ms):
            initial = (
                jnp.full((total,), settings.V_REST, dtype=jnp.float32),
                jnp.zeros((total,), dtype=jnp.int16),
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((16, 16), dtype=jnp.float32),
            )

            def step(carry, input_spike):
                v, refractory, prior_rate, retina_trace = carry
                retina_trace = retina_trace * .82 + input_spike.reshape(16, 16)
                gx = jnp.zeros_like(retina_trace).at[:, 1:-1].set(
                    retina_trace[:, 2:] - retina_trace[:, :-2]
                )
                gy = jnp.zeros_like(retina_trace).at[1:-1, :].set(
                    retina_trace[2:, :] - retina_trace[:-2, :]
                )
                oriented = jnp.stack([
                    jnp.abs(gy),
                    jnp.sqrt(jnp.maximum(-gx * gy, 0)) * 3,
                    jnp.abs(gx),
                    jnp.sqrt(jnp.maximum(gx * gy, 0)) * 3,
                ])
                pooled = oriented.reshape(4, 4, 4, 4, 4).mean(axis=(2, 4)).reshape(total)
                drive = pooled * settings.GAIN * 100.0
                inhibition = jnp.repeat(prior_rate.sum() - prior_rate, group_size)
                inhibition = inhibition * settings.LATERAL_INHIBITION
                active = refractory <= 0
                dv = ((settings.V_REST - v) + drive - inhibition) * dt_ms / settings.TAU_MEMBRANE_MS
                v = jnp.where(active, v + dv, v)
                fired = active & (v >= settings.V_THRESHOLD)
                spike = fired.astype(jnp.float32)
                v = jnp.where(fired, settings.V_RESET, v)
                refractory = jnp.where(active, 0, refractory - 1)
                refractory = jnp.where(fired, settings.REFRACTORY_MS, refractory)
                rate = spike.reshape(4, group_size).mean(axis=1)
                return (v, refractory, rate, retina_trace), (spike, v)

            _, outputs = jax.lax.scan(step, initial, input_spikes)
            return outputs

        self._run = jax.jit(simulate_kernel)
        # Warm-up with the default static shape so subsequent UI runs reuse compilation.
        warmup = jnp.zeros((settings.DEFAULT_DURATION_MS, settings.INPUT_NEURONS), dtype=jnp.float32)
        self._run(warmup, settings.DEFAULT_DT_MS)[0].block_until_ready()

    def run(self, input_spikes: np.ndarray, dt_ms: float):
        started = time.perf_counter()
        spikes, voltage = self._run(self.jnp.asarray(input_spikes), dt_ms)
        spikes.block_until_ready()
        elapsed = (time.perf_counter() - started) * 1000
        return np.asarray(spikes), np.asarray(voltage), elapsed
