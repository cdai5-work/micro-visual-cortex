import unittest
import numpy as np

from cortex_demo import StimulusConfig, simulate
from cortex_demo.model import MODEL
from cortex_demo.stimuli import generate_stimulus, poisson_encode


class StimulusTests(unittest.TestCase):
    def test_all_stimuli_are_bounded_16_square(self):
        for kind in ("spot", "horizontal", "vertical", "diagonal_left", "diagonal_right", "blank"):
            image = generate_stimulus(kind, seed=1)
            self.assertEqual(image.shape, (16, 16))
            self.assertGreaterEqual(float(image.min()), 0)
            self.assertLessEqual(float(image.max()), 1)

    def test_poisson_seed_is_reproducible(self):
        image = generate_stimulus("vertical", noise=0)
        a = poisson_encode(image, 200, 1, 100, 12)
        b = poisson_encode(image, 200, 1, 100, 12)
        np.testing.assert_array_equal(a, b)

    def test_more_brightness_means_no_less_input_activity(self):
        low = generate_stimulus("vertical", brightness=.2, noise=0)
        high = generate_stimulus("vertical", brightness=.9, noise=0)
        # Use the same uniforms through an identical seed.
        self.assertLessEqual(poisson_encode(low, 1000, 1, 100, 3).sum(),
                             poisson_encode(high, 1000, 1, 100, 3).sum())


class ModelTests(unittest.TestCase):
    def test_vertical_prefers_90_degree_group(self):
        result = simulate(StimulusConfig("vertical", angle_deg=90, noise=0,
                                         duration_ms=500, seed=4))
        self.assertEqual(max(result.group_rates_hz, key=result.group_rates_hz.get), 90)

    def test_blank_is_near_baseline(self):
        result = simulate(StimulusConfig("blank", brightness=0, noise=0,
                                         duration_ms=200, seed=4))
        self.assertEqual(sum(result.group_rates_hz.values()), 0)

    def test_model_is_reused(self):
        before = MODEL.initialization_count
        simulate(StimulusConfig(duration_ms=20))
        simulate(StimulusConfig(duration_ms=20))
        self.assertEqual(MODEL.initialization_count, before)

    def test_invalid_stimulus_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "未知刺激类型"):
            simulate(StimulusConfig("not-a-stimulus"))

    def test_all_showcase_stimuli_finish(self):
        for kind in ("spot", "horizontal", "vertical", "diagonal_left", "diagonal_right"):
            result = simulate(StimulusConfig(kind, noise=0, duration_ms=50, seed=8))
            self.assertEqual(result.v1_spikes.shape, (50, 128))
            self.assertEqual(set(result.group_rates_hz), {0, 45, 90, 135})


if __name__ == "__main__":
    unittest.main()
