"""
Unit tests for LMModelState class.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.ai_sleep.model_state import LMModelState, SleepMode, StateSnapshot


class TestSleepMode(unittest.TestCase):
    """Tests for SleepMode enumeration."""

    def test_sleep_modes_exist(self):
        """Test that all expected sleep modes exist."""
        self.assertEqual(SleepMode.AWAKE.value, "awake")
        self.assertEqual(SleepMode.LIGHT_SLEEP.value, "light_sleep")
        self.assertEqual(SleepMode.DEEP_SLEEP.value, "deep_sleep")
        self.assertEqual(SleepMode.TRANSITIONING.value, "transitioning")


class TestLMModelState(unittest.TestCase):
    """Tests for LMModelState class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_id = "test-model"
        self.state = LMModelState(model_id=self.model_id)

    def test_initialization(self):
        """Test model state initialization."""
        self.assertEqual(self.state.model_id, self.model_id)
        self.assertEqual(self.state.current_mode, SleepMode.AWAKE)
        self.assertEqual(len(self.state.state_history), 0)
        self.assertEqual(len(self.state.kv_cache), 0)

    def test_valid_transitions(self):
        """Test valid sleep mode transitions."""
        # Awake -> Light Sleep
        self.state.transition_to(SleepMode.LIGHT_SLEEP)
        self.assertEqual(self.state.current_mode, SleepMode.LIGHT_SLEEP)

        # Light Sleep -> Deep Sleep
        self.state.transition_to(SleepMode.DEEP_SLEEP)
        self.assertEqual(self.state.current_mode, SleepMode.DEEP_SLEEP)

        # Deep Sleep -> Light Sleep
        self.state.transition_to(SleepMode.LIGHT_SLEEP)
        self.assertEqual(self.state.current_mode, SleepMode.LIGHT_SLEEP)

        # Light Sleep -> Awake
        self.state.transition_to(SleepMode.AWAKE)
        self.assertEqual(self.state.current_mode, SleepMode.AWAKE)

    def test_invalid_transitions(self):
        """Test that invalid transitions raise errors."""
        # Awake -> Deep Sleep (must go through Light Sleep)
        with self.assertRaises(ValueError):
            self.state.transition_to(SleepMode.DEEP_SLEEP)

    def test_kv_cache_operations(self):
        """Test KV cache management."""
        layer_id = "layer_0"
        keys = np.random.rand(10, 64)
        values = np.random.rand(10, 64)

        # Update cache
        self.state.update_kv_cache(layer_id, keys, values)
        self.assertIn(layer_id, self.state.kv_cache)
        self.assertTrue(np.array_equal(self.state.kv_cache[layer_id]["keys"], keys))

        # Clear cache
        self.state.clear_kv_cache([layer_id])
        self.assertNotIn(layer_id, self.state.kv_cache)

    def test_attention_head_pruning(self):
        """Test attention head pruning."""
        layer_id = "layer_0"
        heads_to_prune = [0, 2, 5]

        self.state.prune_attention_heads(layer_id, heads_to_prune)

        self.assertIn(layer_id, self.state.attention_heads)
        self.assertEqual(self.state.attention_heads[layer_id]["pruned"], heads_to_prune)

    def test_layer_norm_update(self):
        """Test layer normalization updates."""
        layer_id = "layer_0"
        gamma = 1.1
        beta = 0.05

        self.state.update_layer_norm(layer_id, gamma, beta)

        self.assertIn(layer_id, self.state.layer_norms)
        self.assertEqual(self.state.layer_norms[layer_id]["gamma"], gamma)
        self.assertEqual(self.state.layer_norms[layer_id]["beta"], beta)

    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        # Store gradients
        self.state.store_gradient("param1", np.array([5.0, 5.0, 5.0]))
        self.state.store_gradient("param2", np.array([0.1, 0.2, 0.3]))

        # Clip gradients
        original_norms = self.state.clip_gradients(max_norm=1.0)

        # Check that large gradient was clipped
        self.assertGreater(original_norms["param1"], 1.0)
        self.assertLessEqual(np.linalg.norm(self.state.gradients["param1"]), 1.01)

        # Check that small gradient was not clipped
        self.assertLess(original_norms["param2"], 1.0)

    def test_semantic_memory_consolidation(self):
        """Test semantic memory consolidation."""
        key = "memory_1"
        value = {"data": "test_data"}

        self.state.consolidate_semantic_memory(key, value)

        self.assertIn(key, self.state.semantic_memory)
        self.assertEqual(self.state.semantic_memory[key]["value"], value)

    def test_metric_recording(self):
        """Test metric recording."""
        metric_name = "perplexity"
        values = [3.5, 3.3, 3.1]

        for value in values:
            self.state.record_metric(metric_name, value)

        self.assertIn(metric_name, self.state.performance_metrics)
        self.assertEqual(self.state.performance_metrics[metric_name], values)

    def test_state_snapshot(self):
        """Test state snapshot creation."""
        self.state.record_metric("loss", 0.5)
        snapshot = self.state.create_snapshot()

        self.assertIsInstance(snapshot, StateSnapshot)
        self.assertEqual(snapshot.sleep_mode, SleepMode.AWAKE)
        self.assertIn("loss", snapshot.metrics)

    def test_save_and_load_state(self):
        """Test state serialization and deserialization."""
        # Prepare state
        self.state.transition_to(SleepMode.LIGHT_SLEEP)
        self.state.record_metric("loss", 0.5)
        self.state.update_layer_norm("layer_0", 1.1, 0.05)

        # Save state
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            temp_path = f.name

        try:
            self.state.save_state(temp_path)

            # Load state
            loaded_state = LMModelState.load_state(temp_path)

            # Verify
            self.assertEqual(loaded_state.model_id, self.model_id)
            self.assertEqual(loaded_state.current_mode, SleepMode.LIGHT_SLEEP)
            self.assertIn("layer_0", loaded_state.layer_norms)
            self.assertEqual(loaded_state.layer_norms["layer_0"]["gamma"], 1.1)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_state_summary(self):
        """Test state summary generation."""
        summary = self.state.get_state_summary()

        self.assertIn("model_id", summary)
        self.assertIn("current_mode", summary)
        self.assertIn("sleep_cycle_count", summary)
        self.assertEqual(summary["model_id"], self.model_id)


if __name__ == "__main__":
    unittest.main()
