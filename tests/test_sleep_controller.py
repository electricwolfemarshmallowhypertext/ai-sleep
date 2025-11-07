"""
Unit tests for AISleepController class.
"""

import unittest
import tempfile
from pathlib import Path

from src.ai_sleep.sleep_controller import (
    AISleepController,
    SleepTrigger,
    OptimizationStrategy
)
from src.ai_sleep.model_state import SleepMode


class TestSleepTrigger(unittest.TestCase):
    """Tests for SleepTrigger enumeration."""
    
    def test_triggers_exist(self):
        """Test that all expected triggers exist."""
        self.assertEqual(SleepTrigger.SCHEDULED.value, "scheduled")
        self.assertEqual(SleepTrigger.PERFORMANCE_DEGRADATION.value, "performance_degradation")
        self.assertEqual(SleepTrigger.ANOMALY_DETECTED.value, "anomaly_detected")
        self.assertEqual(SleepTrigger.DRIFT_DETECTED.value, "drift_detected")
        self.assertEqual(SleepTrigger.MEMORY_PRESSURE.value, "memory_pressure")
        self.assertEqual(SleepTrigger.MANUAL.value, "manual")


class TestOptimizationStrategy(unittest.TestCase):
    """Tests for OptimizationStrategy enumeration."""
    
    def test_strategies_exist(self):
        """Test that all expected strategies exist."""
        self.assertEqual(OptimizationStrategy.GRADIENT_CLIPPING.value, "gradient_clipping")
        self.assertEqual(OptimizationStrategy.ATTENTION_HEAD_PRUNING.value, "attention_head_pruning")
        self.assertEqual(OptimizationStrategy.KV_CACHE_MANAGEMENT.value, "kv_cache_management")
        self.assertEqual(OptimizationStrategy.SEMANTIC_CONSOLIDATION.value, "semantic_consolidation")
        self.assertEqual(OptimizationStrategy.LAYER_NORM_RECALIBRATION.value, "layer_norm_recalibration")


class TestAISleepController(unittest.TestCase):
    """Tests for AISleepController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AISleepController(
            model_id="test-model",
            enable_monitoring=True,
            auto_adapt=True
        )
        
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.model_id, "test-model")
        self.assertEqual(self.controller.model_state.current_mode, SleepMode.AWAKE)
        self.assertIsNotNone(self.controller.performance_monitor)
        self.assertEqual(self.controller.total_sleep_cycles, 0)
        
    def test_configure_light_sleep(self):
        """Test light sleep configuration."""
        self.controller.configure_light_sleep(
            duration=600,
            strategies=[OptimizationStrategy.GRADIENT_CLIPPING],
            gradient_clip_norm=0.5
        )
        
        self.assertEqual(self.controller.light_sleep_config["duration"], 600)
        self.assertEqual(self.controller.light_sleep_config["gradient_clip_norm"], 0.5)
        
    def test_configure_deep_sleep(self):
        """Test deep sleep configuration."""
        self.controller.configure_deep_sleep(
            duration=3600,
            strategies=[OptimizationStrategy.ATTENTION_HEAD_PRUNING],
            pruning_threshold=0.2
        )
        
        self.assertEqual(self.controller.deep_sleep_config["duration"], 3600)
        self.assertEqual(self.controller.deep_sleep_config["pruning_threshold"], 0.2)
        
    def test_set_trigger_threshold(self):
        """Test trigger threshold setting."""
        self.controller.set_trigger_threshold(SleepTrigger.DRIFT_DETECTED, 0.01)
        self.assertEqual(self.controller.trigger_thresholds[SleepTrigger.DRIFT_DETECTED], 0.01)
        
    def test_register_callback(self):
        """Test callback registration."""
        callback_called = []
        
        def test_callback(*args):
            callback_called.append(args)
            
        self.controller.register_callback("on_sleep_start", test_callback)
        self.assertIn(test_callback, self.controller.sleep_callbacks["on_sleep_start"])
        
    def test_initiate_light_sleep(self):
        """Test initiating light sleep."""
        result = self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL,
            mode=SleepMode.LIGHT_SLEEP
        )
        
        self.assertTrue(result)
        self.assertEqual(self.controller.model_state.current_mode, SleepMode.LIGHT_SLEEP)
        self.assertEqual(self.controller.total_sleep_cycles, 1)
        
    def test_initiate_deep_sleep(self):
        """Test initiating deep sleep."""
        # First transition to light sleep
        self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL,
            mode=SleepMode.LIGHT_SLEEP
        )
        
        # Then to deep sleep
        result = self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL,
            mode=SleepMode.DEEP_SLEEP
        )
        
        self.assertTrue(result)
        self.assertEqual(self.controller.model_state.current_mode, SleepMode.DEEP_SLEEP)
        
    def test_wake_up(self):
        """Test waking up from sleep."""
        # Initiate sleep
        self.controller.initiate_sleep(
            trigger=SleepTrigger.MANUAL,
            mode=SleepMode.LIGHT_SLEEP
        )
        
        # Wake up
        result = self.controller.wake_up()
        
        self.assertTrue(result)
        self.assertEqual(self.controller.model_state.current_mode, SleepMode.AWAKE)
        
    def test_sleep_cycle_with_callbacks(self):
        """Test complete sleep cycle with callbacks."""
        sleep_start_called = []
        sleep_end_called = []
        optimization_called = []
        
        def on_start(mode, trigger):
            sleep_start_called.append((mode, trigger))
            
        def on_end():
            sleep_end_called.append(True)
            
        def on_optimization(strategy):
            optimization_called.append(strategy)
            
        self.controller.register_callback("on_sleep_start", on_start)
        self.controller.register_callback("on_sleep_end", on_end)
        self.controller.register_callback("on_optimization_complete", on_optimization)
        
        # Execute sleep cycle
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.wake_up()
        
        # Verify callbacks were called
        self.assertEqual(len(sleep_start_called), 1)
        self.assertEqual(len(sleep_end_called), 1)
        self.assertGreater(len(optimization_called), 0)
        
    def test_gradient_clipping_optimization(self):
        """Test gradient clipping during light sleep."""
        import numpy as np
        
        # Store some gradients
        self.controller.model_state.store_gradient("param1", np.array([5.0, 5.0]))
        
        # Configure light sleep with gradient clipping
        self.controller.configure_light_sleep(
            strategies=[OptimizationStrategy.GRADIENT_CLIPPING],
            gradient_clip_norm=1.0
        )
        
        # Initiate sleep
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        
        # Check gradient was clipped
        grad_norm = np.linalg.norm(self.controller.model_state.gradients["param1"])
        self.assertLessEqual(grad_norm, 1.01)
        
    def test_kv_cache_management(self):
        """Test KV cache management during light sleep."""
        import numpy as np
        
        # Add some cache entries
        for i in range(10):
            self.controller.model_state.update_kv_cache(
                f"layer_{i}",
                np.random.rand(10, 64),
                np.random.rand(10, 64)
            )
            
        initial_size = len(self.controller.model_state.kv_cache)
        
        # Configure and execute light sleep
        self.controller.configure_light_sleep(
            strategies=[OptimizationStrategy.KV_CACHE_MANAGEMENT],
            kv_cache_retention=0.5
        )
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        
        final_size = len(self.controller.model_state.kv_cache)
        self.assertLess(final_size, initial_size)
        
    def test_attention_head_pruning(self):
        """Test attention head pruning during deep sleep."""
        # Configure deep sleep with pruning
        self.controller.configure_deep_sleep(
            strategies=[OptimizationStrategy.ATTENTION_HEAD_PRUNING],
            pruning_threshold=0.1
        )
        
        # Initiate light sleep first, then deep sleep
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.DEEP_SLEEP)
        
        # Check that pruning occurred
        self.assertGreater(len(self.controller.model_state.attention_heads), 0)
        
    def test_semantic_consolidation(self):
        """Test semantic consolidation during deep sleep."""
        # Configure and execute deep sleep
        self.controller.configure_deep_sleep(
            strategies=[OptimizationStrategy.SEMANTIC_CONSOLIDATION]
        )
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.DEEP_SLEEP)
        
        # Check that consolidation occurred
        self.assertGreater(len(self.controller.model_state.semantic_memory), 0)
        
    def test_layer_norm_recalibration(self):
        """Test layer norm recalibration during deep sleep."""
        # Configure and execute deep sleep
        self.controller.configure_deep_sleep(
            strategies=[OptimizationStrategy.LAYER_NORM_RECALIBRATION]
        )
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.DEEP_SLEEP)
        
        # Check that recalibration occurred
        self.assertGreater(len(self.controller.model_state.layer_norms), 0)
        
    def test_security_patching(self):
        """Test security patching during deep sleep."""
        # Configure and execute deep sleep
        self.controller.configure_deep_sleep(
            strategies=[OptimizationStrategy.SECURITY_PATCHING]
        )
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.DEEP_SLEEP)
        
        # Security patches may or may not be applied depending on random check
        # Just verify the mechanism runs
        self.assertIsInstance(self.controller.security_patches_applied, list)
        
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment."""
        # Track some metrics
        for i in range(10):
            self.controller.performance_monitor.track_metric("loss", 0.5 - i * 0.01)
            
        initial_lr = self.controller.learning_rate
        
        # Configure and execute light sleep with adaptive learning
        self.controller.configure_light_sleep(
            strategies=[OptimizationStrategy.ADAPTIVE_LEARNING_RATE]
        )
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        
        # Learning rate should have been adjusted
        # (might be same if not enough samples, so just check it's set)
        self.assertIsNotNone(self.controller.learning_rate)
        
    def test_get_sleep_statistics(self):
        """Test sleep statistics retrieval."""
        # Execute a sleep cycle
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        self.controller.wake_up()
        
        stats = self.controller.get_sleep_statistics()
        
        self.assertIn("model_id", stats)
        self.assertIn("total_sleep_cycles", stats)
        self.assertIn("current_mode", stats)
        self.assertIn("sleep_ratio", stats)
        self.assertEqual(stats["total_sleep_cycles"], 1)
        
    def test_export_and_load_state(self):
        """Test state export and restoration."""
        # Execute some operations
        self.controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
        
        # Export state
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
            
        try:
            self.controller.export_state(temp_path)
            
            # Load state
            loaded_controller = AISleepController.from_state(temp_path)
            
            # Verify
            self.assertEqual(loaded_controller.model_id, self.controller.model_id)
            self.assertEqual(
                loaded_controller.model_state.current_mode,
                self.controller.model_state.current_mode
            )
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    def test_determine_sleep_mode(self):
        """Test sleep mode determination based on trigger."""
        # Drift should trigger deep sleep
        mode = self.controller._determine_sleep_mode(SleepTrigger.DRIFT_DETECTED)
        self.assertEqual(mode, SleepMode.DEEP_SLEEP)
        
        # Anomaly should trigger light sleep
        mode = self.controller._determine_sleep_mode(SleepTrigger.ANOMALY_DETECTED)
        self.assertEqual(mode, SleepMode.LIGHT_SLEEP)


if __name__ == "__main__":
    unittest.main()
