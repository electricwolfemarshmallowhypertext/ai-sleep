"""
Unit tests for HuggingFace integration.
"""

import unittest
import warnings

from src.ai_sleep.huggingface_integration import (
    HuggingFaceModelAdapter,
    create_sleep_enabled_model,
    configure_optimal_sleep_schedule
)
from src.ai_sleep.model_state import SleepMode


class TestHuggingFaceModelAdapter(unittest.TestCase):
    """Tests for HuggingFaceModelAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use None for model to avoid requiring transformers library
        self.adapter = HuggingFaceModelAdapter(
            model=None,
            model_id="test-model",
            enable_monitoring=True,
            auto_adapt=True
        )
        
    def test_initialization(self):
        """Test adapter initialization."""
        self.assertEqual(self.adapter.model_id, "test-model")
        self.assertIsNotNone(self.adapter.controller)
        self.assertFalse(self.adapter.sleep_enabled)
        
    def test_enable_sleep_cycles(self):
        """Test enabling sleep cycles."""
        self.adapter.enable_sleep_cycles()
        self.assertTrue(self.adapter.sleep_enabled)
        
    def test_disable_sleep_cycles(self):
        """Test disabling sleep cycles."""
        self.adapter.enable_sleep_cycles()
        self.adapter.disable_sleep_cycles()
        self.assertFalse(self.adapter.sleep_enabled)
        
    def test_initiate_light_sleep_without_enable(self):
        """Test that light sleep warns if not enabled."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.adapter.initiate_light_sleep()
            self.assertFalse(result)
            self.assertEqual(len(w), 1)
            
    def test_initiate_light_sleep_with_enable(self):
        """Test light sleep initiation when enabled."""
        self.adapter.enable_sleep_cycles()
        result = self.adapter.initiate_light_sleep(duration=300)
        
        self.assertTrue(result)
        self.assertEqual(self.adapter.controller.model_state.current_mode, SleepMode.LIGHT_SLEEP)
        
    def test_initiate_deep_sleep_with_enable(self):
        """Test deep sleep initiation when enabled."""
        self.adapter.enable_sleep_cycles()
        
        # First go to light sleep
        self.adapter.initiate_light_sleep()
        
        # Then deep sleep
        result = self.adapter.initiate_deep_sleep(duration=1800)
        
        self.assertTrue(result)
        self.assertEqual(self.adapter.controller.model_state.current_mode, SleepMode.DEEP_SLEEP)
        
    def test_wake_model(self):
        """Test waking the model."""
        self.adapter.enable_sleep_cycles()
        self.adapter.initiate_light_sleep()
        
        result = self.adapter.wake_model()
        
        self.assertTrue(result)
        self.assertEqual(self.adapter.controller.model_state.current_mode, SleepMode.AWAKE)
        
    def test_track_inference_metrics(self):
        """Test tracking inference metrics."""
        self.adapter.track_inference_metrics(
            perplexity=3.5,
            loss=0.5,
            inference_time=0.1,
            memory_usage=512.0
        )
        
        monitor = self.adapter.controller.performance_monitor
        self.assertIn("perplexity", monitor.metrics)
        self.assertIn("loss", monitor.metrics)
        self.assertIn("inference_time", monitor.metrics)
        self.assertIn("memory_usage", monitor.metrics)
        
    def test_get_model_status(self):
        """Test getting model status."""
        self.adapter.enable_sleep_cycles()
        self.adapter.track_inference_metrics(loss=0.5)
        
        status = self.adapter.get_model_status()
        
        self.assertIn("model_id", status)
        self.assertIn("sleep_enabled", status)
        self.assertIn("current_mode", status)
        self.assertIn("sleep_statistics", status)
        self.assertIn("monitoring", status)
        self.assertTrue(status["sleep_enabled"])


class TestCreateSleepEnabledModel(unittest.TestCase):
    """Tests for create_sleep_enabled_model function."""
    
    def test_create_without_loading(self):
        """Test creating adapter without loading actual model."""
        adapter = create_sleep_enabled_model(
            "gpt2",
            enable_monitoring=True,
            auto_adapt=True,
            load_model=False
        )
        
        self.assertIsInstance(adapter, HuggingFaceModelAdapter)
        self.assertEqual(adapter.model_id, "gpt2")
        self.assertIsNone(adapter.model)
        
    def test_create_with_loading_mock(self):
        """Test creating adapter with load_model=True (will use mock)."""
        # This will warn about transformers not available, but should work
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = create_sleep_enabled_model(
                "gpt2",
                load_model=True
            )
            
        self.assertIsInstance(adapter, HuggingFaceModelAdapter)


class TestConfigureOptimalSleepSchedule(unittest.TestCase):
    """Tests for configure_optimal_sleep_schedule function."""
    
    def test_continuous_workload(self):
        """Test configuration for continuous workload."""
        adapter = create_sleep_enabled_model("test-model", load_model=False)
        configure_optimal_sleep_schedule(adapter, workload_type="continuous")
        
        # Check that configuration was applied
        self.assertIn("duration", adapter.controller.light_sleep_config)
        self.assertIn("strategies", adapter.controller.light_sleep_config)
        
    def test_batch_workload(self):
        """Test configuration for batch workload."""
        adapter = create_sleep_enabled_model("test-model", load_model=False)
        configure_optimal_sleep_schedule(adapter, workload_type="batch")
        
        # Batch should have longer durations
        self.assertEqual(adapter.controller.light_sleep_config["duration"], 600)
        self.assertEqual(adapter.controller.deep_sleep_config["duration"], 3600)
        
    def test_interactive_workload(self):
        """Test configuration for interactive workload."""
        adapter = create_sleep_enabled_model("test-model", load_model=False)
        configure_optimal_sleep_schedule(adapter, workload_type="interactive")
        
        # Interactive should have shorter durations
        self.assertEqual(adapter.controller.light_sleep_config["duration"], 60)
        self.assertEqual(adapter.controller.deep_sleep_config["duration"], 600)


if __name__ == "__main__":
    unittest.main()
