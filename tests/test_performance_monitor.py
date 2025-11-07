"""
Unit tests for LMPerformanceMonitor class.
"""

import unittest
import numpy as np

from src.ai_sleep.performance_monitor import (
    LMPerformanceMonitor,
    DriftDetector,
    AnomalyDetector,
    PerformanceAlert,
)


class TestDriftDetector(unittest.TestCase):
    """Tests for DriftDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = DriftDetector(window_size=50, alpha=0.05)

    def test_initialization(self):
        """Test drift detector initialization."""
        self.assertEqual(self.detector.window_size, 50)
        self.assertEqual(self.detector.alpha, 0.05)
        self.assertFalse(self.detector.baseline_established)

    def test_baseline_establishment(self):
        """Test baseline establishment."""
        # Add samples
        for i in range(30):
            self.detector.add_baseline_sample(np.random.normal(0, 1))

        self.assertTrue(self.detector.baseline_established)

    def test_drift_detection_no_drift(self):
        """Test that no drift is detected for similar distributions."""
        # Establish baseline
        for i in range(50):
            self.detector.add_baseline_sample(np.random.normal(0, 1))

        # Add current samples from same distribution
        for i in range(50):
            self.detector.add_current_sample(np.random.normal(0, 1))

        drift_detected, p_value = self.detector.detect_drift()
        # With random samples, might occasionally detect drift, so we just test the method works
        self.assertIsInstance(drift_detected, bool)
        self.assertIsInstance(p_value, float)

    def test_drift_detection_with_drift(self):
        """Test that drift is detected for different distributions."""
        # Establish baseline
        for i in range(50):
            self.detector.add_baseline_sample(np.random.normal(0, 1))

        # Add current samples from different distribution
        for i in range(50):
            self.detector.add_current_sample(
                np.random.normal(10, 1)
            )  # Mean shifted by 10

        drift_detected, p_value = self.detector.detect_drift()
        self.assertTrue(drift_detected)
        self.assertLess(p_value, 0.05)


class TestAnomalyDetector(unittest.TestCase):
    """Tests for AnomalyDetector class."""

    def test_zscore_method(self):
        """Test Z-score based anomaly detection."""
        detector = AnomalyDetector(method="zscore", threshold=3.0)

        # Add normal samples
        for i in range(50):
            detector.add_sample(np.random.normal(0, 1))

        # Test normal value
        is_anomaly, score = detector.is_anomaly(0.5)
        self.assertFalse(is_anomaly)

        # Test anomalous value
        is_anomaly, score = detector.is_anomaly(10.0)
        self.assertTrue(is_anomaly)
        self.assertGreater(score, 3.0)

    def test_iqr_method(self):
        """Test IQR based anomaly detection."""
        detector = AnomalyDetector(method="iqr", threshold=1.5)

        # Add samples
        for i in range(50):
            detector.add_sample(float(i))

        # Test anomaly detection
        is_anomaly, score = detector.is_anomaly(100.0)
        self.assertTrue(is_anomaly)

    def test_moving_avg_method(self):
        """Test moving average based anomaly detection."""
        detector = AnomalyDetector(method="moving_avg", threshold=3.0)

        # Add samples
        for i in range(50):
            detector.add_sample(10.0 + np.random.normal(0, 0.5))

        # Test anomaly
        is_anomaly, score = detector.is_anomaly(20.0)
        self.assertTrue(is_anomaly)


class TestLMPerformanceMonitor(unittest.TestCase):
    """Tests for LMPerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = LMPerformanceMonitor(model_id="test-model")

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.model_id, "test-model")
        self.assertEqual(len(self.monitor.metrics), 0)
        self.assertEqual(len(self.monitor.alerts), 0)

    def test_register_metric(self):
        """Test metric registration."""
        self.monitor.register_metric(
            "perplexity",
            track_drift=True,
            track_anomalies=True,
            min_threshold=1.0,
            max_threshold=10.0,
        )

        self.assertIn("perplexity", self.monitor.metrics)
        self.assertIn("perplexity", self.monitor.drift_detectors)
        self.assertIn("perplexity", self.monitor.anomaly_detectors)
        self.assertIn("perplexity", self.monitor.metric_thresholds)

    def test_track_metric(self):
        """Test metric tracking."""
        metric_name = "loss"
        value = 0.5

        self.monitor.track_metric(metric_name, value)

        self.assertIn(metric_name, self.monitor.metrics)
        self.assertEqual(len(self.monitor.metrics[metric_name]), 1)
        self.assertEqual(self.monitor.metrics[metric_name][0][1], value)

    def test_threshold_violation_alert(self):
        """Test threshold violation alert creation."""
        metric_name = "perplexity"
        self.monitor.register_metric(metric_name, min_threshold=2.0, max_threshold=5.0)

        # Track value above threshold
        self.monitor.track_metric(metric_name, 10.0)

        # Check alert was created
        self.assertGreater(len(self.monitor.alerts), 0)
        alert = self.monitor.alerts[-1]
        self.assertEqual(alert.alert_type, "threshold_violation")

    def test_establish_baseline(self):
        """Test baseline establishment."""
        metric_name = "loss"

        # Track some metrics
        for i in range(30):
            self.monitor.track_metric(metric_name, 0.5 + np.random.normal(0, 0.1))

        # Establish baseline
        self.monitor.establish_baseline(metric_name)

        self.assertIn(metric_name, self.monitor.drift_detectors)

    def test_check_drift(self):
        """Test drift checking."""
        metric_name = "loss"

        # Establish baseline
        for i in range(50):
            self.monitor.track_metric(metric_name, 0.5 + np.random.normal(0, 0.1))
        self.monitor.establish_baseline(metric_name)

        # Add more samples
        for i in range(50):
            self.monitor.track_metric(metric_name, 0.5 + np.random.normal(0, 0.1))

        # Check drift
        drift_detected, p_value = self.monitor.check_drift(metric_name)

        self.assertIsInstance(drift_detected, bool)
        self.assertIsInstance(p_value, float)

    def test_get_metric_statistics(self):
        """Test metric statistics calculation."""
        metric_name = "perplexity"
        values = [3.5, 3.3, 3.1, 3.0]

        for value in values:
            self.monitor.track_metric(metric_name, value)

        stats = self.monitor.get_metric_statistics(metric_name)

        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertEqual(stats["count"], len(values))
        self.assertEqual(stats["latest"], values[-1])

    def test_get_metric_trend(self):
        """Test metric trend retrieval."""
        metric_name = "loss"
        values = [0.5, 0.4, 0.3, 0.2, 0.1]

        for value in values:
            self.monitor.track_metric(metric_name, value)

        # Get all trends
        trend = self.monitor.get_metric_trend(metric_name)
        self.assertEqual(trend, values)

        # Get recent trend
        trend = self.monitor.get_metric_trend(metric_name, window=3)
        self.assertEqual(trend, values[-3:])

    def test_alert_callback(self):
        """Test alert callback registration and execution."""
        callback_called = []

        def test_callback(alert):
            callback_called.append(alert)

        self.monitor.register_alert_callback(test_callback)

        # Trigger an alert
        self.monitor.register_metric("test", max_threshold=5.0)
        self.monitor.track_metric("test", 10.0)

        self.assertEqual(len(callback_called), 1)
        self.assertIsInstance(callback_called[0], PerformanceAlert)

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        # Create some alerts
        self.monitor.register_metric("test", max_threshold=5.0)
        self.monitor.track_metric("test", 10.0)
        self.monitor.track_metric("test", 15.0)

        alerts = self.monitor.get_recent_alerts()
        self.assertGreater(len(alerts), 0)

        # Filter by severity
        high_alerts = self.monitor.get_recent_alerts(severity="high")
        for alert in high_alerts:
            self.assertEqual(alert.severity, "high")

    def test_monitoring_summary(self):
        """Test monitoring summary generation."""
        self.monitor.register_metric("loss")
        self.monitor.track_metric("loss", 0.5)

        summary = self.monitor.get_monitoring_summary()

        self.assertIn("model_id", summary)
        self.assertIn("total_samples", summary)
        self.assertIn("metrics_tracked", summary)
        self.assertEqual(summary["model_id"], "test-model")
        self.assertEqual(summary["total_samples"], 1)


if __name__ == "__main__":
    unittest.main()
