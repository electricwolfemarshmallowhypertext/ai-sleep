"""
LMPerformanceMonitor: Performance tracking and anomaly detection for language models.

This module provides drift detection, anomaly detection, and comprehensive performance
monitoring for language models during sleep optimization cycles.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class PerformanceAlert:
    """Represents a performance alert or anomaly detection."""

    timestamp: datetime
    alert_type: str
    severity: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    metadata: Dict[str, Any]


class DriftDetector:
    """
    Detects performance drift using statistical methods.

    Uses Kolmogorov-Smirnov test and rolling statistics to identify
    when model performance characteristics have drifted from baseline.
    """

    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        """
        Initialize drift detector.

        Args:
            window_size: Number of samples to consider for drift detection
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.baseline_samples: deque = deque(maxlen=window_size)
        self.current_samples: deque = deque(maxlen=window_size)
        self.baseline_established = False

    def add_baseline_sample(self, value: float) -> None:
        """Add a sample to the baseline distribution."""
        self.baseline_samples.append(value)
        if len(self.baseline_samples) >= self.window_size // 2:
            self.baseline_established = True

    def add_current_sample(self, value: float) -> None:
        """Add a sample to the current distribution."""
        self.current_samples.append(value)

    def detect_drift(self) -> Tuple[bool, float]:
        """
        Detect if drift has occurred between baseline and current distributions.

        Returns:
            Tuple of (drift_detected, p_value)
        """
        if not self.baseline_established or len(self.current_samples) < 30:
            return False, 1.0

        # Perform Kolmogorov-Smirnov test
        baseline_array = np.array(list(self.baseline_samples))
        current_array = np.array(list(self.current_samples))

        statistic, p_value = stats.ks_2samp(baseline_array, current_array)

        drift_detected = bool(p_value < self.alpha)
        return drift_detected, float(p_value)

    def reset_current(self) -> None:
        """Reset current samples (e.g., after addressing drift)."""
        self.current_samples.clear()


class AnomalyDetector:
    """
    Detects anomalies in performance metrics using statistical methods.

    Implements multiple detection strategies including:
    - Z-score based detection
    - IQR (Interquartile Range) method
    - Moving average deviation
    """

    def __init__(
        self, method: str = "zscore", threshold: float = 3.0, window_size: int = 50
    ):
        """
        Initialize anomaly detector.

        Args:
            method: Detection method ('zscore', 'iqr', 'moving_avg')
            threshold: Threshold for anomaly detection (interpretation depends on method)
            window_size: Size of rolling window for calculations
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.samples: deque = deque(maxlen=window_size)

    def add_sample(self, value: float) -> None:
        """Add a sample to the detector."""
        self.samples.append(value)

    def is_anomaly(self, value: float) -> Tuple[bool, float]:
        """
        Check if a value is anomalous.

        Args:
            value: Value to check

        Returns:
            Tuple of (is_anomalous, anomaly_score)
        """
        if len(self.samples) < 10:
            return False, 0.0

        samples_array = np.array(list(self.samples))

        if self.method == "zscore":
            mean = np.mean(samples_array)
            std = np.std(samples_array)
            if std == 0:
                return False, 0.0
            z_score = abs((value - mean) / std)
            return z_score > self.threshold, z_score

        elif self.method == "iqr":
            q1 = np.percentile(samples_array, 25)
            q3 = np.percentile(samples_array, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            is_anomalous = value < lower_bound or value > upper_bound
            deviation = min(abs(value - lower_bound), abs(value - upper_bound))
            return is_anomalous, deviation

        elif self.method == "moving_avg":
            moving_avg = np.mean(samples_array)
            moving_std = np.std(samples_array)
            if moving_std == 0:
                return False, 0.0
            deviation = abs(value - moving_avg) / moving_std
            return deviation > self.threshold, deviation

        return False, 0.0


class LMPerformanceMonitor:
    """
    Comprehensive performance monitoring for language models.

    Tracks multiple performance metrics, detects drift and anomalies,
    and provides alerting capabilities for model health monitoring
    during sleep cycles.

    Attributes:
        model_id (str): Identifier for the monitored model
        metrics (Dict): Currently tracked metric values
        drift_detectors (Dict): Drift detectors per metric
        anomaly_detectors (Dict): Anomaly detectors per metric
        alerts (List[PerformanceAlert]): History of performance alerts

    Example:
        >>> monitor = LMPerformanceMonitor("gpt-neo-125M")
        >>> monitor.track_metric("perplexity", 3.45)
        >>> drift_detected = monitor.check_drift("perplexity")
    """

    def __init__(
        self,
        model_id: str,
        drift_window: int = 100,
        anomaly_method: str = "zscore",
        anomaly_threshold: float = 3.0,
    ):
        """
        Initialize performance monitor.

        Args:
            model_id: Unique identifier for the model
            drift_window: Window size for drift detection
            anomaly_method: Method for anomaly detection
            anomaly_threshold: Threshold for anomaly detection
        """
        self.model_id = model_id
        self.drift_window = drift_window
        self.anomaly_method = anomaly_method
        self.anomaly_threshold = anomaly_threshold

        # Metric storage
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = {}
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}

        # Detectors
        self.drift_detectors: Dict[str, DriftDetector] = {}
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}

        # Alerts and thresholds
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.metric_thresholds: Dict[str, Dict[str, float]] = {}

        # Statistics
        self.start_time = datetime.now()
        self.total_samples = 0
        self.alert_count = 0

    def register_metric(
        self,
        metric_name: str,
        track_drift: bool = True,
        track_anomalies: bool = True,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ) -> None:
        """
        Register a new metric for monitoring.

        Args:
            metric_name: Name of the metric to track
            track_drift: Whether to enable drift detection
            track_anomalies: Whether to enable anomaly detection
            min_threshold: Minimum acceptable value (optional)
            max_threshold: Maximum acceptable value (optional)
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.metric_metadata[metric_name] = {
                "track_drift": track_drift,
                "track_anomalies": track_anomalies,
                "registered_at": datetime.now(),
            }

        if track_drift and metric_name not in self.drift_detectors:
            self.drift_detectors[metric_name] = DriftDetector(
                window_size=self.drift_window
            )

        if track_anomalies and metric_name not in self.anomaly_detectors:
            self.anomaly_detectors[metric_name] = AnomalyDetector(
                method=self.anomaly_method, threshold=self.anomaly_threshold
            )

        if min_threshold is not None or max_threshold is not None:
            self.metric_thresholds[metric_name] = {
                "min": min_threshold if min_threshold is not None else float("-inf"),
                "max": max_threshold if max_threshold is not None else float("inf"),
            }

    def track_metric(self, metric_name: str, value: float) -> None:
        """
        Record a metric value and perform checks.

        Args:
            metric_name: Name of the metric
            value: Metric value to record
        """
        # Auto-register if not already registered
        if metric_name not in self.metrics:
            self.register_metric(metric_name)

        timestamp = datetime.now()
        self.metrics[metric_name].append((timestamp, value))
        self.total_samples += 1

        # Check thresholds
        if metric_name in self.metric_thresholds:
            thresholds = self.metric_thresholds[metric_name]
            if value < thresholds["min"]:
                self._create_alert(
                    alert_type="threshold_violation",
                    severity="warning",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=thresholds["min"],
                    message=f"Metric {metric_name} below minimum threshold",
                )
            elif value > thresholds["max"]:
                self._create_alert(
                    alert_type="threshold_violation",
                    severity="warning",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=thresholds["max"],
                    message=f"Metric {metric_name} above maximum threshold",
                )

        # Add to detectors
        metadata = self.metric_metadata.get(metric_name, {})

        if metadata.get("track_drift") and metric_name in self.drift_detectors:
            self.drift_detectors[metric_name].add_current_sample(value)

        if metadata.get("track_anomalies") and metric_name in self.anomaly_detectors:
            detector = self.anomaly_detectors[metric_name]
            is_anomaly, score = detector.is_anomaly(value)
            detector.add_sample(value)

            if is_anomaly:
                self._create_alert(
                    alert_type="anomaly",
                    severity="high",
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=score,
                    message=f"Anomalous value detected for {metric_name}",
                )

    def establish_baseline(self, metric_name: str) -> None:
        """
        Mark current samples as baseline for drift detection.

        Args:
            metric_name: Name of the metric
        """
        if metric_name not in self.drift_detectors:
            return

        detector = self.drift_detectors[metric_name]

        # Transfer current samples to baseline
        for sample in detector.current_samples:
            detector.add_baseline_sample(sample)
        detector.reset_current()

    def check_drift(self, metric_name: str) -> Tuple[bool, float]:
        """
        Check if drift has occurred for a metric.

        Args:
            metric_name: Name of the metric to check

        Returns:
            Tuple of (drift_detected, p_value)
        """
        if metric_name not in self.drift_detectors:
            return False, 1.0

        drift_detected, p_value = self.drift_detectors[metric_name].detect_drift()

        if drift_detected:
            self._create_alert(
                alert_type="drift",
                severity="high",
                metric_name=metric_name,
                current_value=0.0,
                threshold_value=p_value,
                message=f"Performance drift detected for {metric_name}",
            )

        return drift_detected, p_value

    def get_metric_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistical summary for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary of statistical measures
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = [v for _, v in self.metrics[metric_name]]

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
            "latest": values[-1] if values else 0.0,
        }

    def get_metric_trend(
        self, metric_name: str, window: Optional[int] = None
    ) -> List[float]:
        """
        Get recent trend for a metric.

        Args:
            metric_name: Name of the metric
            window: Number of recent samples (None for all)

        Returns:
            List of recent metric values
        """
        if metric_name not in self.metrics:
            return []

        values = [v for _, v in self.metrics[metric_name]]

        if window is not None:
            values = values[-window:]

        return values

    def register_alert_callback(
        self, callback: Callable[[PerformanceAlert], None]
    ) -> None:
        """
        Register a callback function to be called when alerts are created.

        Args:
            callback: Function to call with PerformanceAlert objects
        """
        self.alert_callbacks.append(callback)

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        message: str,
    ) -> None:
        """
        Create and store a performance alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            metric_name: Name of affected metric
            current_value: Current metric value
            threshold_value: Threshold or reference value
            message: Alert message
        """
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            metadata={"model_id": self.model_id, "total_samples": self.total_samples},
        )

        self.alerts.append(alert)
        self.alert_count += 1

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                # Silently handle callback errors to prevent monitoring disruption
                pass

    def get_recent_alerts(
        self, severity: Optional[str] = None, count: Optional[int] = None
    ) -> List[PerformanceAlert]:
        """
        Get recent performance alerts.

        Args:
            severity: Filter by severity level (optional)
            count: Number of alerts to return (None for all)

        Returns:
            List of PerformanceAlert objects
        """
        alerts = self.alerts

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        if count is not None:
            alerts = alerts[-count:]

        return alerts

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self.alerts.clear()
        self.alert_count = 0

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of monitoring status.

        Returns:
            Dictionary containing monitoring statistics
        """
        return {
            "model_id": self.model_id,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_samples": self.total_samples,
            "metrics_tracked": len(self.metrics),
            "total_alerts": self.alert_count,
            "recent_alerts": len(self.alerts),
            "drift_detectors": len(self.drift_detectors),
            "anomaly_detectors": len(self.anomaly_detectors),
            "metrics": list(self.metrics.keys()),
        }
