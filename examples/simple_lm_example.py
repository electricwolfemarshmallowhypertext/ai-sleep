"""
Simple Language Model Integration Example

This example demonstrates basic integration of AI Sleep Constructs with
a language model. It shows how to:
1. Create a sleep controller
2. Configure light and deep sleep modes
3. Initiate sleep cycles
4. Monitor performance
5. Track metrics
"""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_sleep import AISleepController, LMPerformanceMonitor
from ai_sleep.model_state import SleepMode
from ai_sleep.sleep_controller import SleepTrigger, OptimizationStrategy


def simulate_model_inference(controller, num_iterations=10):
    """
    Simulate model inference and track performance metrics.

    Args:
        controller: AISleepController instance
        num_iterations: Number of inference iterations to simulate
    """
    print("\n" + "=" * 60)
    print("Simulating Model Inference")
    print("=" * 60)

    monitor = controller.performance_monitor

    for i in range(num_iterations):
        # Simulate inference metrics (in reality, these would come from actual model)
        perplexity = 3.5 + (i * 0.05)  # Gradually increasing perplexity
        loss = 0.5 - (i * 0.02)  # Gradually decreasing loss
        inference_time = 0.1 + (i * 0.01)  # Gradually increasing inference time

        # Track metrics
        monitor.track_metric("perplexity", perplexity)
        monitor.track_metric("loss", loss)
        monitor.track_metric("inference_time", inference_time)

        print(
            f"Iteration {i+1}: perplexity={perplexity:.2f}, "
            f"loss={loss:.3f}, time={inference_time:.3f}s"
        )

        time.sleep(0.1)  # Simulate processing time


def main():
    """Main example execution."""

    print("=" * 60)
    print("AI Sleep Constructs - Simple Language Model Example")
    print("=" * 60)

    # Step 1: Create a sleep controller
    print("\n1. Creating AISleepController...")
    controller = AISleepController(
        model_id="gpt-neo-125M", enable_monitoring=True, auto_adapt=True
    )
    print(f"   ✓ Controller created for model: {controller.model_id}")
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")

    # Step 2: Configure light sleep mode
    print("\n2. Configuring Light Sleep Mode...")
    controller.configure_light_sleep(
        duration=5,  # Short duration for demo (5 seconds)
        strategies=[
            OptimizationStrategy.GRADIENT_CLIPPING,
            OptimizationStrategy.KV_CACHE_MANAGEMENT,
            OptimizationStrategy.ADAPTIVE_LEARNING_RATE,
        ],
        gradient_clip_norm=1.0,
        kv_cache_retention=0.8,
    )
    print("   ✓ Light sleep configured:")
    print(f"     - Duration: {controller.light_sleep_config['duration']}s")
    print(f"     - Strategies: {len(controller.light_sleep_config['strategies'])}")

    # Step 3: Configure deep sleep mode
    print("\n3. Configuring Deep Sleep Mode...")
    controller.configure_deep_sleep(
        duration=10,  # Short duration for demo (10 seconds)
        strategies=[
            OptimizationStrategy.ATTENTION_HEAD_PRUNING,
            OptimizationStrategy.SEMANTIC_CONSOLIDATION,
            OptimizationStrategy.LAYER_NORM_RECALIBRATION,
            OptimizationStrategy.SECURITY_PATCHING,
        ],
        pruning_threshold=0.1,
        consolidation_batch_size=1000,
    )
    print("   ✓ Deep sleep configured:")
    print(f"     - Duration: {controller.deep_sleep_config['duration']}s")
    print(f"     - Strategies: {len(controller.deep_sleep_config['strategies'])}")

    # Step 4: Set up monitoring thresholds
    print("\n4. Setting up Performance Monitoring...")
    monitor = controller.performance_monitor
    monitor.register_metric(
        "perplexity",
        track_drift=True,
        track_anomalies=True,
        min_threshold=2.0,
        max_threshold=6.0,
    )
    monitor.register_metric(
        "loss",
        track_drift=True,
        track_anomalies=True,
        min_threshold=0.0,
        max_threshold=2.0,
    )
    print("   ✓ Metrics registered: perplexity, loss")
    print("   ✓ Drift detection enabled")
    print("   ✓ Anomaly detection enabled")

    # Step 5: Register callbacks
    print("\n5. Registering Sleep Cycle Callbacks...")

    def on_sleep_start(mode, trigger):
        print(f"\n   → Sleep cycle started: {mode.value} (trigger: {trigger.value})")

    def on_sleep_end():
        print("   → Sleep cycle completed")

    def on_optimization(strategy):
        print(f"   → Optimization applied: {strategy.value}")

    controller.register_callback("on_sleep_start", on_sleep_start)
    controller.register_callback("on_sleep_end", on_sleep_end)
    controller.register_callback("on_optimization_complete", on_optimization)
    print("   ✓ Callbacks registered")

    # Step 6: Simulate normal operation
    simulate_model_inference(controller, num_iterations=5)

    # Step 7: Initiate light sleep
    print("\n6. Initiating Light Sleep Cycle...")
    success = controller.initiate_sleep(
        trigger=SleepTrigger.MANUAL, mode=SleepMode.LIGHT_SLEEP
    )
    print(f"   ✓ Light sleep initiated: {success}")
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")

    # Step 8: Wake up from light sleep
    print("\n7. Waking from Light Sleep...")
    time.sleep(1)  # Simulate sleep duration
    success = controller.wake_up()
    print(f"   ✓ Wake up successful: {success}")
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")

    # Step 9: Simulate more inference
    simulate_model_inference(controller, num_iterations=3)

    # Step 10: Establish baseline for drift detection
    print("\n8. Establishing Performance Baseline...")
    monitor.establish_baseline("perplexity")
    monitor.establish_baseline("loss")
    print("   ✓ Baseline established for drift detection")

    # Step 11: Initiate deep sleep
    print("\n9. Initiating Deep Sleep Cycle...")
    controller.initiate_sleep(
        trigger=SleepTrigger.SCHEDULED, mode=SleepMode.LIGHT_SLEEP
    )
    controller.initiate_sleep(trigger=SleepTrigger.SCHEDULED, mode=SleepMode.DEEP_SLEEP)
    print(f"   ✓ Deep sleep initiated")
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")

    time.sleep(1)  # Simulate sleep duration

    # Step 12: Wake up from deep sleep
    print("\n10. Waking from Deep Sleep...")
    controller.wake_up()
    print(f"   ✓ Wake up successful")
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")

    # Step 13: Get comprehensive statistics
    print("\n11. Sleep Cycle Statistics:")
    print("=" * 60)
    stats = controller.get_sleep_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Step 14: Get performance monitoring summary
    print("\n12. Performance Monitoring Summary:")
    print("=" * 60)
    monitoring_summary = monitor.get_monitoring_summary()
    for key, value in monitoring_summary.items():
        print(f"   {key}: {value}")

    # Step 15: Get metric statistics
    print("\n13. Metric Statistics:")
    print("=" * 60)
    for metric in ["perplexity", "loss"]:
        stats = monitor.get_metric_statistics(metric)
        if stats:
            print(f"\n   {metric.upper()}:")
            print(f"     Mean: {stats['mean']:.3f}")
            print(f"     Median: {stats['median']:.3f}")
            print(f"     Std Dev: {stats['std']:.3f}")
            print(f"     Min: {stats['min']:.3f}")
            print(f"     Max: {stats['max']:.3f}")
            print(f"     Count: {stats['count']}")

    # Step 16: Check for alerts
    print("\n14. Checking for Alerts...")
    alerts = monitor.get_recent_alerts()
    if alerts:
        print(f"   Found {len(alerts)} alert(s):")
        for alert in alerts:
            print(f"     - [{alert.severity}] {alert.message}")
    else:
        print("   No alerts generated")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
