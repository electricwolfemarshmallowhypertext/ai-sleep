"""
Custom Architecture Template

This template demonstrates how to extend AI Sleep Constructs for non-LLM systems
such as vision models, reinforcement learning agents, or graph neural networks.

The example shows:
1. How to extend LMModelState for custom architectures
2. How to implement architecture-specific optimization strategies
3. How to create custom triggers
4. How to integrate with the sleep controller
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_sleep.model_state import LMModelState, SleepMode
from ai_sleep.sleep_controller import AISleepController, OptimizationStrategy, SleepTrigger


# ============================================================================
# STEP 1: Extend LMModelState for Your Architecture
# ============================================================================

class CustomArchitectureState(LMModelState):
    """
    Extended state manager for custom AI architecture.
    
    This example shows how to track architecture-specific components.
    Replace these with your actual architecture requirements.
    
    Examples of what you might track:
    - Vision Models: Feature maps, spatial attention, pooling layers
    - RL Agents: Replay buffers, policy checkpoints, value estimates
    - GNNs: Graph embeddings, message passing cache, adjacency matrices
    """
    
    def __init__(self, model_id: str):
        """Initialize custom architecture state."""
        super().__init__(model_id)
        
        # Add your custom state components
        self.custom_component_1 = {}  # e.g., feature_maps for vision models
        self.custom_component_2 = {}  # e.g., replay_buffer for RL agents
        self.custom_metrics = []
        
    def store_custom_data(self, key: str, data: Any) -> None:
        """
        Store architecture-specific data.
        
        Args:
            key: Identifier for the data
            data: Data to store
        """
        self.custom_component_1[key] = {
            "data": data,
            "timestamp": self.last_transition_time
        }
        
    def get_custom_data(self, key: str) -> Any:
        """
        Retrieve architecture-specific data.
        
        Args:
            key: Identifier for the data
            
        Returns:
            Stored data or None
        """
        entry = self.custom_component_1.get(key)
        return entry["data"] if entry else None
        
    def clear_custom_cache(self, threshold: float = 0.5) -> int:
        """
        Clear cached data based on some criterion.
        
        Args:
            threshold: Threshold for clearing (0.0 to 1.0)
            
        Returns:
            Number of entries cleared
        """
        # Example: Clear oldest 50% of cache
        num_entries = len(self.custom_component_1)
        entries_to_clear = int(num_entries * threshold)
        
        # Sort by timestamp and clear oldest
        sorted_keys = sorted(
            self.custom_component_1.keys(),
            key=lambda k: self.custom_component_1[k]["timestamp"]
        )
        
        for key in sorted_keys[:entries_to_clear]:
            del self.custom_component_1[key]
            
        return entries_to_clear


# ============================================================================
# STEP 2: Create Custom Optimization Strategies
# ============================================================================

class CustomOptimizationStrategy:
    """
    Custom optimization strategies for your architecture.
    
    Implement the specific optimizations that make sense for your system.
    """
    
    @staticmethod
    def optimize_custom_component_1(state: CustomArchitectureState, **kwargs) -> None:
        """
        Optimize custom component 1.
        
        Example optimizations:
        - Vision: Prune low-activation feature maps
        - RL: Prioritize important experiences in replay buffer
        - GNN: Prune graph edges below importance threshold
        
        Args:
            state: Custom architecture state
            **kwargs: Additional optimization parameters
        """
        print("   → Applying custom optimization 1...")
        
        # Example: Clear old cache entries
        threshold = kwargs.get("cache_threshold", 0.3)
        cleared = state.clear_custom_cache(threshold)
        print(f"     Cleared {cleared} cache entries")
        
    @staticmethod
    def optimize_custom_component_2(state: CustomArchitectureState, **kwargs) -> None:
        """
        Optimize custom component 2.
        
        Args:
            state: Custom architecture state
            **kwargs: Additional optimization parameters
        """
        print("   → Applying custom optimization 2...")
        
        # Example: Consolidate stored metrics
        if state.custom_metrics:
            avg_metric = np.mean(state.custom_metrics)
            state.custom_metrics = [avg_metric]  # Keep only average
            print(f"     Consolidated metrics to average: {avg_metric:.3f}")
            
    @staticmethod
    def architecture_specific_pruning(state: CustomArchitectureState, **kwargs) -> None:
        """
        Implement architecture-specific pruning.
        
        Examples:
        - Vision: Prune unimportant convolutional filters
        - RL: Remove low-reward episodes from replay buffer
        - GNN: Prune low-weight graph connections
        
        Args:
            state: Custom architecture state
            **kwargs: Additional pruning parameters
        """
        print("   → Applying architecture-specific pruning...")
        
        pruning_ratio = kwargs.get("pruning_ratio", 0.1)
        print(f"     Pruning with ratio: {pruning_ratio}")
        
        # Implement your pruning logic here
        # This is just a placeholder
        num_pruned = int(len(state.custom_component_1) * pruning_ratio)
        print(f"     Pruned {num_pruned} components")


# ============================================================================
# STEP 3: Create Custom Controller
# ============================================================================

class CustomArchitectureController(AISleepController):
    """
    Extended sleep controller for custom architecture.
    
    This controller knows how to apply custom optimizations during sleep cycles.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize custom architecture controller."""
        super().__init__(model_id, **kwargs)
        
        # Replace model_state with custom state
        self.model_state = CustomArchitectureState(model_id)
        
        # Configure custom optimization strategies
        self.custom_strategies = CustomOptimizationStrategy()
        
    def _execute_light_sleep(self) -> None:
        """
        Execute light sleep with custom optimizations.
        
        Override to include architecture-specific optimizations.
        """
        # First execute standard light sleep optimizations
        super()._execute_light_sleep()
        
        # Then apply custom optimizations
        print("\n   Applying Custom Light Sleep Optimizations:")
        self.custom_strategies.optimize_custom_component_1(
            self.model_state,
            cache_threshold=0.3
        )
        
    def _execute_deep_sleep(self) -> None:
        """
        Execute deep sleep with custom optimizations.
        
        Override to include architecture-specific optimizations.
        """
        # First execute standard deep sleep optimizations
        super()._execute_deep_sleep()
        
        # Then apply custom optimizations
        print("\n   Applying Custom Deep Sleep Optimizations:")
        self.custom_strategies.optimize_custom_component_2(self.model_state)
        self.custom_strategies.architecture_specific_pruning(
            self.model_state,
            pruning_ratio=0.1
        )


# ============================================================================
# STEP 4: Example Usage
# ============================================================================

def main():
    """Demonstrate custom architecture integration."""
    
    print("="*60)
    print("AI Sleep Constructs - Custom Architecture Template")
    print("="*60)
    
    # Create controller for custom architecture
    print("\n1. Creating Custom Architecture Controller...")
    controller = CustomArchitectureController(
        model_id="custom-vision-model-v1",
        enable_monitoring=True,
        auto_adapt=True
    )
    print(f"   ✓ Controller created for: {controller.model_id}")
    
    # Get reference to custom state
    custom_state = controller.model_state
    
    # Simulate storing architecture-specific data
    print("\n2. Storing Architecture-Specific Data...")
    for i in range(10):
        custom_state.store_custom_data(f"component_{i}", np.random.rand(5, 5))
        custom_state.custom_metrics.append(np.random.random())
    print(f"   ✓ Stored {len(custom_state.custom_component_1)} data entries")
    print(f"   ✓ Stored {len(custom_state.custom_metrics)} metrics")
    
    # Configure sleep modes with custom parameters
    print("\n3. Configuring Custom Sleep Modes...")
    controller.configure_light_sleep(
        duration=5,
        strategies=[
            OptimizationStrategy.GRADIENT_CLIPPING,
            OptimizationStrategy.KV_CACHE_MANAGEMENT,
        ],
        cache_threshold=0.3  # Custom parameter
    )
    
    controller.configure_deep_sleep(
        duration=10,
        strategies=[
            OptimizationStrategy.SEMANTIC_CONSOLIDATION,
            OptimizationStrategy.LAYER_NORM_RECALIBRATION,
        ],
        pruning_ratio=0.1  # Custom parameter
    )
    print("   ✓ Sleep modes configured with custom parameters")
    
    # Execute light sleep cycle
    print("\n4. Executing Light Sleep Cycle...")
    controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")
    print(f"   ✓ Data entries after light sleep: {len(custom_state.custom_component_1)}")
    
    # Wake up
    print("\n5. Waking from Light Sleep...")
    controller.wake_up()
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")
    
    # Add more data
    print("\n6. Storing More Data...")
    for i in range(10, 20):
        custom_state.store_custom_data(f"component_{i}", np.random.rand(5, 5))
        custom_state.custom_metrics.append(np.random.random())
    print(f"   ✓ Total data entries: {len(custom_state.custom_component_1)}")
    print(f"   ✓ Total metrics: {len(custom_state.custom_metrics)}")
    
    # Execute deep sleep cycle
    print("\n7. Executing Deep Sleep Cycle...")
    controller.initiate_sleep(SleepTrigger.MANUAL, SleepMode.LIGHT_SLEEP)
    controller.initiate_sleep(SleepTrigger.SCHEDULED, SleepMode.DEEP_SLEEP)
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")
    print(f"   ✓ Data entries after deep sleep: {len(custom_state.custom_component_1)}")
    print(f"   ✓ Metrics after consolidation: {len(custom_state.custom_metrics)}")
    
    # Wake up
    print("\n8. Waking from Deep Sleep...")
    controller.wake_up()
    print(f"   ✓ Current mode: {controller.model_state.current_mode.value}")
    
    # Get statistics
    print("\n9. Sleep Statistics:")
    print("="*60)
    stats = controller.get_sleep_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("Custom Architecture Integration Complete!")
    print("="*60)
    
    # Guidance for adaptation
    print("\n" + "="*60)
    print("ADAPTATION GUIDE FOR YOUR ARCHITECTURE")
    print("="*60)
    print("""
To adapt this template for your specific architecture:

1. CustomArchitectureState:
   - Replace custom_component_1/2 with your actual components
   - Implement methods to manage your architecture's state
   - Add serialization if needed for state persistence

2. CustomOptimizationStrategy:
   - Implement optimizations specific to your architecture
   - Consider what can be safely pruned or compressed
   - Think about memory vs. performance trade-offs

3. CustomArchitectureController:
   - Override _execute_light_sleep() and _execute_deep_sleep()
   - Add custom triggers if needed
   - Implement architecture-specific metrics

4. Integration:
   - Test with your actual model
   - Monitor performance impact
   - Tune sleep durations and strategies
   - Set appropriate thresholds

5. Common Architectures:
   - Vision Models: Track feature maps, attention, pooling
   - RL Agents: Manage replay buffer, policy checkpoints
   - GNNs: Handle graph structures, message passing
   - Time Series: Manage sequence buffers, attention windows
   - Multimodal: Coordinate cross-modal caches

For more examples, see docs/architectures/ directory.
""")


if __name__ == "__main__":
    main()
