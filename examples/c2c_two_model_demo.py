"""
C2C Two-Model Demo: Cache-to-Cache fusion between language models.

This example demonstrates KV cache transfer between two models using C2C,
enabling warm-start generation without text relay.

Based on: "Cache-to-Cache: Empirical Study of KV Cache Transfer for LLMs"
arXiv: https://arxiv.org/abs/2405.05277
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ai_sleep.c2c import C2CFuser, KVCache
from ai_sleep.hf_adapters import (
    HFCacheExtractor,
    convert_kvcache_to_pkv,
    get_model_config
)


def create_mock_kv_cache(n_layers: int = 12, n_heads: int = 12, seq_len: int = 32, head_dim: int = 64):
    """Create a mock KV cache for demonstration without loading actual models."""
    from ai_sleep.c2c import KVSpec
    
    keys = []
    values = []
    specs = []
    
    for _ in range(n_layers):
        K = torch.randn(1, n_heads, seq_len, head_dim)
        V = torch.randn(1, n_heads, seq_len, head_dim)
        keys.append(K)
        values.append(V)
        specs.append(KVSpec(
            n_heads=n_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            dtype=K.dtype,
            device=K.device
        ))
    
    return KVCache(keys=keys, values=values, spec=specs)


def demo_c2c_fusion_mock():
    """
    Demonstrate C2C fusion with mock caches (no actual model loading).
    
    This is useful for testing the C2C mechanics without downloading models.
    """
    print("="*60)
    print("C2C Two-Model Demo - Mock Mode")
    print("="*60)
    
    # Configuration for mock models
    n_layers = 12
    n_heads = 12
    seq_len = 32  # Use same seq_len for both
    head_dim = 64
    
    print(f"\n1. Creating Mock Caches...")
    print(f"   Source: {n_layers} layers, {n_heads} heads, seq_len={seq_len}")
    print(f"   Target: {n_layers} layers, {n_heads} heads, seq_len={seq_len}")
    
    # Simulate Model A processing a prompt
    print("\n2. Model A: Processing priming prompt...")
    src_cache = create_mock_kv_cache(n_layers, n_heads, seq_len, head_dim)
    print(f"   ✓ Source cache created: {len(src_cache)} layers")
    
    # Simulate Model B with initial cache
    print("\n3. Model B: Starting with initial cache...")
    tgt_cache = create_mock_kv_cache(n_layers, n_heads, seq_len, head_dim)
    print(f"   ✓ Target cache created: {len(tgt_cache)} layers")
    
    # Create C2C fuser
    print("\n4. Initializing C2C Fuser...")
    fuser = C2CFuser(n_layers=n_layers, head_dim=head_dim, rank=32)
    print(f"   ✓ Fuser created with rank=32")
    
    # Get initial fusion statistics
    stats = fuser.get_fusion_statistics()
    print(f"   ✓ Projector parameters: {stats['projector_params']:,}")
    print(f"   ✓ Gate statistics: mean={stats['gate_stats']['mean']:.3f}, "
          f"std={stats['gate_stats']['std']:.3f}")
    
    # Perform fusion
    print("\n5. Fusing Source → Target...")
    fused_cache = fuser.fuse(src_cache, tgt_cache)
    print(f"   ✓ Fusion complete: {len(fused_cache)} layers")
    
    # Verify fusion
    print("\n6. Verification:")
    for i in range(min(3, n_layers)):
        print(f"   Layer {i}:")
        print(f"     - Keys shape: {fused_cache.keys[i].shape}")
        print(f"     - Values shape: {fused_cache.values[i].shape}")
    
    # Demonstrate selective fusion with layer mask
    print("\n7. Demonstrating Selective Fusion...")
    mask = torch.zeros(n_layers)
    mask[:n_layers//2] = 1.0  # Only fuse first half of layers
    print(f"   Mask: fusing first {n_layers//2} layers only")
    
    selective_fused = fuser.fuse(src_cache, tgt_cache, layer_mask=mask)
    print(f"   ✓ Selective fusion complete")
    
    # Save and load weights
    print("\n8. Testing Weight Persistence...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name
    
    try:
        fuser.save_weights(temp_path)
        print(f"   ✓ Weights saved to {temp_path}")
        
        loaded_fuser = C2CFuser.load_weights(temp_path)
        print(f"   ✓ Weights loaded successfully")
        
        # Verify loaded fuser works
        test_fused = loaded_fuser.fuse(src_cache, tgt_cache)
        print(f"   ✓ Loaded fuser produces valid output")
    finally:
        Path(temp_path).unlink(missing_ok=True)
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. C2C enables KV cache transfer between models")
    print("2. Low-rank projectors map source → target space")
    print("3. Per-layer gates control fusion strength")
    print("4. Selective fusion allows fine-grained control")
    print("5. Weights can be saved/loaded for reuse")


def demo_c2c_fusion_real():
    """
    Demonstrate C2C fusion with real HuggingFace models.
    
    This requires downloading models and is more resource intensive.
    """
    if not HF_AVAILABLE:
        print("HuggingFace transformers not available.")
        print("Install with: pip install transformers")
        return
    
    print("="*60)
    print("C2C Two-Model Demo - Real Models")
    print("="*60)
    print("\nWarning: This will download models (~500MB each)")
    print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
    
    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Use small models for demo
    model_name_a = "gpt2"
    model_name_b = "gpt2"  # Same architecture for simplicity
    
    print(f"\n1. Loading Model A: {model_name_a}...")
    model_a = AutoModelForCausalLM.from_pretrained(model_name_a)
    tokenizer_a = AutoTokenizer.from_pretrained(model_name_a)
    model_a.eval()
    
    print(f"\n2. Loading Model B: {model_name_b}...")
    model_b = AutoModelForCausalLM.from_pretrained(model_name_b)
    tokenizer_b = AutoTokenizer.from_pretrained(model_name_b)
    model_b.eval()
    
    # Get model configurations
    config_a = get_model_config(model_a)
    config_b = get_model_config(model_b)
    
    print(f"\nModel A config: {config_a}")
    print(f"Model B config: {config_b}")
    
    # Create cache extractors
    extractor_a = HFCacheExtractor(model_a)
    extractor_b = HFCacheExtractor(model_b)
    
    # Priming prompt for Model A
    priming_prompt = "The future of artificial intelligence is"
    print(f"\n3. Model A: Processing priming prompt...")
    print(f"   Prompt: '{priming_prompt}'")
    
    input_ids_a = tokenizer_a(priming_prompt, return_tensors="pt").input_ids
    cache_a = extractor_a.extract_cache(input_ids_a)
    print(f"   ✓ Cache extracted: {len(cache_a)} layers")
    
    # Model B starts with empty prompt (just BOS token)
    print(f"\n4. Model B: Starting generation...")
    input_ids_b = tokenizer_b(tokenizer_b.bos_token, return_tensors="pt").input_ids
    cache_b = extractor_b.extract_cache(input_ids_b)
    print(f"   ✓ Initial cache: {len(cache_b)} layers")
    
    # Create fuser
    print(f"\n5. Creating C2C Fuser...")
    fuser = C2CFuser(
        n_layers=config_b["n_layers"],
        head_dim=config_b["head_dim"],
        rank=64
    )
    print(f"   ✓ Fuser initialized")
    
    # Fuse caches
    print(f"\n6. Fusing Model A cache → Model B...")
    
    # Note: In practice, seq_len must match for fusion
    # This is a simplified demo
    if cache_a.keys[0].shape[2] != cache_b.keys[0].shape[2]:
        print(f"   Note: Sequence lengths differ (A: {cache_a.keys[0].shape[2]}, "
              f"B: {cache_b.keys[0].shape[2]})")
        print(f"   In production, use padding/truncation to match")
        print(f"   Skipping actual fusion in this demo")
    else:
        fused_cache = fuser.fuse(cache_a, cache_b)
        print(f"   ✓ Fusion complete")
        
        # Continue generation with fused cache
        print(f"\n7. Model B: Continuing generation with fused cache...")
        pkv_fused = convert_kvcache_to_pkv(fused_cache)
        
        next_token_id = tokenizer_b(" amazing", return_tensors="pt").input_ids[0, -1:]
        outputs = model_b(
            input_ids=next_token_id.unsqueeze(0),
            past_key_values=pkv_fused,
            use_cache=True
        )
        print(f"   ✓ Generation successful")
    
    print("\n" + "="*60)
    print("Real Model Demo Complete!")
    print("="*60)


def main():
    """Main demo execution."""
    print("\nAI Sleep Constructs - C2C Two-Model Demo")
    print("=" * 60)
    print("\nThis demo shows Cache-to-Cache (C2C) fusion between models.")
    print("\nOptions:")
    print("1. Mock demo (fast, no downloads)")
    print("2. Real models demo (slow, downloads required)")
    
    choice = input("\nSelect option (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        demo_c2c_fusion_mock()
    elif choice == "2":
        demo_c2c_fusion_real()
    else:
        print(f"Invalid choice: {choice}")
        return
    
    print("\n" + "="*60)
    print("For more information:")
    print("- Paper: https://arxiv.org/abs/2405.05277")
    print("- Code: https://github.com/LINs-lab/cache-to-cache")
    print("="*60)


if __name__ == "__main__":
    main()
