"""
Unit tests for C2C (Cache-to-Cache) functionality.
"""

import unittest
import tempfile
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.ai_sleep.c2c import (
        KVSpec,
        KVCache,
        CacheProjector,
        LayerGate,
        C2CFuser
    )


def create_dummy_kv(
    batch: int = 1,
    layers: int = 4,
    heads: int = 8,
    seq: int = 64,
    head_dim: int = 128,
    device: str = "cpu"
) -> KVCache:
    """Create dummy KV cache for testing."""
    keys = []
    values = []
    specs = []
    
    for _ in range(layers):
        K = torch.randn(batch, heads, seq, head_dim, device=device)
        V = torch.randn(batch, heads, seq, head_dim, device=device)
        keys.append(K)
        values.append(V)
        specs.append(KVSpec(
            n_heads=heads,
            head_dim=head_dim,
            seq_len=seq,
            dtype=K.dtype,
            device=K.device
        ))
    
    return KVCache(keys=keys, values=values, spec=specs)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestKVCache(unittest.TestCase):
    """Tests for KVCache data structure."""
    
    def test_kvcache_creation(self):
        """Test KVCache creation and basic properties."""
        cache = create_dummy_kv(layers=4)
        
        self.assertEqual(len(cache), 4)
        self.assertEqual(len(cache.keys), 4)
        self.assertEqual(len(cache.values), 4)
        self.assertEqual(len(cache.spec), 4)
        
    def test_kvcache_to_device(self):
        """Test moving cache to different device."""
        cache = create_dummy_kv(layers=2, device="cpu")
        
        # Should work without error (even if GPU not available)
        cache_cpu = cache.to(torch.device("cpu"))
        self.assertEqual(cache_cpu.keys[0].device.type, "cpu")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestCacheProjector(unittest.TestCase):
    """Tests for CacheProjector module."""
    
    def test_projector_initialization(self):
        """Test projector initialization."""
        projector = CacheProjector(head_dim=128, rank=64)
        
        self.assertEqual(projector.head_dim, 128)
        self.assertEqual(projector.rank, 64)
        
    def test_projector_forward(self):
        """Test projector forward pass."""
        projector = CacheProjector(head_dim=128, rank=64)
        
        K = torch.randn(1, 8, 64, 128)
        V = torch.randn(1, 8, 64, 128)
        
        K_proj, V_proj = projector(K, V)
        
        # Output shapes should match input shapes
        self.assertEqual(K_proj.shape, K.shape)
        self.assertEqual(V_proj.shape, V.shape)
        
    def test_projector_low_rank(self):
        """Test that projector uses low-rank bottleneck."""
        projector = CacheProjector(head_dim=128, rank=32)
        
        # Check that bottleneck layers have correct dimensions
        self.assertEqual(projector.k_down.out_features, 32)
        self.assertEqual(projector.k_up.in_features, 32)
        self.assertEqual(projector.v_down.out_features, 32)
        self.assertEqual(projector.v_up.in_features, 32)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestLayerGate(unittest.TestCase):
    """Tests for LayerGate module."""
    
    def test_gate_initialization(self):
        """Test gate initialization."""
        gate = LayerGate(n_layers=10)
        
        self.assertEqual(gate.n_layers, 10)
        self.assertEqual(len(gate.alpha), 10)
        
    def test_gate_output_range(self):
        """Test that gate outputs are in [0, 1] range."""
        gate = LayerGate(n_layers=5)
        g = gate()
        
        self.assertEqual(g.shape, (5,))
        self.assertTrue(torch.all(g >= 0))
        self.assertTrue(torch.all(g <= 1))
        
    def test_gate_conservative_initialization(self):
        """Test that gates initialize near 0.5 (conservative)."""
        gate = LayerGate(n_layers=10)
        g = gate()
        
        # Initialized at 0, sigmoid(0) = 0.5
        self.assertTrue(torch.allclose(g, torch.tensor(0.5), atol=0.01))
        
    def test_gate_statistics(self):
        """Test gate statistics computation."""
        gate = LayerGate(n_layers=5)
        stats = gate.get_gate_statistics()
        
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestC2CFuser(unittest.TestCase):
    """Tests for C2CFuser module."""
    
    def test_fuser_initialization(self):
        """Test fuser initialization."""
        fuser = C2CFuser(n_layers=4, head_dim=128, rank=64)
        
        self.assertEqual(fuser.n_layers, 4)
        self.assertEqual(fuser.head_dim, 128)
        self.assertEqual(fuser.rank, 64)
        
    def test_fuse_shapes_match(self):
        """Test that fusion preserves shapes."""
        src = create_dummy_kv(batch=1, layers=4, heads=8, seq=64, head_dim=128)
        tgt = create_dummy_kv(batch=1, layers=4, heads=8, seq=64, head_dim=128)
        
        fuser = C2CFuser(n_layers=4, head_dim=128, rank=64)
        fused = fuser.fuse(src, tgt)
        
        self.assertEqual(len(fused.keys), 4)
        self.assertEqual(len(fused.values), 4)
        
        for i in range(4):
            self.assertEqual(fused.keys[i].shape, tgt.keys[i].shape)
            self.assertEqual(fused.values[i].shape, tgt.values[i].shape)
            
    def test_fuse_layer_mismatch_raises(self):
        """Test that mismatched layer counts raise error."""
        src = create_dummy_kv(layers=4)
        tgt = create_dummy_kv(layers=6)
        
        fuser = C2CFuser(n_layers=4, head_dim=128)
        
        with self.assertRaises(ValueError):
            fuser.fuse(src, tgt)
            
    def test_fuse_shape_mismatch_raises(self):
        """Test that mismatched shapes raise error."""
        src = create_dummy_kv(layers=2, seq=32)
        tgt = create_dummy_kv(layers=2, seq=64)
        
        fuser = C2CFuser(n_layers=2, head_dim=128)
        
        with self.assertRaises(ValueError):
            fuser.fuse(src, tgt)
            
    def test_fuse_with_layer_mask(self):
        """Test fusion with layer masking."""
        src = create_dummy_kv(layers=4)
        tgt = create_dummy_kv(layers=4)
        
        fuser = C2CFuser(n_layers=4, head_dim=128)
        
        # Mask out layers 2 and 3 (only fuse 0 and 1)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        fused = fuser.fuse(src, tgt, layer_mask=mask)
        
        # Should still return valid cache
        self.assertEqual(len(fused.keys), 4)
        
    def test_fusion_statistics(self):
        """Test fusion statistics retrieval."""
        fuser = C2CFuser(n_layers=4, head_dim=128, rank=64)
        stats = fuser.get_fusion_statistics()
        
        self.assertIn("n_layers", stats)
        self.assertIn("head_dim", stats)
        self.assertIn("rank", stats)
        self.assertIn("gate_stats", stats)
        self.assertIn("projector_params", stats)
        
        self.assertEqual(stats["n_layers"], 4)
        self.assertEqual(stats["head_dim"], 128)
        self.assertEqual(stats["rank"], 64)
        
    def test_save_and_load_weights(self):
        """Test saving and loading fuser weights."""
        fuser = C2CFuser(n_layers=4, head_dim=128, rank=64)
        
        # Modify some weights
        fuser.gate.alpha.data.fill_(1.0)
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name
            
        try:
            # Save weights
            fuser.save_weights(temp_path)
            
            # Load weights
            loaded_fuser = C2CFuser.load_weights(temp_path)
            
            # Verify configuration matches
            self.assertEqual(loaded_fuser.n_layers, fuser.n_layers)
            self.assertEqual(loaded_fuser.head_dim, fuser.head_dim)
            self.assertEqual(loaded_fuser.rank, fuser.rank)
            
            # Verify weights match
            self.assertTrue(
                torch.allclose(
                    loaded_fuser.gate.alpha,
                    fuser.gate.alpha
                )
            )
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
            
    def test_fuse_preserves_target_device(self):
        """Test that fusion preserves target device."""
        src = create_dummy_kv(layers=2, device="cpu")
        tgt = create_dummy_kv(layers=2, device="cpu")
        
        fuser = C2CFuser(n_layers=2, head_dim=128)
        fused = fuser.fuse(src, tgt)
        
        for k, v in zip(fused.keys, fused.values):
            self.assertEqual(k.device.type, "cpu")
            self.assertEqual(v.device.type, "cpu")


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not available")
class TestC2CIntegration(unittest.TestCase):
    """Integration tests for C2C workflow."""
    
    def test_end_to_end_fusion_workflow(self):
        """Test complete fusion workflow."""
        # Create source and target caches
        src_cache = create_dummy_kv(batch=1, layers=4, heads=8, seq=32, head_dim=64)
        tgt_cache = create_dummy_kv(batch=1, layers=4, heads=8, seq=32, head_dim=64)
        
        # Create fuser
        fuser = C2CFuser(n_layers=4, head_dim=64, rank=32)
        
        # Perform fusion
        fused_cache = fuser.fuse(src_cache, tgt_cache)
        
        # Verify result
        self.assertEqual(len(fused_cache), 4)
        self.assertIsInstance(fused_cache, KVCache)
        
        # Verify shapes preserved
        for i in range(4):
            self.assertEqual(
                fused_cache.keys[i].shape,
                tgt_cache.keys[i].shape
            )
            
    def test_multiple_fusion_rounds(self):
        """Test multiple rounds of fusion."""
        cache1 = create_dummy_kv(layers=2)
        cache2 = create_dummy_kv(layers=2)
        cache3 = create_dummy_kv(layers=2)
        
        fuser = C2CFuser(n_layers=2, head_dim=128)
        
        # Round 1: fuse cache1 into cache2
        result1 = fuser.fuse(cache1, cache2)
        
        # Round 2: fuse cache3 into result1
        result2 = fuser.fuse(cache3, result1)
        
        # Should complete without error
        self.assertEqual(len(result2), 2)


if __name__ == "__main__":
    unittest.main()
