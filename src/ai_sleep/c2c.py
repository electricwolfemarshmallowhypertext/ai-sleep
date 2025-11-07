"""
C2C (Cache-to-Cache): KV cache fusion between language models.

This module implements cross-model KV cache transfer via learned projectors
and gating mechanisms, enabling warm-start generation without text relay.

Based on: "Cache-to-Cache: Empirical Study of KV Cache Transfer for LLMs"
arXiv: https://arxiv.org/abs/2405.05277
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn


@dataclass
class KVSpec:
    """
    Shape specification for a single layer's KV cache.
    
    Attributes:
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        seq_len: Sequence length
        dtype: Data type of tensors
        device: Device (CPU/GPU)
    """
    n_heads: int
    head_dim: int
    seq_len: int
    dtype: torch.dtype
    device: torch.device


@dataclass
class KVCache:
    """
    Per-layer KV cache container.
    
    Keys and values are shaped [batch, n_heads, seq_len, head_dim]
    
    Attributes:
        keys: List of key tensors, one per layer
        values: List of value tensors, one per layer
        spec: List of KVSpec for each layer
    """
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    spec: List[KVSpec]
    
    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.keys)
    
    def to(self, device: torch.device) -> "KVCache":
        """Move cache to device."""
        return KVCache(
            keys=[k.to(device) for k in self.keys],
            values=[v.to(device) for v in self.values],
            spec=[KVSpec(s.n_heads, s.head_dim, s.seq_len, s.dtype, device) 
                  for s in self.spec]
        )


class CacheProjector(nn.Module):
    """
    Projects source KV cache into target KV cache space.
    
    Implements per-layer low-rank adapters with residuals for efficient
    cross-model knowledge transfer.
    
    Args:
        head_dim: Dimension of attention heads
        rank: Bottleneck dimension for low-rank projection (default: 64)
        
    Example:
        >>> projector = CacheProjector(head_dim=128, rank=64)
        >>> K_src = torch.randn(1, 32, 64, 128)
        >>> V_src = torch.randn(1, 32, 64, 128)
        >>> K_proj, V_proj = projector(K_src, V_src)
    """
    
    def __init__(self, head_dim: int, rank: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.rank = rank
        
        # Low-rank projection for keys: head_dim -> rank -> head_dim
        self.k_down = nn.Linear(head_dim, rank, bias=False)
        self.k_up = nn.Linear(rank, head_dim, bias=False)
        
        # Low-rank projection for values: head_dim -> rank -> head_dim
        self.v_down = nn.Linear(head_dim, rank, bias=False)
        self.v_up = nn.Linear(rank, head_dim, bias=False)
        
        # Initialize with small weights for conservative fusion
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection weights conservatively."""
        for module in [self.k_down, self.k_up, self.v_down, self.v_up]:
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self, 
        K: torch.Tensor, 
        V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project source KV to target space.
        
        Args:
            K: Source keys [batch, n_heads, seq_len, head_dim]
            V: Source values [batch, n_heads, seq_len, head_dim]
            
        Returns:
            Tuple of (projected_keys, projected_values)
        """
        # Project keys through bottleneck
        K_proj = self.k_up(self.k_down(K))
        
        # Project values through bottleneck
        V_proj = self.v_up(self.v_down(V))
        
        return K_proj, V_proj


class LayerGate(nn.Module):
    """
    Learnable per-layer gate for fusion strength control.
    
    Each layer has a learnable parameter α that determines fusion strength
    via g = sigmoid(α), where g ∈ [0, 1].
    
    Args:
        n_layers: Number of layers to gate
        
    Example:
        >>> gate = LayerGate(n_layers=32)
        >>> g = gate()  # Returns [32] tensor of gate values
    """
    
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        
        # Initialize near 0 → sigmoid ≈ 0.5 → conservative fusion
        self.alpha = nn.Parameter(torch.zeros(n_layers))
        
    def forward(self) -> torch.Tensor:
        """
        Compute gate values for each layer.
        
        Returns:
            Gate values [n_layers], each in [0, 1]
        """
        return torch.sigmoid(self.alpha)
    
    def get_gate_statistics(self) -> Dict[str, float]:
        """
        Get statistics about current gate values.
        
        Returns:
            Dictionary with min, max, mean, std of gate values
        """
        gates = self.forward().detach()
        return {
            "min": float(gates.min()),
            "max": float(gates.max()),
            "mean": float(gates.mean()),
            "std": float(gates.std())
        }


class C2CFuser(nn.Module):
    """
    Cache-to-Cache fusion module.
    
    Fuses source model KV cache into target model KV cache using:
        K_target = (1-g) * K_target + g * Projector(K_source)
        V_target = (1-g) * V_target + g * Projector(V_source)
    
    Where g is a learnable per-layer gate value.
    
    Args:
        n_layers: Number of layers to fuse
        head_dim: Dimension of attention heads
        rank: Bottleneck dimension for projector (default: 64)
        
    Example:
        >>> fuser = C2CFuser(n_layers=32, head_dim=128, rank=64)
        >>> src_cache = KVCache(keys=[...], values=[...], spec=[...])
        >>> tgt_cache = KVCache(keys=[...], values=[...], spec=[...])
        >>> fused = fuser.fuse(src_cache, tgt_cache)
    """
    
    def __init__(self, n_layers: int, head_dim: int, rank: int = 64):
        super().__init__()
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.rank = rank
        
        # Single projector shared across layers
        self.projector = CacheProjector(head_dim, rank)
        
        # Per-layer gates
        self.gate = LayerGate(n_layers)
        
    @torch.no_grad()
    def fuse(
        self, 
        src: KVCache, 
        tgt: KVCache,
        layer_mask: Optional[torch.Tensor] = None
    ) -> KVCache:
        """
        Fuse source cache into target cache.
        
        Args:
            src: Source KV cache
            tgt: Target KV cache
            layer_mask: Optional binary mask [n_layers] to disable fusion
                       for specific layers (1=fuse, 0=skip)
                       
        Returns:
            Fused KV cache with target's spec
            
        Raises:
            ValueError: If src and tgt have different number of layers
        """
        if len(src) != len(tgt):
            raise ValueError(
                f"Source and target must have same number of layers: "
                f"src={len(src)}, tgt={len(tgt)}"
            )
        
        if len(src) != self.n_layers:
            raise ValueError(
                f"Cache has {len(src)} layers but fuser expects {self.n_layers}"
            )
        
        # Get gate values [n_layers]
        g = self.gate()
        
        # Apply layer mask if provided
        if layer_mask is not None:
            g = g * layer_mask
        
        K_out, V_out = [], []
        
        for i, (K_s, V_s, K_t, V_t) in enumerate(
            zip(src.keys, src.values, tgt.keys, tgt.values)
        ):
            # Validate shapes match
            if K_s.shape != K_t.shape or V_s.shape != V_t.shape:
                raise ValueError(
                    f"Layer {i} shape mismatch: "
                    f"src K={K_s.shape}, tgt K={K_t.shape}, "
                    f"src V={V_s.shape}, tgt V={V_t.shape}"
                )
            
            # Project source KV
            K_s_proj, V_s_proj = self.projector(K_s, V_s)
            
            # Apply gating: reshape for broadcasting
            g_i = g[i].view(1, 1, 1, 1)
            
            # Fuse with target
            K_fused = (1 - g_i) * K_t + g_i * K_s_proj
            V_fused = (1 - g_i) * V_t + g_i * V_s_proj
            
            K_out.append(K_fused)
            V_out.append(V_fused)
        
        return KVCache(keys=K_out, values=V_out, spec=tgt.spec)
    
    def get_fusion_statistics(self) -> Dict[str, any]:
        """
        Get statistics about fusion configuration.
        
        Returns:
            Dictionary with gate statistics and projector info
        """
        return {
            "n_layers": self.n_layers,
            "head_dim": self.head_dim,
            "rank": self.rank,
            "gate_stats": self.gate.get_gate_statistics(),
            "projector_params": sum(p.numel() for p in self.projector.parameters())
        }
    
    def save_weights(self, filepath: str) -> None:
        """
        Save fuser weights to file.
        
        Args:
            filepath: Path to save weights (e.g., 'c2c_weights.pt')
        """
        torch.save({
            "projector": self.projector.state_dict(),
            "gate": self.gate.state_dict(),
            "config": {
                "n_layers": self.n_layers,
                "head_dim": self.head_dim,
                "rank": self.rank
            }
        }, filepath)
    
    @classmethod
    def load_weights(cls, filepath: str) -> "C2CFuser":
        """
        Load fuser weights from file.
        
        Args:
            filepath: Path to saved weights
            
        Returns:
            C2CFuser instance with loaded weights
        """
        checkpoint = torch.load(filepath)
        config = checkpoint["config"]
        
        fuser = cls(
            n_layers=config["n_layers"],
            head_dim=config["head_dim"],
            rank=config["rank"]
        )
        
        fuser.projector.load_state_dict(checkpoint["projector"])
        fuser.gate.load_state_dict(checkpoint["gate"])
        
        return fuser
