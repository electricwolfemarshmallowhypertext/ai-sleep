"""
HuggingFace Adapters: Extract and set KV caches across model architectures.

Provides utilities to work with past_key_values from HuggingFace Transformers
and convert them to/from the C2C KVCache format.
"""

from typing import List, Tuple, Optional, Any, Dict
import torch

try:
    from transformers import PreTrainedModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedModel = Any

from .c2c import KVCache, KVSpec


def get_pkv_from_outputs(outputs) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract past_key_values from HuggingFace model outputs.
    
    Args:
        outputs: Model outputs with past_key_values attribute
        
    Returns:
        List of (key, value) tuples, one per layer
        
    Example:
        >>> outputs = model(input_ids, use_cache=True)
        >>> pkv = get_pkv_from_outputs(outputs)
    """
    if not hasattr(outputs, 'past_key_values'):
        raise ValueError("Model outputs do not contain past_key_values")
    
    if outputs.past_key_values is None:
        raise ValueError("past_key_values is None. Did you set use_cache=True?")
    
    return [(k, v) for (k, v) in outputs.past_key_values]


def set_pkv_in_inputs(
    inputs: Dict[str, Any], 
    kv_list: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, Any]:
    """
    Set past_key_values in model inputs dictionary.
    
    Args:
        inputs: Model inputs dictionary
        kv_list: List of (key, value) tuples
        
    Returns:
        Updated inputs dictionary with past_key_values
        
    Example:
        >>> inputs = {"input_ids": ids, "attention_mask": mask}
        >>> inputs = set_pkv_in_inputs(inputs, pkv)
        >>> outputs = model(**inputs)
    """
    inputs = dict(inputs)
    inputs["past_key_values"] = tuple(kv_list)
    return inputs


def infer_kv_spec_from_tensor(
    keys: List[torch.Tensor],
    values: List[torch.Tensor]
) -> List[KVSpec]:
    """
    Infer KVSpec from key/value tensors.
    
    Args:
        keys: List of key tensors
        values: List of value tensors
        
    Returns:
        List of KVSpec, one per layer
        
    Raises:
        ValueError: If keys and values have different lengths
    """
    if len(keys) != len(values):
        raise ValueError(
            f"Keys and values must have same length: "
            f"len(keys)={len(keys)}, len(values)={len(values)}"
        )
    
    specs = []
    for k, v in zip(keys, values):
        # Assume shape [batch, n_heads, seq_len, head_dim]
        if len(k.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor [batch, n_heads, seq_len, head_dim], "
                f"got shape {k.shape}"
            )
        
        batch, n_heads, seq_len, head_dim = k.shape
        
        specs.append(KVSpec(
            n_heads=n_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=k.dtype,
            device=k.device
        ))
    
    return specs


def convert_pkv_to_kvcache(
    pkv: List[Tuple[torch.Tensor, torch.Tensor]]
) -> KVCache:
    """
    Convert HuggingFace past_key_values to KVCache.
    
    Args:
        pkv: past_key_values from HuggingFace model
        
    Returns:
        KVCache object
        
    Example:
        >>> outputs = model(input_ids, use_cache=True)
        >>> pkv = outputs.past_key_values
        >>> cache = convert_pkv_to_kvcache(pkv)
    """
    keys = [k for (k, _) in pkv]
    values = [v for (_, v) in pkv]
    spec = infer_kv_spec_from_tensor(keys, values)
    
    return KVCache(keys=keys, values=values, spec=spec)


def convert_kvcache_to_pkv(
    cache: KVCache
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Convert KVCache to HuggingFace past_key_values format.
    
    Args:
        cache: KVCache object
        
    Returns:
        Tuple of (key, value) tuples suitable for HuggingFace models
        
    Example:
        >>> cache = KVCache(keys=[...], values=[...], spec=[...])
        >>> pkv = convert_kvcache_to_pkv(cache)
        >>> outputs = model(input_ids, past_key_values=pkv)
    """
    return tuple((k, v) for k, v in zip(cache.keys, cache.values))


class HFCacheExtractor:
    """
    Helper class to extract and manage KV caches from HuggingFace models.
    
    Args:
        model: HuggingFace PreTrainedModel instance
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> extractor = HFCacheExtractor(model)
        >>> cache = extractor.extract_cache(input_ids, use_cache=True)
    """
    
    def __init__(self, model: PreTrainedModel):
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library not available. "
                "Install with: pip install transformers"
            )
        self.model = model
        
    def extract_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> KVCache:
        """
        Run model forward pass and extract KV cache.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            **kwargs: Additional model arguments
            
        Returns:
            Extracted KVCache
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **kwargs
            )
        
        pkv = get_pkv_from_outputs(outputs)
        return convert_pkv_to_kvcache(pkv)
    
    def inject_cache(
        self,
        input_ids: torch.Tensor,
        cache: KVCache,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Run model with injected KV cache.
        
        Args:
            input_ids: Input token IDs (typically just next token)
            cache: KVCache to inject
            attention_mask: Optional attention mask
            **kwargs: Additional model arguments
            
        Returns:
            Model outputs
        """
        pkv = convert_kvcache_to_pkv(cache)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=pkv,
                attention_mask=attention_mask,
                use_cache=True,
                **kwargs
            )
        
        return outputs


def get_model_config(model: PreTrainedModel) -> Dict[str, int]:
    """
    Extract relevant config from HuggingFace model.
    
    Args:
        model: HuggingFace model
        
    Returns:
        Dictionary with n_layers, n_heads, head_dim
    """
    config = model.config
    
    # Common config attribute names across architectures
    n_layers = getattr(config, 'num_hidden_layers', None) or \
               getattr(config, 'n_layer', None) or \
               getattr(config, 'num_layers', None)
    
    n_heads = getattr(config, 'num_attention_heads', None) or \
              getattr(config, 'n_head', None) or \
              getattr(config, 'num_heads', None)
    
    hidden_size = getattr(config, 'hidden_size', None) or \
                  getattr(config, 'n_embd', None) or \
                  getattr(config, 'd_model', None)
    
    if n_layers is None or n_heads is None or hidden_size is None:
        raise ValueError(
            f"Could not extract model config. "
            f"Found: n_layers={n_layers}, n_heads={n_heads}, hidden_size={hidden_size}"
        )
    
    head_dim = hidden_size // n_heads
    
    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size
    }
