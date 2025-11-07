# AI Sleep Constructs

Production-ready Python framework for engineered sleep cycles in language models.  
Light and deep sleep modes enable offline optimization, performance monitoring, gradient clipping, semantic consolidation, and adaptive learning.

<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg"></a>
<a href="https://doi.org/10.5281/zenodo.17547016"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17547016.svg"></a>

---

## Overview
AI Sleep Constructs provides a comprehensive framework for implementing **chronological intelligence** in stateful language-model systems.  
It enables:
- **Light Sleep Mode:** quick maintenance (gradient clipping, KV-cache management, adaptive learning)  
- **Deep Sleep Mode:** intensive optimization (head pruning, semantic consolidation, layer-norm recalibration, security patching)  
- **Performance Monitoring:** drift & anomaly detection  
- **Contextual Triggers:** automated sleep initiation based on performance or schedule  
- **Hugging Face Integration:** seamless wrapper for Transformers models  

---

## Installation
```bash
pip install -e .
# for HuggingFace integration
pip install -e ".[huggingface]"
# for development
pip install -e ".[dev]"
```

---

## Quick Start

Start with mock mode (no model downloads) for a fast demo, then try the 10-line real-model example.

```python
from ai_sleep import AISleepController, SleepTrigger
from ai_sleep.model_state import SleepMode

controller = AISleepController(model_id="gpt-neo-125M", enable_monitoring=True)
controller.configure_light_sleep(duration=300)
controller.configure_deep_sleep(duration=1800)
controller.initiate_sleep(trigger=SleepTrigger.MANUAL, mode=SleepMode.LIGHT_SLEEP)
controller.wake_up()
print(controller.get_sleep_statistics())
```

---

## Hugging Face Integration
```python
from ai_sleep.huggingface_integration import (
    create_sleep_enabled_model, configure_optimal_sleep_schedule
)

adapter = create_sleep_enabled_model("gpt2", enable_monitoring=True)
configure_optimal_sleep_schedule(adapter, workload_type="continuous")
adapter.enable_sleep_cycles()
adapter.initiate_light_sleep(duration=300)
```

---

## C2C (Cache-to-Cache) Fusion
```python
from ai_sleep import C2CFuser, HFCacheExtractor, convert_kvcache_to_pkv
from transformers import AutoModelForCausalLM

model_a = AutoModelForCausalLM.from_pretrained("gpt2")
model_b = AutoModelForCausalLM.from_pretrained("gpt2-medium")

extractor = HFCacheExtractor(model_a)
cache_a = extractor.extract_cache(priming_input_ids)
cache_b = extractor.extract_cache(initial_input_ids)

fuser = C2CFuser(n_layers=24, head_dim=64, rank=32)
fused = fuser.fuse(cache_a, cache_b)
pkv = convert_kvcache_to_pkv(fused)
outputs = model_b(next_tokens, past_key_values=pkv)
```

### Benefits
- **5-10× faster** than text relay  
- Preserves **85-99%** latent quality  
- Enables **cross-model transfer**  

### Telemetry
Fusion and performance logging:
```
{layer: g, rank, fused_tokens, Δppl, Δlatency_ms, rollback, donor_model, tgt_model, tokenizer_hash}
```

### Safety by Default
See `docs/c2c.md`.  
- Redaction **on** by default  
- Cross-tenant **off** by default  
- Metadata logged for audit  

### When NOT to Fuse
Skip fusion when:
- Tokenizers differ or rotary embeddings misalign  
- Sequence-length mismatch > 25%  
- Donor model flagged for privacy risk  
- KV dtype/device mismatch  

---

## Testing
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0**  
Credit must be given to **Tionne Smith, Antiparty, Inc.**

---

## Citation

```
Smith, T. (2025). AI Sleep Constructs: Implementing Chronological Intelligence in Stateful Systems. Zenodo. DOI: 10.5281/zenodo.17547016
```