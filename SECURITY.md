# Security Policy

## Supported Versions

The following versions of AI Sleep Constructs are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take the security of AI Sleep Constructs seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Public Disclose

**Please do not** open a public issue or pull request. Security vulnerabilities should be reported privately.

### 2. Report Privately

Send an email to the maintainers with:
- **Subject**: "AI Sleep Constructs Security Vulnerability"
- **Description**: Detailed description of the vulnerability
- **Impact**: Potential impact and affected components
- **Steps to Reproduce**: Clear steps to reproduce the issue
- **Suggested Fix**: If you have suggestions for remediation

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Varies by severity, typically 14-30 days

### 4. Disclosure Process

1. Vulnerability is confirmed and assessed
2. Fix is developed and tested
3. Security advisory is prepared
4. Patch is released
5. Public disclosure (coordinated with reporter)

## Security Best Practices

### When Using AI Sleep Constructs

#### 1. Validate Input Data

```python
# Always validate metric values
def safe_track_metric(monitor, name, value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric value must be numeric, got {type(value)}")
    if np.isnan(value) or np.isinf(value):
        raise ValueError(f"Metric value must be finite, got {value}")
    monitor.track_metric(name, value)
```

#### 2. Secure State Serialization

```python
import os
from pathlib import Path

# Use secure file permissions
def save_state_securely(controller, filepath):
    path = Path(filepath)
    controller.export_state(path)
    # Set secure permissions (owner read/write only)
    os.chmod(path, 0o600)
```

#### 3. Sanitize Model IDs

```python
import re

def safe_model_id(model_id: str) -> str:
    """Sanitize model ID to prevent path traversal."""
    # Remove any path separators
    model_id = re.sub(r'[/\\]', '_', model_id)
    # Remove special characters
    model_id = re.sub(r'[^\w\-.]', '_', model_id)
    return model_id

controller = AISleepController(
    model_id=safe_model_id(user_input_model_id)
)
```

#### 4. Limit Resource Consumption

```python
# Set maximum cache sizes
MAX_KV_CACHE_SIZE = 1000
MAX_HISTORY_SIZE = 10000

class SecureLMModelState(LMModelState):
    def update_kv_cache(self, layer_id, keys, values):
        if len(self.kv_cache) >= MAX_KV_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(
                self.kv_cache.keys(),
                key=lambda k: self.kv_cache[k]['timestamp']
            )
            del self.kv_cache[oldest_key]
        super().update_kv_cache(layer_id, keys, values)
```

#### 5. Validate State Transitions

```python
# Add validation wrapper
def safe_transition(state, new_mode):
    """Safely transition with additional validation."""
    if state.current_mode == new_mode:
        return True
    
    # Validate prerequisites
    if new_mode == SleepMode.DEEP_SLEEP:
        if state.current_mode not in [SleepMode.LIGHT_SLEEP, SleepMode.TRANSITIONING]:
            raise ValueError("Must be in LIGHT_SLEEP to enter DEEP_SLEEP")
    
    try:
        state.transition_to(new_mode)
        return True
    except Exception as e:
        logger.error(f"Transition failed: {e}")
        return False
```

### 6. Monitor for Anomalous Behavior

```python
# Set up security-focused alerts
def security_alert_callback(alert):
    """Handle potential security-related alerts."""
    security_indicators = [
        "gradient_explosion",
        "cache_overflow",
        "memory_pressure"
    ]
    
    if any(indicator in alert.message.lower() for indicator in security_indicators):
        logger.critical(f"Security alert: {alert.message}")
        # Take protective action
        controller.wake_up()  # Exit sleep mode
        controller.model_state.clear_kv_cache()  # Clear caches

monitor.register_alert_callback(security_alert_callback)
```

## Known Security Considerations

### 1. State Persistence

**Risk**: Serialized state files may contain sensitive information.

**Mitigation**:
- Store state files in secure locations with appropriate permissions
- Consider encrypting state files at rest
- Implement secure deletion of old state files

```python
import pickle
from cryptography.fernet import Fernet

def save_encrypted_state(controller, filepath, key):
    """Save state with encryption."""
    # Serialize state
    import io
    buffer = io.BytesIO()
    pickle.dump(controller.model_state, buffer)
    data = buffer.getvalue()
    
    # Encrypt
    f = Fernet(key)
    encrypted = f.encrypt(data)
    
    # Save
    with open(filepath, 'wb') as file:
        file.write(encrypted)
```

### 2. Callback Execution

**Risk**: User-provided callbacks execute arbitrary code.

**Mitigation**:
- Callbacks run with same privileges as main process
- Validate callback functions before registration
- Implement timeout mechanisms

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """Context manager for timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError("Callback timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def safe_callback_execution(callback, *args):
    """Execute callback with timeout."""
    try:
        with timeout(5):  # 5 second timeout
            callback(*args)
    except TimeoutError:
        logger.error("Callback timed out")
    except Exception as e:
        logger.error(f"Callback error: {e}")
```

### 3. Resource Exhaustion

**Risk**: Unbounded growth of caches and histories could lead to DoS.

**Mitigation**:
- Implement maximum sizes for all data structures
- Regularly clean up old data
- Monitor memory usage

```python
class ResourceLimitedController(AISleepController):
    MAX_MEMORY_MB = 1000
    
    def _check_memory_usage(self):
        """Check if memory limit exceeded."""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.MAX_MEMORY_MB:
            logger.warning(f"Memory limit exceeded: {memory_mb}MB")
            # Clear caches
            self.model_state.clear_kv_cache()
            self.model_state.semantic_memory.clear()
            return True
        return False
```

### 4. Pickle Deserialization

**Risk**: Loading untrusted pickle files can execute arbitrary code.

**Mitigation**:
- Only load state files from trusted sources
- Implement allowlist of safe classes
- Consider using safer serialization formats (JSON, protobuf)

```python
import pickle
import io

class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows safe classes."""
    
    SAFE_CLASSES = {
        'builtins.dict',
        'builtins.list',
        'builtins.str',
        'builtins.int',
        'builtins.float',
        'datetime.datetime',
        'numpy.ndarray',
    }
    
    def find_class(self, module, name):
        full_name = f"{module}.{name}"
        if full_name not in self.SAFE_CLASSES:
            raise pickle.UnpicklingError(
                f"Class {full_name} not in allowlist"
            )
        return super().find_class(module, name)

def safe_load_state(filepath):
    """Load state with restricted unpickler."""
    with open(filepath, 'rb') as f:
        return SafeUnpickler(f).load()
```

### 5. Denial of Service

**Risk**: Malicious metrics or configurations could cause excessive computation.

**Mitigation**:
- Validate all input parameters
- Implement timeouts for long-running operations
- Rate limit sleep cycle initiations

```python
from datetime import datetime, timedelta

class RateLimitedController(AISleepController):
    MIN_SLEEP_INTERVAL = timedelta(minutes=5)
    
    def initiate_sleep(self, trigger, mode=None):
        """Initiate sleep with rate limiting."""
        now = datetime.now()
        if self.last_sleep_time:
            elapsed = now - self.last_sleep_time
            if elapsed < self.MIN_SLEEP_INTERVAL:
                logger.warning(f"Sleep cycle rate limited. Wait {self.MIN_SLEEP_INTERVAL - elapsed}")
                return False
        
        return super().initiate_sleep(trigger, mode)
```

## Security Checklist

When deploying AI Sleep Constructs in production:

- [ ] Validate all input data
- [ ] Secure state file storage
- [ ] Implement resource limits
- [ ] Monitor for anomalous behavior
- [ ] Set up proper logging
- [ ] Use secure callback practices
- [ ] Regularly update dependencies
- [ ] Review and test configurations
- [ ] Implement access controls
- [ ] Plan incident response

## Dependency Security

### Monitoring Dependencies

Regularly check for vulnerabilities in dependencies:

```bash
# Install safety
pip install safety

# Check for known vulnerabilities
safety check

# Or use pip-audit
pip install pip-audit
pip-audit
```

### Pinning Dependencies

For production, pin exact versions:

```
# requirements-prod.txt
numpy==1.21.0
scipy==1.7.0
transformers==4.20.0
```

## Security Updates

Subscribe to security advisories:
- GitHub Security Advisories for this repository
- Python security mailing list
- NumPy/SciPy security announcements

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities. Contributors who report valid security issues will be acknowledged in:
- CHANGELOG.md
- Security advisories
- Repository documentation

## Contact

For security concerns, please contact the maintainers through:
- Email: [Use repository contact information]
- GitHub Security Advisory: Private vulnerability reporting

## Legal

This security policy is subject to the project's CC-BY-NC-SA 4.0 license. Security updates and patches will be provided as-is without warranty.

---

Last updated: 2025-01-07
