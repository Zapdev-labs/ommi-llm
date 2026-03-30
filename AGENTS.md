# AGENTS.md - Ommi LLM

## Commands

### Development Setup
```bash
# Install in development mode
pip install -e ".[dev]"

# Install with all dependencies
pip install -e .
```

### Testing
```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_adapters.py

# Run a single test class
pytest tests/test_adapters.py::TestLlamaAdapter

# Run a single test method
pytest tests/test_adapters.py::TestLlamaAdapter::test_layer_names_dict

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=ommi_llm tests/
```

### Code Quality
```bash
# Format code (Black, 100 char line length)
black src/

# Check formatting without changes
black --check src/

# Lint with ruff
ruff check src/

# Auto-fix lint issues
ruff check --fix src/

# Type checking (strict mode)
mypy src/

# Run all quality checks
black src/ && ruff check src/ && mypy src/
```

### Building
```bash
# Build package (uses hatchling)
python -m build

# Or install build tool
pip install hatchling
python -m hatchling build
```

### CLI Usage
```bash
# Entry point is `ommi`
ommi --help

# Common commands
ommi memory                    # Show memory stats
ommi list-architectures        # List supported model architectures
ommi download <model_id>       # Download from HuggingFace
ommi shard <model> <output>    # Shard model for layer-wise loading
ommi generate <model> <prompt> # Generate text
ommi tui                       # Launch TUI
```

## Code Style Guidelines

### Imports (Ruff-enforced)
Order: standard library → third-party → local
```python
# Standard library
import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import torch
import torch.nn as nn
from rich.console import Console
from transformers import AutoConfig

# Local imports (absolute)
from .utils.memory import MemoryManager
from .core.engine import LayerWiseInferenceEngine
```

### Formatting (Black)
- Line length: 100 characters
- Target Python: 3.9+
- Use trailing commas for multi-line collections

### Type Hints (Strict mypy)
- All function parameters and return types must be annotated
- Use `Optional[T]` for nullable values
- Use `Dict`, `List`, `Tuple` from typing (not builtins)
- Enable strict mode: `disallow_untyped_defs = true`
```python
def process_layer(
    self,
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Process a single layer."""
    return hidden_states
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `LayerWiseInferenceEngine`)
- Functions/variables: `snake_case` (e.g., `load_layer_weights`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `SUPPORTED_ARCHITECTURES`)
- Private methods: `_leading_underscore` (e.g., `_get_nested_attr`)
- Abstract methods: marked with `@abstractmethod`

### Docstrings (Google Style)
All modules, classes, and functions must have docstrings:
```python
"""
Brief description of the module.
"""

class MyClass:
    """
    Class description.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
    """

    def method(self, param: str) -> int:
        """
        Method description.

        Args:
            param: Description of param

        Returns:
            Description of return value

        Raises:
            ValueError: When invalid input
        """
        return 0
```

### Error Handling
- Use specific exceptions, not bare `except:`
- Log errors with `logger.exception()` for stack traces
- CLI commands should use `raise typer.Exit(1)` on failure
- Validation should happen early with clear error messages
```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise typer.Exit(1)
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

### Testing (pytest)
- Test class names: `Test{Feature}` (e.g., `TestLlamaAdapter`)
- Test method names: `test_{description}` (e.g., `test_layer_names_dict`)
- Use fixtures for common setup
- Mock external dependencies (torch.cuda, API calls)
```python
class TestMyFeature:
    """Test my feature."""

    @pytest.fixture
    def adapter(self):
        """Create adapter fixture."""
        with patch("torch.cuda.is_available", return_value=False):
            return MyAdapter()

    def test_basic_operation(self, adapter):
        """Test basic operation works."""
        result = adapter.operation()
        assert result == expected
```

## Architecture

### Core Layer-Wise Inference

The `LayerWiseInferenceEngine` (in `src/ommi_llm_zapdev/core/engine.py`) implements memory-efficient inference:

1. **Meta Device Pattern**: Models initialized with `init_empty_weights()` create a skeleton with zero memory footprint
2. **Layer-wise Processing**: One transformer layer loaded from disk, computed, then immediately freed
3. **Async Prefetching**: ThreadPoolExecutor overlaps layer N+1 loading with layer N computation (CUDA only)

Key methods:
- `init_model()`: Creates empty model skeleton, fixes config attributes for edge cases
- `forward()`: Layer-wise forward pass with optional prefetching
- `load_layer_to_cpu()` / `move_layer_to_device()`: Weight loading pipeline
- `unload_layer()`: Frees layer memory after computation

### Model Architecture Detection

`AutoModel` (in `src/ommi_llm_zapdev/core/auto_model.py`) handles architecture detection:

1. Checks `config.architectures` for exact match in `ARCHITECTURE_REGISTRY`
2. Falls back to `config.model_type` pattern matching in `MODEL_TYPE_PATTERNS`
3. Uses `LlamaAdapter` for llama-like causal LM patterns
4. Falls back to `GenericAdapter` with auto-detection for unknown architectures

To add support for a new architecture:
1. Create adapter class in `src/ommi_llm_zapdev/adapters/<model>.py`
2. Implement `set_layer_names_dict()` defining layer paths
3. Register in `ARCHITECTURE_REGISTRY` or `MODEL_TYPE_PATTERNS`

### Adapter System

Adapters define architecture-specific layer naming:
```python
class LlamaAdapter(ModelAdapter):
    def set_layer_names_dict(self) -> None:
        self.layer_names_dict = {
            "embed": "model.embed_tokens",      # Embedding layer path
            "layer_prefix": "model.layers",       # Transformer layers container
            "norm": "model.norm",                 # Final norm layer
            "lm_head": "lm_head",                 # Output head
        }
```

The engine uses these paths with `_get_nested_attr()` to locate layers.

### Project Structure

```
src/ommi_llm_zapdev/
├── core/               # Inference engine
│   ├── engine.py       # LayerWiseInferenceEngine base class
│   └── auto_model.py   # Architecture detection
├── adapters/           # Model-specific implementations
│   ├── base.py         # ModelAdapter base class
│   ├── llama.py        # Llama/llama-like models
│   ├── mistral.py      # Mistral models
│   ├── mixtral.py      # MoE support
│   ├── qwen.py         # Qwen/Qwen2/Qwen3/Qwen3.5
│   ├── baichuan.py     # Baichuan models
│   ├── chatglm.py      # ChatGLM models
│   ├── internlm.py     # InternLM models
│   └── generic.py      # Auto-detection fallback
├── persistence/        # Model sharding & loading
│   ├── loader.py       # SafeTensors loading
│   └── sharder.py      # Model pre-sharding
├── compression/        # Quantization support
│   └── compressor.py   # 4-bit/8-bit compression
├── server/             # MCP server
│   └── mcp_server.py   # Model Context Protocol server
├── skills/             # Pluggable optimizations
│   └── registry.py     # Skill registry
├── utils/              # Utilities
│   ├── memory.py       # Memory management
│   └── constants.py    # Architecture lists
├── cli.py              # Typer-based CLI
├── cli_compression.py  # Compression CLI commands
├── tui_launcher.py   # TUI launcher
└── __init__.py
```

## Important Notes

- **Prefetching**: Disabled for compressed models (not yet supported)
- **Attention**: Uses BetterTransformer for Flash Attention if available, falls back to SDPA
- **Quantization**: `compression` parameter supports "4bit" and "8bit" via bitsandbytes
- **Memory Strategy**: Peak VRAM ~4GB for 70B models (1.6GB layer + 0.4GB activations + cache)
- **Trade-off**: Layer-wise inference uses ~4GB VRAM but is 35-100+ seconds per token
- **Package Name**: The actual package is `ommi_llm_zapdev`, not `ommi_llm`
- **CLI Entry Point**: Defined in pyproject.toml as `ommi = "ommi_llm_zapdev.cli:main"`
