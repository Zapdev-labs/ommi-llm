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

# No tests currently exist in the repo
```

### Code Quality
```bash
# Format code
black src/

# Lint with ruff
ruff check src/
ruff check --fix src/

# Type checking
mypy src/
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
```

## Architecture

### Core Layer-Wise Inference

The `LayerWiseInferenceEngine` (in `src/ommi_llm/core/engine.py`) is the abstract base class that implements the memory-efficient inference pattern:

1. **Meta Device Pattern**: Models are initialized with `init_empty_weights()` creating a skeleton with zero memory footprint
2. **Layer-wise Processing**: One transformer layer is loaded from disk, computed, then immediately freed
3. **Async Prefetching**: ThreadPoolExecutor overlaps layer N+1 loading with layer N computation (CUDA only)

Key methods to understand:
- `init_model()`: Creates empty model skeleton, fixes config attributes for edge cases (Qwen3.5, etc.)
- `forward()`: Layer-wise forward pass with optional prefetching
- `load_layer_to_cpu()` / `move_layer_to_device()`: Weight loading pipeline
- `unload_layer()`: Frees layer memory after computation

### Model Architecture Detection

`AutoModel` (in `src/ommi_llm/core/auto_model.py`) handles architecture detection:

1. Checks `config.architectures` for exact match in `ARCHITECTURE_REGISTRY`
2. Falls back to `config.model_type` pattern matching in `MODEL_TYPE_PATTERNS`
3. Uses `LlamaAdapter` for llama-like causal LM patterns
4. Falls back to `GenericAdapter` with auto-detection for unknown architectures

To add support for a new architecture:
1. Create adapter class in `src/ommi_llm/adapters/<model>.py`
2. Implement `set_layer_names_dict()` defining layer paths
3. Register in `ARCHITECTURE_REGISTRY` or `MODEL_TYPE_PATTERNS`

### Adapter System

Adapters (`src/ommi_llm/adapters/`) define architecture-specific layer naming:

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

The engine uses these paths with `_get_nested_attr()` to locate layers in the model hierarchy.

### Persistence Layer

`src/ommi_llm/persistence/` handles model storage:

- **loader.py**: `load_layer_weights()` loads SafeTensors shards
- **sharder.py**: `ModelSharder` pre-shards models into per-layer files for efficient loading

Sharding is done ahead-of-time to enable fast layer-by-layer disk I/O during inference.

### Configuration Edge Cases

The engine handles models with incomplete configs (see `init_model()` in engine.py):
- Qwen3.5 models often lack `vocab_size`, `hidden_size`, `num_hidden_layers`
- Fallback values are set based on `model_type` patterns
- `text_config` nested structures are checked as alternative sources

### Project Structure

```
src/ommi_llm/
├── core/               # Inference engine
│   ├── engine.py       # LayerWiseInferenceEngine base class
│   └── auto_model.py   # Architecture detection
├── adapters/           # Model-specific implementations
│   ├── base.py         # ModelAdapter base class
│   ├── llama.py        # Llama/llama-like models (also used for Gemma, Phi, etc.)
│   ├── mistral.py
│   ├── mixtral.py      # MoE support
│   ├── qwen.py         # Qwen/Qwen2/Qwen3/Qwen3.5
│   └── generic.py      # Auto-detection fallback
├── persistence/        # Model sharding & loading
│   ├── loader.py       # SafeTensors loading
│   └── sharder.py      # Model pre-sharding
├── server/             # MCP server
│   └── mcp_server.py   # Model Context Protocol server
├── skills/             # Pluggable optimizations
│   └── registry.py     # Skill registry
├── utils/              # Utilities
│   ├── memory.py       # Memory management
│   └── constants.py    # Architecture lists
├── cli.py              # Typer-based CLI
└── __init__.py
```

## Important Notes

- **Prefetching**: Disabled for compressed models (not yet supported)
- **Attention**: Uses BetterTransformer for Flash Attention if available, falls back to SDPA
- **Quantization**: `compression` parameter supports "4bit" and "8bit" via bitsandbytes
- **Memory Strategy**: Peak VRAM ~4GB for 70B models (1.6GB layer + 0.4GB activations + cache)
- **Trade-off**: Layer-wise inference uses ~4GB VRAM but is 35-100+ seconds per token
