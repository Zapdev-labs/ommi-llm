"""
Ommi LLM - Run 70B+ LLMs on consumer GPUs with layer-wise inference.

Ommi LLM is a memory-efficient inference engine that enables running large language
models with 70B+ parameters on consumer GPUs with limited VRAM (4-8GB) through
innovative layer-wise processing.

Core Features:
- Layer-wise inference: Process one transformer layer at a time
- Meta device pattern: Virtual model skeleton with zero memory footprint
- SafeTensor sharding: Pre-shard models for efficient loading
- Async prefetching: Overlap I/O with computation
- Quantization support: 4-bit and 8-bit compression for 3x speedup
- MCP server integration: Model management through Model Context Protocol
- Skill-based configuration: Pluggable optimizations

Example:
    >>> from ommi_llm import AutoModel
    >>> model = AutoModel.from_pretrained("meta-llama/Llama-2-70b")
    >>> output = model.generate("What is the capital of France?")
"""

__version__ = "0.2.3"
__author__ = "Ommi Team"

from .core.engine import LayerWiseInferenceEngine
from .core.auto_model import AutoModel
from .adapters.base import ModelAdapter
from .skills.registry import SkillRegistry, get_skill_registry
from .utils.memory import MemoryManager

__all__ = [
    "LayerWiseInferenceEngine",
    "AutoModel",
    "ModelAdapter",
    "SkillRegistry",
    "get_skill_registry",
    "MemoryManager",
]
