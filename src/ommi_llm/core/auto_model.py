"""
Auto model loader with architecture detection.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from transformers import AutoConfig

from ..adapters.llama import LlamaAdapter
from ..adapters.mistral import MistralAdapter
from ..adapters.qwen import QwenAdapter
from ..adapters.baichuan import BaichuanAdapter
from ..adapters.chatglm import ChatGLMAdapter
from ..adapters.internlm import InternLMAdapter
from ..adapters.mixtral import MixtralAdapter
from .engine import LayerWiseInferenceEngine

# Registry of supported architectures
ARCHITECTURE_REGISTRY: Dict[str, Type[LayerWiseInferenceEngine]] = {
    # Llama family
    "LlamaForCausalLM": LlamaAdapter,
    "LlamaModel": LlamaAdapter,
    # Mistral family
    "MistralForCausalLM": MistralAdapter,
    "MistralModel": MistralAdapter,
    # Mixtral MoE
    "MixtralForCausalLM": MixtralAdapter,
    "MixtralModel": MixtralAdapter,
    # Qwen family
    "Qwen2ForCausalLM": QwenAdapter,
    "QwenForCausalLM": QwenAdapter,
    "Qwen2Model": QwenAdapter,
    "QwenModel": QwenAdapter,
    # Baichuan
    "BaichuanForCausalLM": BaichuanAdapter,
    "BaichuanModel": BaichuanAdapter,
    # ChatGLM
    "ChatGLMForConditionalGeneration": ChatGLMAdapter,
    "ChatGLMModel": ChatGLMAdapter,
    # InternLM
    "InternLMForCausalLM": InternLMAdapter,
    "InternLM2ForCausalLM": InternLMAdapter,
    "InternLMModel": InternLMAdapter,
}


class AutoModel:
    """
    Auto model loader that detects architecture and returns appropriate adapter.

    Similar to transformers.AutoModel, but returns layer-wise inference engines.

    Example:
        >>> from ommi_llm import AutoModel
        >>> model = AutoModel.from_pretrained("meta-llama/Llama-2-70b-chat")
        >>> output = model.generate("Hello, how are you?")
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_memory: Optional[Dict[str, Union[int, str]]] = None,
        prefetching: bool = True,
        compression: Optional[str] = None,
        local_path: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> LayerWiseInferenceEngine:
        """
        Load model from pretrained checkpoint.

        Automatically detects model architecture and returns appropriate adapter.

        Args:
            model_name: HuggingFace model ID or local path
            device: Target device (cuda/cpu/mps)
            dtype: Data type (float16/bfloat16/float32)
            max_memory: Memory limits per device
            prefetching: Enable async layer prefetching
            compression: Quantization (4bit/8bit)
            local_path: Local path to model files
            trust_remote_code: Allow custom architectures

        Returns:
            Initialized layer-wise inference engine
        """
        # Determine path
        path = local_path or model_name

        # Load config to detect architecture
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)

        # Get architecture
        architectures = getattr(config, "architectures", [])
        if not architectures:
            raise ValueError(f"No architecture found in config for {model_name}")

        architecture = architectures[0]

        # Find adapter
        if architecture not in ARCHITECTURE_REGISTRY:
            # Try generic fallback
            raise ValueError(
                f"Architecture '{architecture}' not supported. "
                f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}"
            )

        adapter_class = ARCHITECTURE_REGISTRY[architecture]

        # Initialize adapter
        return adapter_class.from_pretrained(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_memory=max_memory,
            prefetching=prefetching,
            compression=compression,
            local_path=local_path,
            **kwargs,
        )

    @classmethod
    def register_adapter(
        cls, architecture: str, adapter_class: Type[LayerWiseInferenceEngine]
    ) -> None:
        """
        Register a custom adapter for an architecture.

        Args:
            architecture: Architecture name from config
            adapter_class: Adapter class to use
        """
        ARCHITECTURE_REGISTRY[architecture] = adapter_class

    @classmethod
    def list_supported_architectures(cls) -> Dict[str, str]:
        """
        List all supported architectures.

        Returns:
            Dictionary mapping architecture names to adapter classes
        """
        return {arch: adapter.__name__ for arch, adapter in ARCHITECTURE_REGISTRY.items()}
