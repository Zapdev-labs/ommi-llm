"""
Auto model loader with architecture detection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from transformers import AutoConfig

from ..adapters.llama import LlamaAdapter
from ..adapters.mistral import MistralAdapter
from ..adapters.qwen import QwenAdapter
from ..adapters.baichuan import BaichuanAdapter
from ..adapters.chatglm import ChatGLMAdapter
from ..adapters.internlm import InternLMAdapter
from ..adapters.mixtral import MixtralAdapter
from ..adapters.generic import GenericAdapter
from .engine import LayerWiseInferenceEngine

logger = logging.getLogger(__name__)

# Registry of supported architectures with their adapters
ARCHITECTURE_REGISTRY: Dict[str, Type[LayerWiseInferenceEngine]] = {
    # Llama family (all versions)
    "LlamaForCausalLM": LlamaAdapter,
    "LlamaModel": LlamaAdapter,
    # Mistral family
    "MistralForCausalLM": MistralAdapter,
    "MistralModel": MistralAdapter,
    # Mixtral MoE
    "MixtralForCausalLM": MixtralAdapter,
    "MixtralModel": MixtralAdapter,
    # Qwen family (Qwen, Qwen2, Qwen3, Qwen3.5)
    "Qwen2ForCausalLM": QwenAdapter,
    "QwenForCausalLM": QwenAdapter,
    "Qwen2Model": QwenAdapter,
    "QwenModel": QwenAdapter,
    "Qwen3ForCausalLM": QwenAdapter,
    "Qwen3Model": QwenAdapter,
    "Qwen3_5ForCausalLM": QwenAdapter,
    "Qwen3_5Model": QwenAdapter,
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
    # Gemma family
    "GemmaForCausalLM": LlamaAdapter,
    "GemmaModel": LlamaAdapter,
    "Gemma2ForCausalLM": LlamaAdapter,
    "Gemma2Model": LlamaAdapter,
    # Phi family
    "PhiForCausalLM": LlamaAdapter,
    "PhiModel": LlamaAdapter,
    "Phi3ForCausalLM": LlamaAdapter,
    "Phi3Model": LlamaAdapter,
    # Falcon family
    "FalconForCausalLM": LlamaAdapter,
    "FalconModel": LlamaAdapter,
    # StableLM
    "StableLmForCausalLM": LlamaAdapter,
    "StableLmModel": LlamaAdapter,
    # GPT-Neo family
    "GPTNeoForCausalLM": LlamaAdapter,
    "GPTNeoModel": LlamaAdapter,
    # GPT-J
    "GPTJForCausalLM": LlamaAdapter,
    "GPTJModel": LlamaAdapter,
    # OPT
    "OPTForCausalLM": LlamaAdapter,
    "OPTModel": LlamaAdapter,
    # MPT
    "MptForCausalLM": LlamaAdapter,
    "MptModel": LlamaAdapter,
    # Pythia/GPT-NeoX
    "GPTNeoXForCausalLM": LlamaAdapter,
    "GPTNeoXModel": LlamaAdapter,
    # BLOOM
    "BloomForCausalLM": LlamaAdapter,
    "BloomModel": LlamaAdapter,
    # CodeLlama/CodeQwen
    "CodeLlamaForCausalLM": LlamaAdapter,
    # Yi
    "YiForCausalLM": LlamaAdapter,
    "YiModel": LlamaAdapter,
    # DeepSeek
    "DeepseekForCausalLM": LlamaAdapter,
    "DeepseekModel": LlamaAdapter,
    "DeepseekV2ForCausalLM": LlamaAdapter,
    "DeepseekV3ForCausalLM": LlamaAdapter,
    # StarCoder
    "GPTBigCodeForCausalLM": LlamaAdapter,
    # DBRX
    "DbrxForCausalLM": LlamaAdapter,
    "DbrxModel": LlamaAdapter,
    # Command-R
    "CohereForCausalLM": LlamaAdapter,
    # OLMo
    "OlmoForCausalLM": LlamaAdapter,
    # Jamba
    "JambaForCausalLM": LlamaAdapter,
    # Minimax
    "MiniMaxForCausalLM": LlamaAdapter,
    "MiniMaxText01ForCausalLM": LlamaAdapter,
    # Granite (IBM)
    "GraniteForCausalLM": LlamaAdapter,
    "GraniteMoeForCausalLM": LlamaAdapter,
    # Arctic (Snowflake)
    "ArcticForCausalLM": LlamaAdapter,
    # Nemotron (NVIDIA)
    "NemotronForCausalLM": LlamaAdapter,
    "Nemotron3ForCausalLM": LlamaAdapter,
    # Exaone (LG)
    "ExaoneForCausalLM": LlamaAdapter,
    # TeleChat (China Telecom)
    "TeleChatForCausalLM": LlamaAdapter,
    # Solar (Upstage)
    "SolarForCausalLM": LlamaAdapter,
    # Grok
    "GrokForCausalLM": LlamaAdapter,
    # JetMoe
    "JetMoeForCausalLM": LlamaAdapter,
    # Chameleon (Meta)
    "ChameleonForCausalLM": LlamaAdapter,
    # Cerebras
    "CerebrasForCausalLM": LlamaAdapter,
    # Persimmon
    "PersimmonForCausalLM": LlamaAdapter,
}

# Pattern-based architecture detection for model types
# Maps model_type to adapter class
MODEL_TYPE_PATTERNS: Dict[str, Type[LayerWiseInferenceEngine]] = {
    # Core families
    "llama": LlamaAdapter,
    "mistral": MistralAdapter,
    "mixtral": MixtralAdapter,
    "qwen": QwenAdapter,
    "qwen2": QwenAdapter,
    "qwen3": QwenAdapter,
    "qwen3_5": QwenAdapter,
    "baichuan": BaichuanAdapter,
    "chatglm": ChatGLMAdapter,
    "internlm": InternLMAdapter,
    "internlm2": InternLMAdapter,
    # Extended support
    "gemma": LlamaAdapter,
    "gemma2": LlamaAdapter,
    "phi": LlamaAdapter,
    "phi3": LlamaAdapter,
    "falcon": LlamaAdapter,
    "stablelm": LlamaAdapter,
    "gpt_neo": LlamaAdapter,
    "gptj": LlamaAdapter,
    "gpt2": LlamaAdapter,
    "opt": LlamaAdapter,
    "mpt": LlamaAdapter,
    "gpt_neox": LlamaAdapter,
    "bloom": LlamaAdapter,
    "codellama": LlamaAdapter,
    "yi": LlamaAdapter,
    "deepseek": LlamaAdapter,
    "deepseek_v2": LlamaAdapter,
    "deepseek_v3": LlamaAdapter,
    "gpt_bigcode": LlamaAdapter,
    "dbrx": LlamaAdapter,
    "cohere": LlamaAdapter,
    "olmo": LlamaAdapter,
    "jamba": LlamaAdapter,
    "minimax": LlamaAdapter,
    "minimax_text": LlamaAdapter,
    "granite": LlamaAdapter,
    "granite_moe": LlamaAdapter,
    "arctic": LlamaAdapter,
    "nemotron": LlamaAdapter,
    "nemotron3": LlamaAdapter,
    "exaone": LlamaAdapter,
    "telechat": LlamaAdapter,
    "solar": LlamaAdapter,
    "grok": LlamaAdapter,
    "jetmoe": LlamaAdapter,
    "chameleon": LlamaAdapter,
    "cerebras": LlamaAdapter,
    "persimmon": LlamaAdapter,
    "starcoder": LlamaAdapter,
    "starcoder2": LlamaAdapter,
    "codeqwen": QwenAdapter,
    "tulu": LlamaAdapter,
    "openllama": LlamaAdapter,
    "vicuna": LlamaAdapter,
    "alpaca": LlamaAdapter,
    "guanaco": LlamaAdapter,
    "koala": LlamaAdapter,
    "wizardlm": LlamaAdapter,
    "manticore": LlamaAdapter,
    "openbuddy": LlamaAdapter,
    "tigerbot": LlamaAdapter,
    "xverse": LlamaAdapter,
    "aquilachat": LlamaAdapter,
    "seallm": LlamaAdapter,
}

# Model types that use generic adapter with llama-like patterns
GENERIC_LLAMA_LIKE_PATTERNS = ["causallm", "lm", "gpt", "transformer", "decoder"]


class AutoModel:
    """
    Auto model loader that detects architecture and returns appropriate adapter.

    Similar to transformers.AutoModel, but returns layer-wise inference engines.
    Supports ALL transformer-based language models through automatic detection.

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
        Falls back to GenericAdapter with auto-detection for unknown architectures.

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
        model_type = getattr(config, "model_type", "").lower()

        if not architectures and not model_type:
            raise ValueError(f"No architecture or model_type found in config for {model_name}")

        # Try to find adapter - prioritize exact architecture match
        adapter_class = None

        # 1. Check exact architecture match
        if architectures:
            architecture = architectures[0]
            if architecture in ARCHITECTURE_REGISTRY:
                adapter_class = ARCHITECTURE_REGISTRY[architecture]
                logger.info(f"Found exact architecture match: {architecture}")
            else:
                # Try case-insensitive match
                arch_lower = architecture.lower()
                for arch, adapter in ARCHITECTURE_REGISTRY.items():
                    if arch.lower() == arch_lower:
                        adapter_class = adapter
                        logger.info(f"Found case-insensitive match: {arch} -> {architecture}")
                        break

        # 2. Check model_type patterns
        if adapter_class is None and model_type:
            if model_type in MODEL_TYPE_PATTERNS:
                adapter_class = MODEL_TYPE_PATTERNS[model_type]
                logger.info(f"Found model_type pattern match: {model_type}")
            else:
                # Try partial matches
                for pattern, adapter in MODEL_TYPE_PATTERNS.items():
                    if pattern in model_type or model_type in pattern:
                        adapter_class = adapter
                        logger.info(f"Found partial model_type match: {model_type} -> {pattern}")
                        break

        # 3. Check for llama-like patterns in architecture name
        if adapter_class is None and architectures:
            arch_lower = architectures[0].lower()
            # Check if it's a causal LM (most use llama-like patterns)
            if "forcausallm" in arch_lower or "forconditionalgeneration" in arch_lower:
                # Try llama-like pattern first
                adapter_class = LlamaAdapter
                logger.info(f"Using llama-like adapter for: {architectures[0]}")

        # 4. Use GenericAdapter with auto-detection as fallback
        if adapter_class is None:
            logger.warning(
                f"Unknown architecture: {architectures[0] if architectures else 'unknown'} "
                f"(model_type: {model_type}). Using GenericAdapter with auto-detection."
            )
            adapter_class = GenericAdapter

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
        logger.info(f"Registered adapter for {architecture}: {adapter_class.__name__}")

    @classmethod
    def register_model_type(
        cls, model_type: str, adapter_class: Type[LayerWiseInferenceEngine]
    ) -> None:
        """
        Register a model_type pattern with an adapter.

        Args:
            model_type: Model type pattern
            adapter_class: Adapter class to use
        """
        MODEL_TYPE_PATTERNS[model_type] = adapter_class
        logger.info(f"Registered model_type pattern: {model_type} -> {adapter_class.__name__}")

    @classmethod
    def list_supported_architectures(cls) -> Dict[str, str]:
        """
        List all supported architectures.

        Returns:
            Dictionary mapping architecture names to adapter classes
        """
        return {arch: adapter.__name__ for arch, adapter in ARCHITECTURE_REGISTRY.items()}

    @classmethod
    def list_supported_model_types(cls) -> Dict[str, str]:
        """
        List all supported model types.

        Returns:
            Dictionary mapping model types to adapter classes
        """
        return {model_type: adapter.__name__ for model_type, adapter in MODEL_TYPE_PATTERNS.items()}

    @classmethod
    def is_architecture_supported(cls, architecture: str) -> bool:
        """
        Check if an architecture is explicitly supported.

        Args:
            architecture: Architecture name to check

        Returns:
            True if supported, False otherwise
        """
        return architecture in ARCHITECTURE_REGISTRY

    @classmethod
    def can_load_with_generic(cls, model_name: str, trust_remote_code: bool = True) -> bool:
        """
        Check if a model can be loaded with GenericAdapter.

        This analyzes the model structure without loading weights.

        Args:
            model_name: Model name or path
            trust_remote_code: Allow custom architectures

        Returns:
            True if GenericAdapter can likely handle this model
        """
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

            # Check if it's a decoder-only/causal LM architecture
            architectures = getattr(config, "architectures", [])
            model_type = getattr(config, "model_type", "").lower()

            # Most decoder-only models can be handled generically
            is_causal_lm = (
                any("causal" in arch.lower() or "lm" in arch.lower() for arch in architectures)
                if architectures
                else False
            )

            return is_causal_lm or "decoder" in model_type or "causal" in model_type

        except Exception as e:
            logger.warning(f"Could not analyze model {model_name}: {e}")
            return False
