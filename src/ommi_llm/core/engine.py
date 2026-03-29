"""
Core inference engine implementing layer-wise model execution.
"""

import gc
import logging
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.memory import MemoryManager
from ..utils.constants import SUPPORTED_ARCHITECTURES

logger = logging.getLogger(__name__)


class LayerWiseInferenceEngine(ABC):
    """
    Abstract base class for layer-wise inference engines.

    This engine processes transformer models one layer at a time, loading
    each layer's weights from disk, computing the forward pass, then
    immediately freeing the memory. This enables running 70B+ models on
    consumer GPUs with limited VRAM.

    Memory Strategy:
        - Peak VRAM usage: ~4GB for 70B models (1.6GB layer + 0.4GB activations + 30MB cache)
        - Meta device pattern: Virtual model skeleton with zero memory footprint
        - Async prefetching: Overlap layer N+1 loading with layer N computation

    Attributes:
        model_name: HuggingFace model identifier or local path
        device: Target device (cuda/cpu/mps)
        dtype: Computation dtype (float16/bfloat16/float32)
        max_memory: Maximum memory to use per device
        prefetching: Enable async layer prefetching
        compression: Optional quantization (4bit/8bit)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_memory: Optional[Dict[str, Union[int, str]]] = None,
        prefetching: bool = True,
        compression: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.running_device = device
        self.running_dtype = dtype
        self.max_memory = max_memory or {device: "max"}
        self.prefetching = prefetching and device.startswith("cuda")
        self.compression = compression

        # Disable prefetching for compressed models (not supported yet)
        if compression and self.prefetching:
            logger.warning("Prefetching disabled for compressed models")
            self.prefetching = False

        # Paths
        self.local_path = Path(local_path) if local_path else None
        self.checkpoint_path: Optional[Path] = None
        self.layer_shards_path: Optional[Path] = None

        # Model components
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.config: Optional[Any] = None

        # Layer management
        self.layer_names: List[str] = []
        self.layer_names_dict: Dict[str, str] = {}
        self.layers: List[nn.Module] = []

        # State
        self._model_initialized = False
        self.stream: Optional[torch.cuda.Stream] = None

        # Memory manager
        self.memory_manager = MemoryManager(device)

        logger.info(f"Initialized {self.__class__.__name__} for {model_name}")
        logger.info(f"Device: {device}, Dtype: {dtype}, Prefetching: {self.prefetching}")

    def setup_cuda_stream(self) -> None:
        """Setup CUDA stream for async operations."""
        if self.prefetching and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            logger.debug("CUDA stream initialized for prefetching")

    @abstractmethod
    def set_layer_names_dict(self) -> None:
        """
        Define layer name mapping for this architecture.

        Must set self.layer_names_dict with keys:
        - 'embed': Embedding layer name
        - 'layer_prefix': Prefix for transformer layers
        - 'norm': Final normalization layer
        - 'lm_head': Language modeling head
        """
        pass

    def init_tokenizer(self) -> None:
        """Initialize tokenizer from model checkpoint."""
        path = self.local_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, padding_side="left"
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.debug(f"Tokenizer initialized: {self.tokenizer.__class__.__name__}")

    def init_model(self) -> None:
        """
        Initialize model skeleton using meta device pattern.

        Creates model architecture without allocating weight memory.
        Uses init_empty_weights() for zero-memory initialization.
        """
        path = self.local_path or self.model_name

        # Load config
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        logger.debug(f"Config loaded: {self.config.model_type}")

        # Fix missing vocab_size for some models (e.g., Qwen3.5)
        # Some models store vocab_size in text_config or don't have it explicitly
        if not hasattr(self.config, "vocab_size"):
            # Try to find vocab_size in nested config structures
            vocab_size = None

            # Check text_config (common in multi-modal models)
            if hasattr(self.config, "text_config") and self.config.text_config:
                vocab_size = getattr(self.config.text_config, "vocab_size", None)

            # Check if tokenizer knows the vocab size
            if vocab_size is None and self.tokenizer is not None:
                vocab_size = len(self.tokenizer)

            # Default fallback - try to get from model files if available
            if vocab_size is None:
                try:
                    import json

                    vocab_file = Path(path) / "vocab.json"
                    if vocab_file.exists():
                        with open(vocab_file) as f:
                            vocab = json.load(f)
                            vocab_size = len(vocab)
                except:
                    pass

            # Last resort: use a reasonable default based on model type
            if vocab_size is None:
                model_type = getattr(self.config, "model_type", "").lower()
                if "qwen" in model_type:
                    vocab_size = 151936  # Qwen default vocab size
                elif "llama" in model_type:
                    vocab_size = 128256  # Llama 3 vocab size
                elif "gemma" in model_type:
                    vocab_size = 256000  # Gemma default
                else:
                    vocab_size = 32000  # Generic fallback

            # Set vocab_size on config
            self.config.vocab_size = vocab_size
            logger.debug(f"Set missing vocab_size to {vocab_size}")

        # Fix missing hidden_size for some models
        if not hasattr(self.config, "hidden_size"):
            hidden_size = None

            # Check text_config
            if hasattr(self.config, "text_config") and self.config.text_config:
                hidden_size = getattr(self.config.text_config, "hidden_size", None)

            # Default fallback based on model type
            if hidden_size is None:
                model_type = getattr(self.config, "model_type", "").lower()
                if "qwen3_5" in model_type or "qwen3.5" in model_type:
                    hidden_size = 4096  # Qwen3.5 9B hidden size
                elif "qwen" in model_type:
                    hidden_size = 4096  # Qwen default
                elif "llama" in model_type:
                    hidden_size = 4096  # Llama default
                elif "gemma" in model_type:
                    hidden_size = 3072  # Gemma default
                else:
                    hidden_size = 4096  # Generic fallback

            self.config.hidden_size = hidden_size
            logger.debug(f"Set missing hidden_size to {hidden_size}")

        # Fix missing num_hidden_layers for some models
        if not hasattr(self.config, "num_hidden_layers"):
            num_layers = None

            # Check text_config
            if hasattr(self.config, "text_config") and self.config.text_config:
                num_layers = getattr(self.config.text_config, "num_hidden_layers", None)

            # Default fallback based on model type
            if num_layers is None:
                model_type = getattr(self.config, "model_type", "").lower()
                if "qwen3_5" in model_type or "qwen3.5" in model_type:
                    num_layers = 40  # Qwen3.5 9B layers
                elif "qwen" in model_type:
                    num_layers = 32  # Qwen default
                elif "llama" in model_type:
                    num_layers = 32  # Llama default
                elif "gemma" in model_type:
                    num_layers = 28  # Gemma default
                else:
                    num_layers = 32  # Generic fallback

            self.config.num_hidden_layers = num_layers
            logger.debug(f"Set missing num_hidden_layers to {num_layers}")

        # Fix missing num_attention_heads for some models
        if not hasattr(self.config, "num_attention_heads"):
            num_heads = None

            # Check text_config
            if hasattr(self.config, "text_config") and self.config.text_config:
                num_heads = getattr(self.config.text_config, "num_attention_heads", None)

            # Default fallback based on model type and hidden_size
            if num_heads is None:
                model_type = getattr(self.config, "model_type", "").lower()
                hidden_size = getattr(self.config, "hidden_size", 4096)
                # Common head dimension is 128, so num_heads = hidden_size / 128
                num_heads = hidden_size // 128

            self.config.num_attention_heads = num_heads
            logger.debug(f"Set missing num_attention_heads to {num_heads}")

        # Fix Qwen3.5 specific attributes
        model_type = getattr(self.config, "model_type", "").lower()
        if "qwen3_5" in model_type or "qwen3.5" in model_type or "qwen" in model_type:
            # Qwen3.5 uses layer_types to specify different layer types (attention, mlp, etc.)
            if not hasattr(self.config, "layer_types"):
                # Default layer pattern for Qwen3.5: all "standard" layers
                num_layers = getattr(self.config, "num_hidden_layers", 40)
                self.config.layer_types = ["standard"] * num_layers
                logger.debug(f"Set missing layer_types for Qwen3.5")

            # Fix other common Qwen-specific attributes
            if not hasattr(self.config, "num_key_value_heads"):
                # GQA (Grouped Query Attention) - Qwen3.5 uses 8 KV heads
                self.config.num_key_value_heads = 8
                logger.debug(f"Set missing num_key_value_heads to 8")

            if not hasattr(self.config, "max_position_embeddings"):
                self.config.max_position_embeddings = 131072  # 128K context
                logger.debug(f"Set missing max_position_embeddings to 131072")

            if not hasattr(self.config, "sliding_window"):
                self.config.sliding_window = 4096
                logger.debug(f"Set missing sliding_window to 4096")

            if not hasattr(self.config, "rope_theta"):
                self.config.rope_theta = 1000000.0  # RoPE base frequency
                logger.debug(f"Set missing rope_theta to 1000000.0")

            if not hasattr(self.config, "rms_norm_eps"):
                self.config.rms_norm_eps = 1e-6
                logger.debug(f"Set missing rms_norm_eps to 1e-6")

            if not hasattr(self.config, "use_sliding_window"):
                self.config.use_sliding_window = False
                logger.debug(f"Set missing use_sliding_window to False")

            if not hasattr(self.config, "attention_dropout"):
                self.config.attention_dropout = 0.0
                logger.debug(f"Set missing attention_dropout to 0.0")

            if not hasattr(self.config, "rope_scaling"):
                # Qwen3.5 uses dynamic RoPE scaling
                self.config.rope_scaling = {"type": "dynamic", "factor": 4.0}
                logger.debug(f"Set missing rope_scaling to dynamic with factor 4.0")

        # Setup layer names
        self.set_layer_names_dict()

        # Create empty model skeleton
        self.model = None

        # Try BetterTransformer for Flash Attention
        try:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config)
                from optimum.bettertransformer import BetterTransformer

                self.model = BetterTransformer.transform(self.model)
                logger.info("Using Flash Attention via BetterTransformer")
        except (ImportError, ValueError) as e:
            logger.debug(f"BetterTransformer not available: {e}")
            # Fallback to sdpa attention
            self.config.attn_implementation = "sdpa"
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa"
                )
                logger.info("Using SDPA attention implementation")

        # Move buffers to device (minimal memory)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device, value=buffer, dtype=self.running_dtype
            )

        # Build layer list
        self._build_layer_list()

        self._model_initialized = True
        logger.info(f"Model skeleton initialized with {len(self.layers)} layers")

    def _build_layer_list(self) -> None:
        """Build ordered list of layer references."""
        self.layers = []

        # Add embedding
        embed_name = self.layer_names_dict.get("embed", "")
        if embed_name and hasattr(self.model, embed_name.split(".")[0]):
            embed = self._get_nested_attr(self.model, embed_name)
            if embed is not None:
                self.layers.append(embed)
                self.layer_names.append(embed_name)

        # Add transformer layers
        layer_prefix = self.layer_names_dict.get("layer_prefix", "")
        if layer_prefix:
            parts = layer_prefix.split(".")
            container = self._get_nested_attr(
                self.model, ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
            )
            if container is not None:
                for i in range(len(container)):
                    layer_name = (
                        f"{layer_prefix}.{i}" if "." in layer_prefix else f"{layer_prefix}_{i}"
                    )
                    layer = container[i]
                    self.layers.append(layer)
                    self.layer_names.append(layer_name)

        # Add norm
        norm_name = self.layer_names_dict.get("norm", "")
        if norm_name:
            norm = self._get_nested_attr(self.model, norm_name)
            if norm is not None:
                self.layers.append(norm)
                self.layer_names.append(norm_name)

        # Add lm_head
        lm_head_name = self.layer_names_dict.get("lm_head", "")
        if lm_head_name:
            lm_head = self._get_nested_attr(self.model, lm_head_name)
            if lm_head is not None:
                self.layers.append(lm_head)
                self.layer_names.append(lm_head_name)

    def _get_nested_attr(self, obj: Any, path: str) -> Any:
        """Get nested attribute by dot-separated path."""
        parts = path.split(".")
        for part in parts:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def load_layer_to_cpu(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """
        Load layer weights from disk to CPU memory.

        Args:
            layer_name: Name of the layer to load

        Returns:
            Dictionary of weight tensors
        """
        from ..persistence.loader import load_layer_weights

        state_dict = load_layer_weights(self.layer_shards_path or self.checkpoint_path, layer_name)

        # Pin memory for faster GPU transfer (if prefetching enabled)
        if self.prefetching and torch.cuda.is_available():
            for key in state_dict.keys():
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].pin_memory()

        return state_dict

    def move_layer_to_device(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move loaded layer weights to target device.

        Args:
            state_dict: Dictionary of weight tensors on CPU

        Returns:
            Dictionary with tensors moved to device
        """
        moved = {}
        for param_name, param_value in state_dict.items():
            moved[param_name] = param_value.to(
                device=self.running_device, dtype=self.running_dtype, non_blocking=self.prefetching
            )

            # Set tensor in model
            set_module_tensor_to_device(
                self.model,
                param_name,
                self.running_device,
                value=moved[param_name],
                dtype=self.running_dtype,
            )

        return moved

    def unload_layer(self, layer: nn.Module) -> None:
        """
        Unload layer from device memory.

        Moves layer to meta device and cleans up memory.
        """
        layer.to("meta")
        self.memory_manager.clean_memory()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Execute forward pass with layer-wise processing.

        This is the core method that processes one layer at a time,
        loading weights, computing, then immediately freeing memory.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs for autoregressive generation
            use_cache: Whether to return KV cache

        Returns:
            Model outputs with logits and optional cache
        """
        if not self._model_initialized:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Prepare batch
        batch = [input_ids]

        # Initialize KV cache if needed
        kv_cache_list = None
        if use_cache and past_key_values is not None:
            kv_cache_list = [[[], []] for _ in range(len(self.layer_names))]

        # Layer-wise inference with async prefetching
        with torch.inference_mode():
            if self.prefetching:
                return self._forward_with_prefetching(
                    batch,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    kv_cache_list,
                )
            else:
                return self._forward_sequential(
                    batch,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    kv_cache_list,
                )

    def _forward_sequential(
        self,
        batch: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
        output_attentions: bool,
        kv_cache_list: Optional[List],
    ) -> CausalLMOutputWithPast:
        """Forward pass without prefetching."""
        for i, (layer_name, layer) in enumerate(zip(self.layer_names, self.layers)):
            # Load layer weights
            state_dict = self.load_layer_to_cpu(layer_name)
            moved_layers = self.move_layer_to_device(state_dict)

            # Process each sequence in batch
            for j, seq in enumerate(batch):
                batch[j] = self._run_layer(
                    layer,
                    layer_name,
                    seq,
                    i,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    kv_cache_list,
                )

            # Free layer memory
            self.unload_layer(layer)

        return self._build_output(batch[0], kv_cache_list, use_cache)

    def _forward_with_prefetching(
        self,
        batch: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
        output_attentions: bool,
        kv_cache_list: Optional[List],
    ) -> CausalLMOutputWithPast:
        """Forward pass with async layer prefetching."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            # Pre-load first layer
            future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in enumerate(zip(self.layer_names, self.layers)):
                # Wait for current layer to load
                state_dict = future.result()
                moved_layers = self.move_layer_to_device(state_dict)

                # Process each sequence in batch
                for j, seq in enumerate(batch):
                    batch[j] = self._run_layer(
                        layer,
                        layer_name,
                        seq,
                        i,
                        attention_mask,
                        past_key_values,
                        use_cache,
                        output_attentions,
                        kv_cache_list,
                    )

                # Kick off next layer loading while computing
                if (i + 1) < len(self.layer_names):
                    future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])

                # Free layer memory
                self.unload_layer(layer)

        return self._build_output(batch[0], kv_cache_list, use_cache)

    def _run_layer(
        self,
        layer: nn.Module,
        layer_name: str,
        seq: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
        output_attentions: bool,
        kv_cache_list: Optional[List],
    ) -> torch.Tensor:
        """Execute single layer computation."""
        # Embedding layer
        if layer_name == self.layer_names_dict.get("embed", ""):
            return layer(seq)

        # Final norm layer
        if layer_name == self.layer_names_dict.get("norm", ""):
            return layer(seq)

        # LM head
        if layer_name == self.layer_names_dict.get("lm_head", ""):
            return layer(seq)

        # Transformer layer
        kwargs = {"use_cache": use_cache}

        if past_key_values is not None and layer_idx > 0:
            # Use cached KV for autoregressive generation
            k_cache, v_cache = past_key_values[layer_idx - 1]
            kwargs["past_key_value"] = (k_cache, v_cache)

            # Adjust attention mask for cached tokens
            if attention_mask is not None:
                cached_len = k_cache.shape[-2]
                kwargs["attention_mask"] = attention_mask[
                    ..., cached_len : cached_len + seq.shape[-1]
                ]
        else:
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

        # Run layer
        layer_outputs = layer(seq, **kwargs)
        new_seq = layer_outputs[0]

        # Extract and store KV cache
        if use_cache and len(layer_outputs) > 1:
            cache_idx = 2 if output_attentions else 1
            if len(layer_outputs) > cache_idx:
                k_cache, v_cache = layer_outputs[cache_idx]
                if kv_cache_list is not None:
                    kv_cache_list[layer_idx][0].append(k_cache)
                    kv_cache_list[layer_idx][1].append(v_cache)

        return new_seq

    def _build_output(
        self, hidden_states: torch.Tensor, kv_cache_list: Optional[List], use_cache: bool
    ) -> CausalLMOutputWithPast:
        """Build model output from hidden states."""
        # Get logits from LM head output (last layer)
        logits = hidden_states

        # Build past_key_values
        past_key_values = None
        if use_cache and kv_cache_list is not None:
            past_key_values = []
            for k_list, v_list in kv_cache_list:
                if k_list and v_list:
                    past_key_values.append((torch.cat(k_list, dim=-2), torch.cat(v_list, dim=-2)))

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            use_cache: Use KV cache for efficiency

        Returns:
            Generated token IDs
        """
        if not self._model_initialized:
            raise RuntimeError("Model not initialized")

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize past_key_values
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            # Get logits for next token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Update past_key_values
            past_key_values = outputs.past_key_values

        return input_ids

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
        **kwargs,
    ) -> "LayerWiseInferenceEngine":
        """
        Load model from pretrained checkpoint.

        Args:
            model_name: HuggingFace model ID or local path
            device: Target device
            dtype: Data type (float16/bfloat16/float32)
            max_memory: Memory limits per device
            prefetching: Enable async prefetching
            compression: Quantization mode (4bit/8bit)
            local_path: Local path to model files

        Returns:
            Initialized inference engine
        """
        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        # Create instance
        instance = cls(
            model_name=model_name,
            device=device,
            dtype=torch_dtype,
            max_memory=max_memory,
            prefetching=prefetching,
            compression=compression,
            local_path=local_path,
        )

        # Initialize
        instance.init_tokenizer()
        instance.init_model()
        instance.setup_cuda_stream()

        return instance
