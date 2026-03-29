"""
Generic adapter with automatic layer name detection.

This adapter can handle ANY transformer architecture by analyzing
the model structure and auto-detecting the appropriate layer paths.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import ModelAdapter

logger = logging.getLogger(__name__)


class GenericAdapter(ModelAdapter):
    """
    Generic adapter that auto-detects layer names for any architecture.

    This adapter analyzes the model structure at initialization to determine:
    - Embedding layer path
    - Transformer layer prefix
    - Final normalization layer path
    - LM head path

    Supports virtually all transformer-based language models including:
    - Llama family (all versions)
    - Mistral/Mixtral
    - Qwen family (Qwen, Qwen2, Qwen3, Qwen3.5)
    - GPT/GPT-2/GPT-Neo/GPT-J
    - Falcon
    - Gemma
    - Phi family
    - Minimax
    - Command-R
    - DBRX
    - And many more...

    Architecture Detection Strategy:
        1. Try known patterns for common architectures
        2. If no match, analyze model structure:
           - Find embedding by looking for token/word embedding layers
           - Find layers by looking for transformer blocks (self-attention patterns)
           - Find norm by looking for final LayerNorm/RMSNorm
           - Find lm_head by looking for linear projection to vocab size
    """

    # Known layer naming patterns for various architectures
    # These are tried in order before auto-detection
    KNOWN_PATTERNS = {
        # Llama family (Llama, Llama2, Llama3, CodeLlama, etc.)
        "llama": {
            "embed": ["model.embed_tokens", "embed_tokens"],
            "layer_prefix": ["model.layers", "model.decoder.layers", "layers"],
            "norm": ["model.norm", "model.decoder.norm", "norm", "ln_f"],
            "lm_head": ["lm_head", "embed_out", "output"],
        },
        # GPT-2/GPT-Neo/GPT-J family
        "gpt": {
            "embed": ["transformer.wte", "transformer.embed_tokens", "wte"],
            "layer_prefix": ["transformer.h", "transformer.layers", "h"],
            "norm": ["transformer.ln_f", "transformer.norm", "ln_f"],
            "lm_head": ["lm_head"],
        },
        # T5 family
        "t5": {
            "embed": ["encoder.embed_tokens", "shared"],
            "layer_prefix": ["encoder.block", "decoder.block"],
            "norm": ["encoder.final_layer_norm", "decoder.final_layer_norm"],
            "lm_head": ["lm_head"],
        },
        # BLOOM
        "bloom": {
            "embed": ["transformer.word_embeddings", "word_embeddings"],
            "layer_prefix": ["transformer.h", "h"],
            "norm": ["transformer.ln_f", "ln_f"],
            "lm_head": ["lm_head"],
        },
        # Falcon
        "falcon": {
            "embed": ["transformer.word_embeddings", "word_embeddings"],
            "layer_prefix": ["transformer.h", "transformer.layers", "h"],
            "norm": ["transformer.ln_f", "ln_f"],
            "lm_head": ["lm_head"],
        },
        # ChatGLM family
        "chatglm": {
            "embed": ["transformer.embedding.word_embeddings"],
            "layer_prefix": ["transformer.encoder.layers"],
            "norm": ["transformer.encoder.final_layernorm"],
            "lm_head": ["transformer.output_layer"],
        },
        # MPT (MosaicML)
        "mpt": {
            "embed": ["transformer.wte", "wte"],
            "layer_prefix": ["transformer.blocks", "blocks"],
            "norm": ["transformer.norm_f", "norm_f"],
            "lm_head": ["lm_head"],
        },
        # OPT (Facebook)
        "opt": {
            "embed": ["model.decoder.embed_tokens", "decoder.embed_tokens"],
            "layer_prefix": ["model.decoder.layers", "decoder.layers"],
            "norm": ["model.decoder.final_layer_norm", "decoder.final_layer_norm"],
            "lm_head": ["lm_head"],
        },
        # Command-R / Cohere
        "command_r": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # Gemma family
        "gemma": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # Phi family (Phi, Phi-2, Phi-3)
        "phi": {
            "embed": ["model.embed_tokens", "embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm", "ln_f"],
            "lm_head": ["lm_head"],
        },
        # Minimax
        "minimax": {
            "embed": ["model.embed_tokens", "embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # DBRX
        "dbrx": {
            "embed": ["transformer.wte"],
            "layer_prefix": ["transformer.blocks"],
            "norm": ["transformer.norm_f"],
            "lm_head": ["lm_head"],
        },
        # StableLM
        "stablelm": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # Pythia/Polyglot
        "pythia": {
            "embed": ["gpt_neox.embed_in", "embed_in"],
            "layer_prefix": ["gpt_neox.layers", "layers"],
            "norm": ["gpt_neox.final_layer_norm", "final_layer_norm"],
            "lm_head": ["embed_out"],
        },
        # Qwen3 family (including Qwen3.5)
        "qwen3": {
            "embed": ["model.embed_tokens", "embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # Yi
        "yi": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # DeepSeek
        "deepseek": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # StarCoder
        "starcoder": {
            "embed": ["transformer.wte", "wte"],
            "layer_prefix": ["transformer.h", "h"],
            "norm": ["transformer.ln_f", "ln_f"],
            "lm_head": ["lm_head"],
        },
        # CodeLlama/CodeQwen/CodeGemma use llama patterns
        "codellama": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # Jamba (hybrid)
        "jamba": {
            "embed": ["model.embed_tokens"],
            "layer_prefix": ["model.layers"],
            "norm": ["model.norm"],
            "lm_head": ["lm_head"],
        },
        # OLMo
        "olmo": {
            "embed": ["model.transformer.wte"],
            "layer_prefix": ["model.transformer.blocks"],
            "norm": ["model.transformer.ln_f"],
            "lm_head": ["model.transformer.ff_out"],
        },
    }

    def set_layer_names_dict(self) -> None:
        """
        Auto-detect layer names for the model architecture.

        Strategy:
            1. Get model type from config
            2. Try known patterns for this model type
            3. If no match, perform structural analysis
            4. Store the detected layer names
        """
        # First, try to detect from config
        model_type = getattr(self.config, "model_type", "").lower()
        architectures = getattr(self.config, "architectures", [])

        # Try known patterns first
        layer_names = self._try_known_patterns(model_type, architectures)

        if layer_names:
            logger.info(f"Using known pattern for {model_type or architectures[0]}")
            self.layer_names_dict = layer_names
            return

        # If no known pattern matched, perform structural analysis
        logger.info(f"No known pattern for {model_type}, performing structural analysis...")
        layer_names = self._analyze_model_structure()

        if not layer_names:
            raise ValueError(
                f"Could not auto-detect layer names for model type '{model_type}'. "
                f"Model architecture: {architectures}. "
                f"Please report this model type for explicit support."
            )

        logger.info(f"Auto-detected layer names for {model_type}: {layer_names}")
        self.layer_names_dict = layer_names

    def _try_known_patterns(
        self, model_type: str, architectures: List[str]
    ) -> Optional[Dict[str, str]]:
        """
        Try to match model against known layer naming patterns.

        Args:
            model_type: The model_type from config
            architectures: List of architecture classes

        Returns:
            Layer names dict if matched, None otherwise
        """
        # Check model_type patterns
        for pattern_name, patterns in self.KNOWN_PATTERNS.items():
            if pattern_name in model_type or model_type in pattern_name:
                result = self._test_pattern(patterns)
                if result:
                    return result

        # Check architecture patterns
        for arch in architectures:
            arch_lower = arch.lower()
            for pattern_name, patterns in self.KNOWN_PATTERNS.items():
                if pattern_name in arch_lower:
                    result = self._test_pattern(patterns)
                    if result:
                        return result

        # Special case: if architecture contains "ForCausalLM", try llama pattern
        for arch in architectures:
            if "forcausallm" in arch.lower() or "forconditionalgeneration" in arch.lower():
                # Most modern models use llama-like patterns
                result = self._test_pattern(self.KNOWN_PATTERNS["llama"])
                if result:
                    logger.info(f"Using llama-like pattern for {arch}")
                    return result

        return None

    def _test_pattern(self, patterns: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
        """
        Test if a pattern works for the current model.

        Args:
            patterns: Dictionary of possible paths for each component

        Returns:
            Working layer names dict, or None if pattern doesn't match
        """
        result = {}

        # Test embed
        for embed_path in patterns.get("embed", []):
            if self._path_exists(embed_path):
                result["embed"] = embed_path
                break

        # Test layer_prefix
        for layer_prefix in patterns.get("layer_prefix", []):
            if self._get_layer_count(layer_prefix) > 0:
                result["layer_prefix"] = layer_prefix
                break

        # Test norm
        for norm_path in patterns.get("norm", []):
            if self._path_exists(norm_path):
                result["norm"] = norm_path
                break

        # Test lm_head
        for lm_head_path in patterns.get("lm_head", []):
            if self._path_exists(lm_head_path):
                result["lm_head"] = lm_head_path
                break

        # Return only if we found at least layer_prefix
        if "layer_prefix" in result:
            return result

        return None

    def _analyze_model_structure(self) -> Dict[str, str]:
        """
        Perform structural analysis to detect layer names.

        This is the fallback when no known patterns match.
        It examines the model's module structure to identify:
        - Embedding layer
        - Transformer layers
        - Final normalization
        - Output head

        Returns:
            Dictionary with detected layer paths
        """
        if not self.model:
            raise RuntimeError("Model not initialized")

        layer_names = {}

        # Get all module names
        all_modules = dict(self.model.named_modules())
        all_module_names = list(all_modules.keys())

        # 1. Find embedding layer
        embed_candidates = [
            "embed_tokens",
            "wte",
            "word_embeddings",
            "embed_in",
            "embedding",
            "token_embedding",
            "embed",
        ]
        layer_names["embed"] = self._find_best_match(all_module_names, embed_candidates)

        # 2. Find transformer layers
        layer_names["layer_prefix"] = self._detect_layer_prefix(all_modules)

        # 3. Find final normalization
        norm_candidates = ["norm", "ln_f", "final_layer_norm", "norm_f", "layer_norm", "final_norm"]
        layer_names["norm"] = self._find_best_match(all_module_names, norm_candidates)

        # 4. Find LM head
        head_candidates = [
            "lm_head",
            "embed_out",
            "output",
            "head",
            "logits",
            "classifier",
            "score",
        ]
        layer_names["lm_head"] = self._find_best_match(all_module_names, head_candidates)

        return layer_names

    def _find_best_match(self, all_names: List[str], candidates: List[str]) -> Optional[str]:
        """
        Find the best matching module name from candidates.

        Args:
            all_names: List of all module names in the model
            candidates: List of candidate names to search for

        Returns:
            Best matching name, or None if no match
        """
        for name in all_names:
            name_lower = name.lower()
            for candidate in candidates:
                if candidate in name_lower:
                    # Prefer exact matches or simple paths
                    if name == candidate or name.endswith(f".{candidate}"):
                        return name

        # If no good match, try partial matching
        for name in all_names:
            name_lower = name.lower()
            for candidate in candidates:
                if candidate in name_lower:
                    return name

        return None

    def _detect_layer_prefix(self, all_modules: Dict[str, nn.Module]) -> Optional[str]:
        """
        Detect the transformer layer prefix by looking for sequential patterns.

        Args:
            all_modules: Dictionary of all modules in the model

        Returns:
            Layer prefix path, or None if not detected
        """
        # Look for patterns like layers.0, layers.1, h.0, blocks.0, etc.
        import re

        # Pattern to find sequential modules
        sequential_patterns = [
            r"(.+)\.layers?\.(\d+)",
            r"(.+)\.h\.(\d+)",
            r"(.+)\.blocks?\.(\d+)",
            r"(.+)\.encoder\.layers?\.(\d+)",
            r"(.+)\.decoder\.layers?\.(\d+)",
        ]

        # Collect all matches
        matches = {}
        for name in all_modules.keys():
            for pattern in sequential_patterns:
                match = re.match(pattern, name)
                if match:
                    prefix = match.group(1)
                    idx = int(match.group(2))
                    if prefix not in matches:
                        matches[prefix] = []
                    matches[prefix].append(idx)

        # Find the prefix with the most sequential layers
        best_prefix = None
        best_count = 0

        for prefix, indices in matches.items():
            # Check if indices are sequential (0, 1, 2, ...)
            if len(indices) > best_count:
                sorted_indices = sorted(set(indices))
                # Verify it's a proper sequence starting from 0
                if sorted_indices[0] == 0:
                    best_prefix = prefix
                    best_count = len(indices)

        if best_prefix:
            # Return the full prefix path
            return f"{best_prefix}.layers" if "layers" not in best_prefix else best_prefix

        return None

    def _path_exists(self, path: str) -> bool:
        """
        Check if a module path exists in the model.

        Args:
            path: Dot-separated path like "model.layers"

        Returns:
            True if path exists, False otherwise
        """
        if not self.model:
            return False

        parts = path.split(".")
        obj = self.model

        for part in parts:
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)

        return True

    def _get_layer_count(self, prefix: str) -> int:
        """
        Get the number of layers at a given prefix.

        Args:
            prefix: Layer prefix path

        Returns:
            Number of layers, or 0 if prefix doesn't exist
        """
        if not self.model:
            return 0

        parts = prefix.split(".")

        # Navigate to the container
        obj = self.model
        for part in parts[:-1]:  # Exclude the last part (layers, h, blocks, etc.)
            if not hasattr(obj, part):
                return 0
            obj = getattr(obj, part)

        # The last part should be a ModuleList or similar
        last_part = parts[-1]
        if not hasattr(obj, last_part):
            return 0

        container = getattr(obj, last_part)

        # Check if it's iterable (like ModuleList)
        try:
            return len(container)
        except (TypeError, AttributeError):
            return 0

    def _get_nested_attr(self, obj, path: str):
        """
        Get a nested attribute by dot-separated path.

        Args:
            obj: Object to navigate
            path: Dot-separated path

        Returns:
            The attribute, or None if path doesn't exist
        """
        parts = path.split(".")
        for part in parts:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj
