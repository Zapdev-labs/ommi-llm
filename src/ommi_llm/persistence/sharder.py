"""
Model sharding utilities for layer-wise storage.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from ..utils.memory import clean_memory

logger = logging.getLogger(__name__)


class ModelSharder:
    """
    Shards models into layer-wise SafeTensors files.

    This enables efficient layer-by-layer loading for inference.
    """

    def __init__(self, model_name: str, output_path: Path, compression: Optional[str] = None):
        self.model_name = model_name
        self.output_path = Path(output_path)
        self.compression = compression

        self.output_path.mkdir(parents=True, exist_ok=True)

    def shard_model(self, delete_original: bool = False, skip_if_exists: bool = True) -> List[Path]:
        """
        Shard model into layer-wise files.

        Args:
            delete_original: Delete original checkpoint after sharding
            skip_if_exists: Skip if output files already exist

        Returns:
            List of created shard file paths
        """
        # Load config
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        # Determine layer structure
        layer_names = self._get_layer_names(config)

        # Check if already sharded
        if skip_if_exists:
            all_exist = all(
                (self.output_path / f"{name}.safetensors").exists() for name in layer_names
            )
            if all_exist:
                logger.info("Model already sharded, skipping")
                return [self.output_path / f"{name}.safetensors" for name in layer_names]

        # Load original checkpoint
        logger.info(f"Loading model weights from: {self.model_name}")
        state_dict = self._load_checkpoint()

        # Shard by layer
        shard_paths = []
        for layer_name in tqdm(layer_names, desc="Sharding model"):
            shard_path = self._extract_and_save_layer(state_dict, layer_name)
            shard_paths.append(shard_path)

        logger.info(f"Model sharded into {len(shard_paths)} layers")

        # Clean up
        del state_dict
        clean_memory()

        if delete_original:
            self._delete_original_checkpoint()

        return shard_paths

    def _get_layer_names(self, config) -> List[str]:
        """Generate layer names from config."""
        layer_names = []

        # Embedding
        layer_names.append("model.embed_tokens")

        # Transformer layers
        num_layers = getattr(
            config,
            "num_hidden_layers",
            getattr(config, "n_layer", getattr(config, "num_layers", 0)),
        )

        for i in range(num_layers):
            layer_names.append(f"model.layers.{i}")

        # Final norm
        layer_names.append("model.norm")

        # LM head
        layer_names.append("lm_head")

        return layer_names

    def _load_checkpoint(self) -> Dict[str, torch.Tensor]:
        """Load original model checkpoint."""
        # Try loading from index file first
        index_file = Path(self.model_name) / "model.safetensors.index.json"

        if index_file.exists():
            import json

            with open(index_file) as f:
                index = json.load(f)

            # Load all referenced files
            state_dict = {}
            weight_map = index.get("weight_map", {})
            loaded_files = set()

            for param_name, file_name in weight_map.items():
                if file_name in loaded_files:
                    continue

                file_path = Path(self.model_name) / file_name
                file_state_dict = load_file(str(file_path), device="cpu")
                state_dict.update(file_state_dict)
                loaded_files.add(file_name)

            return state_dict

        # Try single file
        single_file = Path(self.model_name) / "model.safetensors"
        if single_file.exists():
            return load_file(str(single_file), device="cpu")

        # Try PyTorch format
        pytorch_file = Path(self.model_name) / "pytorch_model.bin"
        if pytorch_file.exists():
            return torch.load(str(pytorch_file), map_location="cpu")

        # Load via transformers and extract state dict
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )
        return model.state_dict()

    def _extract_and_save_layer(self, state_dict: Dict[str, torch.Tensor], layer_name: str) -> Path:
        """Extract layer weights and save to file."""
        # Extract weights for this layer
        layer_weights = {k: v for k, v in state_dict.items() if k.startswith(layer_name)}

        if not layer_weights:
            logger.warning(f"No weights found for layer: {layer_name}")

        # Apply compression if specified
        if self.compression:
            layer_weights = self._compress_weights(layer_weights)

        # Save to file
        output_file = self.output_path / f"{layer_name}.safetensors"
        save_file(layer_weights, str(output_file))

        return output_file

    def _compress_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress weights using quantization."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.warning("bitsandbytes not installed, skipping compression")
            return weights

        compressed = {}

        for name, tensor in weights.items():
            if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                compressed[name] = tensor
                continue

            if self.compression == "4bit":
                # NF4 quantization
                tensor_quant, quant_state = bnb.functional.quantize_nf4(tensor.cuda(), blocksize=64)
                compressed[name] = tensor_quant.cpu()

                # Save quantization state
                for qs_key, qs_val in quant_state.items():
                    compressed[f"{name}.4bit.{qs_key}"] = qs_val.cpu()

            elif self.compression == "8bit":
                # Block-wise 8-bit quantization
                tensor_quant, quant_state = bnb.functional.quantize_blockwise(
                    tensor.cuda(), blocksize=2048
                )
                compressed[name] = tensor_quant.cpu()

                # Save quantization state
                for qs_key, qs_val in quant_state.items():
                    compressed[f"{name}.8bit.{qs_key}"] = qs_val.cpu()

            else:
                compressed[name] = tensor

        return compressed

    def _delete_original_checkpoint(self) -> None:
        """Delete original checkpoint files."""
        logger.warning("Deleting original checkpoint - this cannot be undone!")
        # Implementation would go here
        pass
