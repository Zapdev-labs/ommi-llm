"""
Model weight loading utilities.
"""

import logging
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def load_layer_weights(checkpoint_path: Path, layer_name: str) -> Dict[str, torch.Tensor]:
    """
    Load layer weights from SafeTensors file.

    Args:
        checkpoint_path: Path to checkpoint directory
        layer_name: Name of the layer to load

    Returns:
        Dictionary of weight tensors
    """
    layer_file = checkpoint_path / f"{layer_name}.safetensors"

    if not layer_file.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_file}")

    logger.debug(f"Loading layer weights from: {layer_file}")

    # Load using SafeTensors (memory efficient)
    state_dict = load_file(str(layer_file), device="cpu")

    return state_dict


def save_layer_weights(
    state_dict: Dict[str, torch.Tensor], layer_name: str, checkpoint_path: Path
) -> None:
    """
    Save layer weights to SafeTensors file.

    Args:
        state_dict: Dictionary of weight tensors
        layer_name: Name of the layer
        checkpoint_path: Path to checkpoint directory
    """
    from safetensors.torch import save_file

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    layer_file = checkpoint_path / f"{layer_name}.safetensors"

    save_file(state_dict, str(layer_file))
    logger.debug(f"Saved layer weights to: {layer_file}")
