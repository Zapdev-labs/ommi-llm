"""
Memory management utilities.
"""

import ctypes
import gc
import logging
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory for layer-wise inference.

    Tracks VRAM usage, performs cleanup, and optimizes memory
    allocation for minimal footprint.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._peak_memory = 0

    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics.

        Returns:
            Dictionary with RAM and VRAM statistics
        """
        stats = {
            "ram": {
                "total": psutil.virtual_memory().total / (1024**3),
                "available": psutil.virtual_memory().available / (1024**3),
                "percent": psutil.virtual_memory().percent,
            }
        }

        if self.device.startswith("cuda") and torch.cuda.is_available():
            stats["vram"] = {
                "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "allocated": torch.cuda.memory_allocated() / (1024**3),
                "reserved": torch.cuda.memory_reserved() / (1024**3),
                "free": (
                    torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                )
                / (1024**3),
            }

        return stats

    def clean_memory(self) -> None:
        """
        Clean up RAM and VRAM.

        Forces garbage collection and clears CUDA cache.
        """
        gc.collect()

        # Force memory release in glibc (Linux only)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_memory_stats(self, prefix: str = "") -> None:
        """Log current memory statistics."""
        stats = self.get_memory_stats()

        ram = stats["ram"]
        msg = f"{prefix}RAM: {ram['available']:.2f}GB available / {ram['total']:.2f}GB total"

        if "vram" in stats:
            vram = stats["vram"]
            msg += f" | VRAM: {vram['allocated']:.2f}GB allocated / {vram['total']:.2f}GB total"

        logger.info(msg)

    def estimate_peak_memory(
        self, num_layers: int, hidden_size: int, batch_size: int = 1, seq_length: int = 1024
    ) -> float:
        """
        Estimate peak memory usage for inference.

        Args:
            num_layers: Number of transformer layers
            hidden_size: Hidden dimension size
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Estimated peak memory in GB
        """
        # Layer weights (assuming FP16, ~1.6GB per layer for 70B model)
        layer_size = hidden_size * hidden_size * 12 * 2 / (1024**3)  # GB

        # Activations (batch_size * seq_length * hidden_size * 2 bytes)
        activation_size = batch_size * seq_length * hidden_size * 2 / (1024**3)

        # KV cache (2 * num_layers * batch_size * seq_length * hidden_size * 2 bytes)
        kv_cache_size = 2 * num_layers * batch_size * seq_length * hidden_size * 2 / (1024**3)

        # Peak = single layer + activations + small KV cache portion
        peak = layer_size + activation_size + (kv_cache_size / num_layers)

        return peak


def clean_memory() -> None:
    """Global memory cleanup function."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
