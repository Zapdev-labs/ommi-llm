"""
Qwen adapter.
"""

from .base import ModelAdapter


class QwenAdapter(ModelAdapter):
    """
    Adapter for Qwen and Qwen2 models.

    Qwen models use a similar architecture to Llama but with
    different layer naming conventions.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i}
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure Qwen layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
