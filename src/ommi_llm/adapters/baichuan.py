"""
Baichuan adapter.
"""

from .base import ModelAdapter


class BaichuanAdapter(ModelAdapter):
    """
    Adapter for Baichuan models.

    Baichuan models support both Chinese and English with
    specialized tokenization and architecture.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i}
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure Baichuan layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
