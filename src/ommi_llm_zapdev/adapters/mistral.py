"""
Mistral adapter.
"""

from .base import ModelAdapter


class MistralAdapter(ModelAdapter):
    """
    Adapter for Mistral models.

    Mistral uses the same architecture as Llama with
    grouped query attention (GQA) and sliding window attention.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i}
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure Mistral layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
