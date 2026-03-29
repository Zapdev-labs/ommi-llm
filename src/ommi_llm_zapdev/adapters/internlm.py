"""
InternLM adapter.
"""

from .base import ModelAdapter


class InternLMAdapter(ModelAdapter):
    """
    Adapter for InternLM models.

    InternLM uses a similar architecture to Llama.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i}
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure InternLM layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
