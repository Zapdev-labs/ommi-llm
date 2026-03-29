"""
Mixtral MoE adapter.
"""

from .base import ModelAdapter


class MixtralAdapter(ModelAdapter):
    """
    Adapter for Mixtral 8x7B and 8x22B MoE models.

    Mixtral uses sparse mixture of experts with 8 experts per layer,
    2 active per token. Architecture is similar to Mistral.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i} (with block_sparse_moe)
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure Mixtral layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
