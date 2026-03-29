"""
Llama/Llama2/Llama3 adapter.
"""

from .base import ModelAdapter


class LlamaAdapter(ModelAdapter):
    """
    Adapter for Llama, Llama 2, and Llama 3 models.

    Supports all Llama family models from 7B to 405B parameters.

    Architecture:
    - Embedding: model.embed_tokens
    - Layers: model.layers.{i}
    - Norm: model.norm
    - LM Head: lm_head
    """

    def set_layer_names_dict(self) -> None:
        """Configure Llama layer names."""
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
