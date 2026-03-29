"""
ChatGLM adapter.
"""

from .base import ModelAdapter


class ChatGLMAdapter(ModelAdapter):
    """
    Adapter for ChatGLM models.

    ChatGLM uses a different architecture than Llama with
    GLU activation and custom attention patterns.

    Architecture:
    - Embedding: transformer.embedding.word_embeddings
    - Layers: transformer.encoder.layers.{i}
    - Norm: transformer.encoder.final_layernorm
    - LM Head: transformer.output_layer
    """

    def set_layer_names_dict(self) -> None:
        """Configure ChatGLM layer names."""
        self.layer_names_dict = {
            "embed": "transformer.embedding.word_embeddings",
            "layer_prefix": "transformer.encoder.layers",
            "norm": "transformer.encoder.final_layernorm",
            "lm_head": "transformer.output_layer",
        }
