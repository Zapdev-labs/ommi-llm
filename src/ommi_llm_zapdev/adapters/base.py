"""
Base adapter interface for model architectures.
"""

from abc import abstractmethod
from typing import Any, Dict

from ..core.engine import LayerWiseInferenceEngine


class ModelAdapter(LayerWiseInferenceEngine):
    """
    Base class for model-specific adapters.

    Each adapter implements architecture-specific layer naming and
    configuration for the layer-wise inference engine.

    Subclasses must implement:
    - set_layer_names_dict(): Define layer name mappings

    Example:
        >>> class MyModelAdapter(ModelAdapter):
        ...     def set_layer_names_dict(self) -> None:
        ...         self.layer_names_dict = {
        ...             'embed': 'model.embed_tokens',
        ...             'layer_prefix': 'model.layers',
        ...             'norm': 'model.norm',
        ...             'lm_head': 'lm_head',
        ...         }
    """

    @abstractmethod
    def set_layer_names_dict(self) -> None:
        """
        Set layer name mappings for this architecture.

        Must define:
        - 'embed': Embedding layer path
        - 'layer_prefix': Prefix for transformer layers
        - 'norm': Final normalization layer path
        - 'lm_head': Language modeling head path
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "architecture": self.__class__.__name__,
            "num_layers": len(self.layers) if self.layers else 0,
            "layer_names": self.layer_names,
            "device": self.device,
            "dtype": str(self.running_dtype),
            "prefetching": self.prefetching,
            "compression": self.compression,
        }
