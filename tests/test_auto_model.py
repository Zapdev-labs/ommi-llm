"""
Tests for auto model loading and architecture detection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ommi_llm.core.auto_model import AutoModel, ARCHITECTURE_REGISTRY, MODEL_TYPE_PATTERNS
from ommi_llm.adapters.llama import LlamaAdapter
from ommi_llm.adapters.qwen import QwenAdapter
from ommi_llm.adapters.mistral import MistralAdapter
from ommi_llm.adapters.mixtral import MixtralAdapter
from ommi_llm.adapters.generic import GenericAdapter


class TestArchitectureRegistry:
    """Test architecture registry contents."""
    
    def test_llama_architectures_registered(self):
        """Test Llama architectures are registered."""
        assert "LlamaForCausalLM" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["LlamaForCausalLM"] == LlamaAdapter
        assert "LlamaModel" in ARCHITECTURE_REGISTRY
    
    def test_qwen_architectures_registered(self):
        """Test Qwen architectures are registered."""
        assert "Qwen2ForCausalLM" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["Qwen2ForCausalLM"] == QwenAdapter
        assert "QwenForCausalLM" in ARCHITECTURE_REGISTRY
        assert "Qwen3ForCausalLM" in ARCHITECTURE_REGISTRY
        assert "Qwen3_5ForCausalLM" in ARCHITECTURE_REGISTRY
    
    def test_mistral_architectures_registered(self):
        """Test Mistral architectures are registered."""
        assert "MistralForCausalLM" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["MistralForCausalLM"] == MistralAdapter
        assert "MistralModel" in ARCHITECTURE_REGISTRY
    
    def test_mixtral_architectures_registered(self):
        """Test Mixtral architectures are registered."""
        assert "MixtralForCausalLM" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["MixtralForCausalLM"] == MixtralAdapter
    
    def test_gemma_uses_llama_adapter(self):
        """Test Gemma models use LlamaAdapter."""
        assert ARCHITECTURE_REGISTRY["GemmaForCausalLM"] == LlamaAdapter
        assert ARCHITECTURE_REGISTRY["Gemma2ForCausalLM"] == LlamaAdapter
    
    def test_phi_uses_llama_adapter(self):
        """Test Phi models use LlamaAdapter."""
        assert ARCHITECTURE_REGISTRY["PhiForCausalLM"] == LlamaAdapter
        assert ARCHITECTURE_REGISTRY["Phi3ForCausalLM"] == LlamaAdapter


class TestModelTypePatterns:
    """Test model type patterns."""
    
    def test_core_model_types(self):
        """Test core model types are mapped."""
        assert "llama" in MODEL_TYPE_PATTERNS
        assert MODEL_TYPE_PATTERNS["llama"] == LlamaAdapter
        assert "mistral" in MODEL_TYPE_PATTERNS
        assert "qwen" in MODEL_TYPE_PATTERNS
        assert MODEL_TYPE_PATTERNS["qwen"] == QwenAdapter
    
    def test_extended_model_types(self):
        """Test extended model types."""
        extended = ["gemma", "phi", "falcon", "deepseek", "yi"]
        for model_type in extended:
            assert model_type in MODEL_TYPE_PATTERNS
            assert MODEL_TYPE_PATTERNS[model_type] == LlamaAdapter


class TestAutoModel:
    """Test AutoModel factory class."""
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_exact_architecture_match(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test loading with exact architecture match."""
        mock_config = Mock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        result = AutoModel.from_pretrained("meta-llama/Llama-2-7b")
        
        assert result is mock_adapter
        mock_from_pretrained.assert_called_once()
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.qwen.QwenAdapter.from_pretrained")
    def test_from_pretrained_model_type_match(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test loading with model_type pattern match."""
        mock_config = Mock()
        mock_config.architectures = []  # No exact architecture
        mock_config.model_type = "qwen"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        result = AutoModel.from_pretrained("Qwen/Qwen-7B")
        
        assert result is mock_adapter
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_llama_like_pattern(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test loading with llama-like pattern detection."""
        mock_config = Mock()
        mock_config.architectures = ["NewModelForCausalLM"]
        mock_config.model_type = "new_model"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        result = AutoModel.from_pretrained("org/new-model")
        
        # Should use LlamaAdapter for unknown causal LM
        assert result is mock_adapter
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.generic.GenericAdapter.from_pretrained")
    def test_from_pretrained_generic_fallback(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test fallback to GenericAdapter for unknown architectures."""
        mock_config = Mock()
        mock_config.architectures = ["UnknownForCausalLM"]
        mock_config.model_type = "unknown"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        result = AutoModel.from_pretrained("unknown/model")
        
        assert result is mock_adapter
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_no_architecture_raises(self, mock_from_config):
        """Test error when no architecture or model_type found."""
        mock_config = Mock()
        mock_config.architectures = []
        mock_config.model_type = ""
        mock_from_config.return_value = mock_config
        
        with pytest.raises(ValueError, match="No architecture or model_type"):
            AutoModel.from_pretrained("unknown/model")
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_with_local_path(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test loading with local path."""
        mock_config = Mock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        result = AutoModel.from_pretrained(
            "local-model",
            local_path="/path/to/model"
        )
        
        assert result is mock_adapter
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_dtype_parsing(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test dtype parameter is passed correctly."""
        mock_config = Mock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        AutoModel.from_pretrained("llama-model", dtype="bfloat16")
        
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["dtype"] == "bfloat16"
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_device_parameter(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test device parameter is passed correctly."""
        mock_config = Mock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        AutoModel.from_pretrained("llama-model", device="cuda:1")
        
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["device"] == "cuda:1"
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("ommi_llm.adapters.llama.LlamaAdapter.from_pretrained")
    def test_from_pretrained_compression_parameter(
        self, mock_from_pretrained, mock_from_config
    ):
        """Test compression parameter is passed correctly."""
        mock_config = Mock()
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        mock_adapter = Mock()
        mock_from_pretrained.return_value = mock_adapter
        
        AutoModel.from_pretrained("llama-model", compression="4bit")
        
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["compression"] == "4bit"


class TestAutoModelRegistryMethods:
    """Test AutoModel registry management methods."""
    
    def test_list_supported_architectures(self):
        """Test listing supported architectures."""
        architectures = AutoModel.list_supported_architectures()
        
        assert isinstance(architectures, dict)
        assert "LlamaForCausalLM" in architectures
        assert architectures["LlamaForCausalLM"] == "LlamaAdapter"
    
    def test_list_supported_model_types(self):
        """Test listing supported model types."""
        model_types = AutoModel.list_supported_model_types()
        
        assert isinstance(model_types, dict)
        assert "llama" in model_types
        assert model_types["llama"] == "LlamaAdapter"
    
    def test_is_architecture_supported_true(self):
        """Test checking supported architecture."""
        assert AutoModel.is_architecture_supported("LlamaForCausalLM") is True
        assert AutoModel.is_architecture_supported("Qwen2ForCausalLM") is True
    
    def test_is_architecture_supported_false(self):
        """Test checking unsupported architecture."""
        assert AutoModel.is_architecture_supported("UnknownForCausalLM") is False
        assert AutoModel.is_architecture_supported("") is False
    
    def test_register_adapter(self):
        """Test registering custom adapter."""
        class CustomAdapter(Mock):
            pass
        
        AutoModel.register_adapter("CustomForCausalLM", CustomAdapter)
        
        assert "CustomForCausalLM" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["CustomForCausalLM"] == CustomAdapter
    
    def test_register_model_type(self):
        """Test registering model type pattern."""
        class CustomAdapter(Mock):
            pass
        
        AutoModel.register_model_type("custom", CustomAdapter)
        
        assert "custom" in MODEL_TYPE_PATTERNS
        assert MODEL_TYPE_PATTERNS["custom"] == CustomAdapter


class TestCanLoadWithGeneric:
    """Test GenericAdapter compatibility check."""
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_can_load_causal_lm(self, mock_from_config):
        """Test that causal LM can be loaded with generic."""
        mock_config = Mock()
        mock_config.architectures = ["SomeForCausalLM"]
        mock_config.model_type = "some_decoder"
        mock_from_config.return_value = mock_config
        
        result = AutoModel.can_load_with_generic("some/model")
        
        assert result is True
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_can_load_decoder_model_type(self, mock_from_config):
        """Test that decoder model type can be loaded."""
        mock_config = Mock()
        mock_config.architectures = []
        mock_config.model_type = "decoder_only"
        mock_from_config.return_value = mock_config
        
        result = AutoModel.can_load_with_generic("some/model")
        
        assert result is True
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_cannot_load_non_causal(self, mock_from_config):
        """Test that non-causal models can't be loaded."""
        mock_config = Mock()
        mock_config.architectures = ["SomeForMaskedLM"]
        mock_config.model_type = "encoder_only"
        mock_from_config.return_value = mock_config
        
        result = AutoModel.can_load_with_generic("some/model")
        
        assert result is False
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_can_load_with_generic_error_handling(self, mock_from_config):
        """Test error handling in can_load_with_generic."""
        mock_from_config.side_effect = Exception("Config not found")
        
        result = AutoModel.can_load_with_generic("some/model")
        
        assert result is False
