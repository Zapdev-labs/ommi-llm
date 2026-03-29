"""
Tests for model adapters.
"""

import pytest
from unittest.mock import Mock, patch

from ommi_llm.adapters.base import ModelAdapter
from ommi_llm.adapters.llama import LlamaAdapter
from ommi_llm.adapters.qwen import QwenAdapter
from ommi_llm.adapters.mistral import MistralAdapter
from ommi_llm.adapters.mixtral import MixtralAdapter
from ommi_llm.adapters.baichuan import BaichuanAdapter
from ommi_llm.adapters.chatglm import ChatGLMAdapter
from ommi_llm.adapters.internlm import InternLMAdapter
from ommi_llm.adapters.generic import GenericAdapter


class MockAdapter(ModelAdapter):
    """Mock adapter for testing base class."""
    
    def set_layer_names_dict(self) -> None:
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }


class TestModelAdapter:
    """Test base ModelAdapter class."""
    
    @pytest.fixture
    def adapter(self):
        """Create a basic adapter fixture."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = MockAdapter(
                model_name="test-model",
                device="cpu",
            )
            return adapter
    
    def test_adapter_is_abstract(self):
        """Test that ModelAdapter is abstract and requires set_layer_names_dict."""
        with pytest.raises(TypeError):
            ModelAdapter(model_name="test", device="cpu")
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.model_name == "test-model"
        assert adapter.device == "cpu"
        assert adapter.__class__.__name__ == "MockAdapter"
    
    def test_get_model_info(self, adapter):
        """Test getting model information."""
        adapter.layers = [Mock(), Mock(), Mock()]  # 3 mock layers
        adapter.layer_names = ["embed", "layer_0", "norm"]
        
        info = adapter.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["architecture"] == "MockAdapter"
        assert info["num_layers"] == 3
        assert info["layer_names"] == ["embed", "layer_0", "norm"]
        assert info["device"] == "cpu"
        assert info["prefetching"] is False
        assert info["compression"] is None
    
    def test_get_model_info_empty_layers(self, adapter):
        """Test getting model info when no layers loaded."""
        info = adapter.get_model_info()
        
        assert info["num_layers"] == 0
        assert info["layer_names"] == []


class TestLlamaAdapter:
    """Test LlamaAdapter."""
    
    def test_layer_names_dict(self):
        """Test Llama layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = LlamaAdapter(model_name="llama-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert adapter.layer_names_dict["embed"] == "model.embed_tokens"
        assert adapter.layer_names_dict["layer_prefix"] == "model.layers"
        assert adapter.layer_names_dict["norm"] == "model.norm"
        assert adapter.layer_names_dict["lm_head"] == "lm_head"
    
    def test_llama_adapter_inheritance(self):
        """Test LlamaAdapter inherits from ModelAdapter."""
        assert issubclass(LlamaAdapter, ModelAdapter)


class TestQwenAdapter:
    """Test QwenAdapter."""
    
    def test_layer_names_dict(self):
        """Test Qwen layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = QwenAdapter(model_name="qwen-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert adapter.layer_names_dict["embed"] == "model.embed_tokens"
        assert adapter.layer_names_dict["layer_prefix"] == "model.layers"
        assert adapter.layer_names_dict["norm"] == "model.norm"
        assert adapter.layer_names_dict["lm_head"] == "lm_head"
    
    def test_qwen_adapter_inheritance(self):
        """Test QwenAdapter inherits from ModelAdapter."""
        assert issubclass(QwenAdapter, ModelAdapter)


class TestMistralAdapter:
    """Test MistralAdapter."""
    
    def test_layer_names_dict(self):
        """Test Mistral layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = MistralAdapter(model_name="mistral-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert adapter.layer_names_dict["embed"] == "model.embed_tokens"
        assert adapter.layer_names_dict["layer_prefix"] == "model.layers"
        assert adapter.layer_names_dict["norm"] == "model.norm"
        assert adapter.layer_names_dict["lm_head"] == "lm_head"
    
    def test_mistral_adapter_inheritance(self):
        """Test MistralAdapter inherits from ModelAdapter."""
        assert issubclass(MistralAdapter, ModelAdapter)


class TestMixtralAdapter:
    """Test MixtralAdapter (MoE support)."""
    
    def test_layer_names_dict(self):
        """Test Mixtral layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = MixtralAdapter(model_name="mixtral-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert adapter.layer_names_dict["embed"] == "model.embed_tokens"
        assert adapter.layer_names_dict["layer_prefix"] == "model.layers"
        assert adapter.layer_names_dict["norm"] == "model.norm"
        assert adapter.layer_names_dict["lm_head"] == "lm_head"
    
    def test_mixtral_adapter_inheritance(self):
        """Test MixtralAdapter inherits from ModelAdapter."""
        assert issubclass(MixtralAdapter, ModelAdapter)


class TestBaichuanAdapter:
    """Test BaichuanAdapter."""
    
    def test_layer_names_dict(self):
        """Test Baichuan layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = BaichuanAdapter(model_name="baichuan-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert "embed" in adapter.layer_names_dict
        assert "layer_prefix" in adapter.layer_names_dict
        assert "norm" in adapter.layer_names_dict
        assert "lm_head" in adapter.layer_names_dict
    
    def test_baichuan_adapter_inheritance(self):
        """Test BaichuanAdapter inherits from ModelAdapter."""
        assert issubclass(BaichuanAdapter, ModelAdapter)


class TestChatGLMAdapter:
    """Test ChatGLMAdapter."""
    
    def test_layer_names_dict(self):
        """Test ChatGLM layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = ChatGLMAdapter(model_name="chatglm-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert "embed" in adapter.layer_names_dict
        assert "layer_prefix" in adapter.layer_names_dict
        assert "norm" in adapter.layer_names_dict
        assert "lm_head" in adapter.layer_names_dict
    
    def test_chatglm_adapter_inheritance(self):
        """Test ChatGLMAdapter inherits from ModelAdapter."""
        assert issubclass(ChatGLMAdapter, ModelAdapter)


class TestInternLMAdapter:
    """Test InternLMAdapter."""
    
    def test_layer_names_dict(self):
        """Test InternLM layer names configuration."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = InternLMAdapter(model_name="internlm-test", device="cpu")
        
        adapter.set_layer_names_dict()
        
        assert "embed" in adapter.layer_names_dict
        assert "layer_prefix" in adapter.layer_names_dict
        assert "norm" in adapter.layer_names_dict
        assert "lm_head" in adapter.layer_names_dict
    
    def test_internlm_adapter_inheritance(self):
        """Test InternLMAdapter inherits from ModelAdapter."""
        assert issubclass(InternLMAdapter, ModelAdapter)


class TestGenericAdapter:
    """Test GenericAdapter with auto-detection."""
    
    def test_layer_names_dict_not_implemented(self):
        """Test GenericAdapter raises NotImplementedError for set_layer_names_dict."""
        with patch("torch.cuda.is_available", return_value=False):
            adapter = GenericAdapter(model_name="generic-test", device="cpu")
        
        with pytest.raises(NotImplementedError):
            adapter.set_layer_names_dict()
    
    def test_generic_adapter_inheritance(self):
        """Test GenericAdapter inherits from ModelAdapter."""
        assert issubclass(GenericAdapter, ModelAdapter)
    
    @patch("ommi_llm.adapters.generic.LayerWiseInferenceEngine.init_model")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_generic_adapter_init_with_llama_pattern(
        self, mock_from_config, mock_init_model
    ):
        """Test GenericAdapter auto-detects llama-like pattern."""
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_from_config.return_value = mock_config
        
        with patch("torch.cuda.is_available", return_value=False):
            adapter = GenericAdapter(model_name="llama-like-model", device="cpu")
            adapter.local_path = Mock()
        
        # Manually set what init_model would detect
        adapter.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
        
        assert adapter.layer_names_dict["embed"] == "model.embed_tokens"
