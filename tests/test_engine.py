"""
Tests for the layer-wise inference engine.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import gc

from ommi_llm.core.engine import LayerWiseInferenceEngine
from ommi_llm.utils.memory import MemoryManager


class MockAdapter(LayerWiseInferenceEngine):
    """Mock adapter for testing."""
    
    def set_layer_names_dict(self) -> None:
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }


@pytest.fixture
def engine():
    """Create a basic engine fixture."""
    with patch("torch.cuda.is_available", return_value=False):
        engine = MockAdapter(
            model_name="test-model",
            device="cpu",
            dtype=torch.float32,
            prefetching=False,
        )
        return engine


@pytest.fixture
def cuda_engine():
    """Create a CUDA engine fixture."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch.object(torch.cuda, "Stream") as mock_stream:
            engine = MockAdapter(
                model_name="test-model",
                device="cuda",
                dtype=torch.float16,
                prefetching=True,
            )
            return engine


class TestLayerWiseInferenceEngineInitialization:
    """Test engine initialization."""
    
    def test_basic_initialization(self, engine):
        """Test basic engine initialization."""
        assert engine.model_name == "test-model"
        assert engine.device == "cpu"
        assert engine.running_dtype == torch.float32
        assert engine.prefetching is False
        assert engine.compression is None
        
    def test_cuda_prefetching_enabled(self):
        """Test that prefetching is enabled for CUDA."""
        with patch("torch.cuda.is_available", return_value=True):
            engine = MockAdapter(
                model_name="test-model",
                device="cuda:0",
                dtype=torch.float16,
                prefetching=True,
            )
            assert engine.prefetching is True
            
    def test_prefetching_disabled_for_non_cuda(self):
        """Test that prefetching is disabled for non-CUDA devices."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = MockAdapter(
                model_name="test-model",
                device="cpu",
                dtype=torch.float32,
                prefetching=True,  # User requests it
            )
            assert engine.prefetching is False
            
    def test_prefetching_disabled_for_compression(self):
        """Test that prefetching is disabled when compression is used."""
        with patch("torch.cuda.is_available", return_value=True):
            engine = MockAdapter(
                model_name="test-model",
                device="cuda",
                dtype=torch.float16,
                prefetching=True,
                compression="4bit",
            )
            assert engine.prefetching is False
            
    def test_local_path_resolution(self):
        """Test local path is converted to Path object."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = MockAdapter(
                model_name="test-model",
                device="cpu",
                local_path="/path/to/model",
            )
            assert isinstance(engine.local_path, Path)
            assert str(engine.local_path) == "/path/to/model"


class TestTokenizerInitialization:
    """Test tokenizer initialization."""
    
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_success(self, mock_from_pretrained, engine):
        """Test successful tokenizer initialization."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        engine.init_tokenizer()
        
        assert engine.tokenizer is mock_tokenizer
        assert engine.tokenizer.pad_token == "<eos>"
        
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_init_tokenizer_with_existing_pad_token(self, mock_from_pretrained, engine):
        """Test tokenizer with existing pad token."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        engine.init_tokenizer()
        
        assert engine.tokenizer.pad_token == "<pad>"


class TestModelInitialization:
    """Test model initialization with config fixing."""
    
    @patch("transformers.AutoConfig.from_pretrained")
    @patch("accelerate.init_empty_weights")
    @patch("transformers.AutoModelForCausalLM.from_config")
    def test_init_model_with_missing_vocab_size(
        self, mock_from_config, mock_init_empty, mock_from_config_func, engine
    ):
        """Test model init fixes missing vocab_size."""
        mock_config = Mock()
        mock_config.model_type = "llama"
        del mock_config.vocab_size  # Remove vocab_size
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        mock_from_config_func.return_value = mock_config
        
        # Mock tokenizer
        engine.tokenizer = Mock()
        len(engine.tokenizer)  # Just to show it's used
        
        mock_model = Mock()
        mock_model.named_buffers.return_value = []
        mock_from_config.return_value = mock_model
        
        with patch.object(engine, "set_layer_names_dict"):
            with patch.object(engine, "_build_layer_list"):
                engine.init_model()
        
        # Check vocab_size was set
        assert hasattr(mock_config, "vocab_size")
        
    @patch("transformers.AutoConfig.from_pretrained")
    def test_init_model_qwen_vocab_size_fallback(self, mock_from_config, engine):
        """Test Qwen vocab_size fallback."""
        mock_config = Mock()
        mock_config.model_type = "qwen"
        del mock_config.vocab_size
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_from_config.return_value = mock_config
        
        engine.local_path = Path("/fake/path")
        
        # Mock the file operations to fail gracefully
        with patch.object(Path, "exists", return_value=False):
            with patch.object(engine, "set_layer_names_dict"):
                with patch.object(engine, "_build_layer_list"):
                    with patch("accelerate.init_empty_weights"):
                        with patch("transformers.AutoModelForCausalLM.from_config") as mock_from_config:
                            mock_model = Mock()
                            mock_model.named_buffers.return_value = []
                            mock_from_config.return_value = mock_model
                            engine.init_model()
        
        # Qwen default vocab size
        assert mock_config.vocab_size == 151936
        
    @patch("transformers.AutoConfig.from_pretrained")
    def test_init_model_hidden_size_fallback(self, mock_from_config, engine):
        """Test hidden_size fallback for missing attribute."""
        mock_config = Mock()
        mock_config.model_type = "llama"
        mock_config.vocab_size = 32000
        # No hidden_size
        mock_config.num_hidden_layers = 32
        mock_from_config.return_value = mock_config
        
        engine.local_path = Path("/fake/path")
        
        with patch.object(engine, "set_layer_names_dict"):
            with patch.object(engine, "_build_layer_list"):
                with patch("accelerate.init_empty_weights"):
                    with patch("transformers.AutoModelForCausalLM.from_config") as mock_from_config:
                        mock_model = Mock()
                        mock_model.named_buffers.return_value = []
                        mock_from_config.return_value = mock_model
                        engine.init_model()
        
        # Llama default hidden size
        assert mock_config.hidden_size == 4096


class TestLayerManagement:
    """Test layer loading and unloading."""
    
    def test_get_nested_attr_success(self, engine):
        """Test getting nested attribute."""
        mock_obj = Mock()
        mock_obj.a = Mock()
        mock_obj.a.b = Mock()
        mock_obj.a.b.c = "value"
        
        result = engine._get_nested_attr(mock_obj, "a.b.c")
        assert result == "value"
        
    def test_get_nested_attr_missing(self, engine):
        """Test getting missing nested attribute."""
        mock_obj = Mock()
        mock_obj.a = Mock()
        
        result = engine._get_nested_attr(mock_obj, "a.b.c")
        assert result is None
        
    def test_build_layer_list(self, engine):
        """Test building layer list from model."""
        # Create mock model structure
        mock_embed = Mock()
        mock_layer0 = Mock()
        mock_layer1 = Mock()
        mock_norm = Mock()
        mock_lm_head = Mock()
        
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.embed_tokens = mock_embed
        mock_model.model.layers = [mock_layer0, mock_layer1]
        mock_model.model.norm = mock_norm
        mock_model.lm_head = mock_lm_head
        
        engine.model = mock_model
        engine.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
        }
        
        engine._build_layer_list()
        
        assert len(engine.layers) == 5  # embed + 2 layers + norm + lm_head
        assert len(engine.layer_names) == 5
        assert engine.layers[0] == mock_embed
        assert engine.layers[1] == mock_layer0
        assert engine.layers[2] == mock_layer1
        assert engine.layers[3] == mock_norm
        assert engine.layers[4] == mock_lm_head
        
    @patch("ommi_llm.core.engine.load_layer_weights")
    def test_load_layer_to_cpu(self, mock_load_weights, engine):
        """Test loading layer weights to CPU."""
        mock_weights = {"weight": torch.randn(10, 10)}
        mock_load_weights.return_value = mock_weights
        
        engine.checkpoint_path = Path("/fake/checkpoint")
        
        result = engine.load_layer_to_cpu("layer_0")
        
        assert result == mock_weights
        mock_load_weights.assert_called_once_with(Path("/fake/checkpoint"), "layer_0")
        
    @patch("accelerate.utils.set_module_tensor_to_device")
    def test_move_layer_to_device(self, mock_set_tensor, engine):
        """Test moving layer to device."""
        mock_model = Mock()
        engine.model = mock_model
        engine.running_device = "cpu"
        engine.running_dtype = torch.float32
        
        state_dict = {"weight": torch.randn(10, 10)}
        
        result = engine.move_layer_to_device(state_dict)
        
        assert "weight" in result
        mock_set_tensor.assert_called_once()
        
    def test_unload_layer(self, engine):
        """Test unloading layer from device."""
        mock_layer = Mock()
        mock_memory_manager = Mock()
        engine.memory_manager = mock_memory_manager
        
        engine.unload_layer(mock_layer)
        
        mock_layer.to.assert_called_once_with("meta")
        mock_memory_manager.clean_memory.assert_called_once()


class TestForwardPass:
    """Test forward pass execution."""
    
    def test_forward_not_initialized_raises(self, engine):
        """Test forward raises if model not initialized."""
        engine._model_initialized = False
        
        with pytest.raises(RuntimeError, match="Model not initialized"):
            engine.forward(torch.randn(1, 10))
            
    @patch.object(LayerWiseInferenceEngine, "_forward_sequential")
    def test_forward_sequential_execution(self, mock_forward_seq, engine):
        """Test forward uses sequential execution without prefetching."""
        engine._model_initialized = True
        engine.prefetching = False
        engine.layers = [Mock()]
        engine.layer_names = ["layer_0"]
        
        input_ids = torch.randint(0, 100, (1, 10))
        
        engine.forward(input_ids)
        
        mock_forward_seq.assert_called_once()
        
    @patch.object(LayerWiseInferenceEngine, "_forward_with_prefetching")
    def test_forward_with_prefetching(self, mock_forward_prefetch, cuda_engine):
        """Test forward uses prefetching when enabled."""
        cuda_engine._model_initialized = True
        cuda_engine.prefetching = True
        cuda_engine.layers = [Mock()]
        cuda_engine.layer_names = ["layer_0"]
        
        input_ids = torch.randint(0, 100, (1, 10))
        
        engine.forward(input_ids)
        
        mock_forward_prefetch.assert_called_once()


class TestGenerate:
    """Test token generation."""
    
    @patch.object(LayerWiseInferenceEngine, "forward")
    def test_generate_basic(self, mock_forward, engine):
        """Test basic generation."""
        engine._model_initialized = True
        
        # Mock forward to return logits
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        mock_forward.return_value = mock_output
        
        input_ids = torch.randint(0, 1000, (1, 5))
        
        result = engine.generate(input_ids, max_new_tokens=3)
        
        assert result.shape[1] == 8  # 5 input + 3 new
        assert mock_forward.call_count == 3
        
    @patch.object(LayerWiseInferenceEngine, "forward")
    def test_generate_with_temperature(self, mock_forward, engine):
        """Test generation with temperature scaling."""
        engine._model_initialized = True
        
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        mock_forward.return_value = mock_output
        
        input_ids = torch.randint(0, 1000, (1, 5))
        
        engine.generate(input_ids, max_new_tokens=1, temperature=0.5)
        
        mock_forward.assert_called_once()
        
    @patch.object(LayerWiseInferenceEngine, "forward")
    def test_generate_with_top_k(self, mock_forward, engine):
        """Test generation with top-k sampling."""
        engine._model_initialized = True
        
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        mock_forward.return_value = mock_output
        
        input_ids = torch.randint(0, 1000, (1, 5))
        
        engine.generate(input_ids, max_new_tokens=1, top_k=10)
        
        mock_forward.assert_called_once()
        
    @patch.object(LayerWiseInferenceEngine, "forward")
    def test_generate_with_top_p(self, mock_forward, engine):
        """Test generation with top-p (nucleus) sampling."""
        engine._model_initialized = True
        
        mock_output = Mock()
        mock_output.logits = torch.randn(1, 1, 1000)
        mock_output.past_key_values = None
        mock_forward.return_value = mock_output
        
        input_ids = torch.randint(0, 1000, (1, 5))
        
        engine.generate(input_ids, max_new_tokens=1, top_p=0.9)
        
        mock_forward.assert_called_once()
        
    def test_generate_not_initialized_raises(self, engine):
        """Test generate raises if model not initialized."""
        engine._model_initialized = False
        
        with pytest.raises(RuntimeError, match="Model not initialized"):
            engine.generate(torch.randint(0, 1000, (1, 5)))


class TestFromPretrained:
    """Test from_pretrained factory method."""
    
    @patch.object(MockAdapter, "init_tokenizer")
    @patch.object(MockAdapter, "init_model")
    @patch.object(MockAdapter, "setup_cuda_stream")
    def test_from_pretrained_basic(self, mock_setup, mock_init_model, mock_init_tok):
        """Test basic from_pretrained flow."""
        with patch("torch.cuda.is_available", return_value=False):
            instance = MockAdapter.from_pretrained(
                model_name="test-model",
                device="cpu",
                dtype="float32",
            )
        
        assert isinstance(instance, MockAdapter)
        assert instance.model_name == "test-model"
        mock_init_tok.assert_called_once()
        mock_init_model.assert_called_once()
        mock_setup.assert_called_once()
        
    @patch.object(MockAdapter, "init_tokenizer")
    @patch.object(MockAdapter, "init_model")
    @patch.object(MockAdapter, "setup_cuda_stream")
    def test_from_pretrained_dtype_parsing(self, mock_setup, mock_init_model, mock_init_tok):
        """Test dtype string parsing."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        for dtype_str, expected_dtype in dtype_map.items():
            with patch("torch.cuda.is_available", return_value=False):
                instance = MockAdapter.from_pretrained(
                    model_name="test-model",
                    device="cpu",
                    dtype=dtype_str,
                )
            
            assert instance.running_dtype == expected_dtype
            
    @patch.object(MockAdapter, "init_tokenizer")
    @patch.object(MockAdapter, "init_model")
    @patch.object(MockAdapter, "setup_cuda_stream")
    def test_from_pretrained_with_local_path(self, mock_setup, mock_init_model, mock_init_tok):
        """Test from_pretrained with local path."""
        with patch("torch.cuda.is_available", return_value=False):
            instance = MockAdapter.from_pretrained(
                model_name="test-model",
                local_path="/local/model",
                device="cpu",
            )
        
        assert str(instance.local_path) == "/local/model"


class TestCUDAStream:
    """Test CUDA stream setup."""
    
    def test_setup_cuda_stream_with_prefetching(self):
        """Test CUDA stream setup when prefetching is enabled."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(torch.cuda, "Stream") as mock_stream_class:
                mock_stream = Mock()
                mock_stream_class.return_value = mock_stream
                
                engine = MockAdapter(
                    model_name="test-model",
                    device="cuda",
                    prefetching=True,
                )
                engine.setup_cuda_stream()
                
                assert engine.stream is mock_stream
                
    def test_setup_cuda_stream_without_cuda(self):
        """Test CUDA stream not setup without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            engine = MockAdapter(
                model_name="test-model",
                device="cpu",
                prefetching=False,
            )
            engine.setup_cuda_stream()
            
            assert engine.stream is None


class TestRunLayer:
    """Test single layer execution."""
    
    def test_run_embedding_layer(self, engine):
        """Test running embedding layer."""
        mock_layer = Mock()
        mock_layer.return_value = torch.randn(1, 10, 768)
        
        engine.layer_names_dict = {"embed": "embed"}
        
        seq = torch.randint(0, 100, (1, 10))
        result = engine._run_layer(
            layer=mock_layer,
            layer_name="embed",
            seq=seq,
            layer_idx=0,
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            kv_cache_list=None,
        )
        
        mock_layer.assert_called_once_with(seq)
        
    def test_run_transformer_layer(self, engine):
        """Test running transformer layer."""
        mock_layer = Mock()
        mock_layer.return_value = (torch.randn(1, 10, 768),)
        
        engine.layer_names_dict = {"layer_prefix": "layers"}
        
        seq = torch.randn(1, 10, 768)
        result = engine._run_layer(
            layer=mock_layer,
            layer_name="layers.0",
            seq=seq,
            layer_idx=1,  # Not first layer
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            kv_cache_list=None,
        )
        
        mock_layer.assert_called_once()
        call_kwargs = mock_layer.call_args[1]
        assert call_kwargs["use_cache"] is True


class TestBuildOutput:
    """Test output construction."""
    
    def test_build_output_basic(self, engine):
        """Test basic output building."""
        hidden_states = torch.randn(1, 10, 768)
        
        result = engine._build_output(hidden_states, None, False)
        
        assert result.logits is hidden_states
        assert result.past_key_values is None
        
    def test_build_output_with_cache(self, engine):
        """Test output building with KV cache."""
        hidden_states = torch.randn(1, 10, 768)
        
        k_cache = torch.randn(1, 8, 10, 64)
        v_cache = torch.randn(1, 8, 10, 64)
        kv_cache_list = [[[k_cache], [v_cache]]]
        
        result = engine._build_output(hidden_states, kv_cache_list, True)
        
        assert result.logits is hidden_states
        assert result.past_key_values is not None
        assert len(result.past_key_values) == 1
