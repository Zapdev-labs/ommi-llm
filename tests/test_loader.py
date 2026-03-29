"""
Tests for persistence layer (loader and sharder).
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from ommi_llm.persistence.loader import load_layer_weights, save_layer_weights
from ommi_llm.persistence.sharder import ModelSharder


class TestLoadLayerWeights:
    """Test layer weight loading."""
    
    @patch("safetensors.torch.load_file")
    def test_load_layer_weights_success(self, mock_load_file):
        """Test successful layer weight loading."""
        expected_weights = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        mock_load_file.return_value = expected_weights
        
        checkpoint_path = Path("/fake/checkpoint")
        layer_name = "model.layers.0"
        
        result = load_layer_weights(checkpoint_path, layer_name)
        
        assert result == expected_weights
        mock_load_file.assert_called_once_with(
            str(checkpoint_path / "model.layers.0.safetensors"),
            device="cpu"
        )
    
    @patch("safetensors.torch.load_file")
    def test_load_layer_weights_file_not_found(self, mock_load_file):
        """Test loading when file doesn't exist."""
        mock_load_file.side_effect = FileNotFoundError("File not found")
        
        checkpoint_path = Path("/fake/checkpoint")
        layer_name = "nonexistent_layer"
        
        with pytest.raises(FileNotFoundError):
            load_layer_weights(checkpoint_path, layer_name)


class TestSaveLayerWeights:
    """Test layer weight saving."""
    
    @patch("safetensors.torch.save_file")
    def test_save_layer_weights_success(self, mock_save_file):
        """Test successful layer weight saving."""
        weights = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        layer_name = "model.layers.0"
        checkpoint_path = Path("/fake/checkpoint")
        
        # Mock the path to avoid actual file system operations
        with patch.object(Path, "mkdir"):
            save_layer_weights(weights, layer_name, checkpoint_path)
        
        mock_save_file.assert_called_once()
        call_args = mock_save_file.call_args
        assert call_args[0][0] == weights
        assert str(call_args[0][1]) == str(checkpoint_path / "model.layers.0.safetensors")
    
    @patch("safetensors.torch.save_file")
    def test_save_layer_weights_creates_directory(self, mock_save_file):
        """Test that save creates parent directories."""
        weights = {"weight": torch.randn(10, 10)}
        layer_name = "model.layers.0"
        checkpoint_path = Path("/fake/checkpoint")
        
        with patch.object(Path, "mkdir") as mock_mkdir:
            save_layer_weights(weights, layer_name, checkpoint_path)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestModelSharder:
    """Test ModelSharder class."""
    
    @pytest.fixture
    def sharder(self, tmp_path):
        """Create a sharder fixture."""
        return ModelSharder(
            model_name="test-model",
            output_path=tmp_path / "sharded",
            compression=None,
        )
    
    def test_sharder_initialization(self, sharder, tmp_path):
        """Test sharder initialization."""
        assert sharder.model_name == "test-model"
        assert sharder.compression is None
        assert sharder.output_path.exists()
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_get_layer_names(self, mock_from_config, sharder):
        """Test layer name generation from config."""
        mock_config = Mock()
        mock_config.num_hidden_layers = 2
        mock_config.n_layer = None
        mock_config.num_layers = None
        mock_from_config.return_value = mock_config
        
        layer_names = sharder._get_layer_names(mock_config)
        
        assert "model.embed_tokens" in layer_names
        assert "model.layers.0" in layer_names
        assert "model.layers.1" in layer_names
        assert "model.norm" in layer_names
        assert "lm_head" in layer_names
        assert len(layer_names) == 5  # embed + 2 layers + norm + lm_head
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_get_layer_names_alt_config(self, mock_from_config, sharder):
        """Test layer names with alternative config attributes."""
        mock_config = Mock()
        mock_config.num_hidden_layers = None
        mock_config.n_layer = 3
        mock_config.num_layers = None
        
        layer_names = sharder._get_layer_names(mock_config)
        
        assert len(layer_names) == 6  # embed + 3 layers + norm + lm_head
        assert "model.layers.0" in layer_names
        assert "model.layers.1" in layer_names
        assert "model.layers.2" in layer_names
    
    @patch("safetensors.torch.load_file")
    @patch("pathlib.Path.exists")
    def test_load_checkpoint_single_file(self, mock_exists, mock_load_file, sharder):
        """Test loading from single safetensors file."""
        expected_weights = {"model.embed_tokens.weight": torch.randn(1000, 768)}
        mock_load_file.return_value = expected_weights
        
        # Mock exists to return True for model.safetensors
        def exists_side_effect(self):
            return str(self).endswith("model.safetensors")
        mock_exists.side_effect = exists_side_effect
        
        result = sharder._load_checkpoint()
        
        assert result == expected_weights
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_load_checkpoint_via_transformers(self, mock_from_pretrained, sharder):
        """Test loading checkpoint via transformers."""
        mock_model = Mock()
        expected_weights = {"model.embed_tokens.weight": torch.randn(1000, 768)}
        mock_model.state_dict.return_value = expected_weights
        mock_from_pretrained.return_value = mock_model
        
        result = sharder._load_checkpoint()
        
        assert result == expected_weights
        mock_from_pretrained.assert_called_once()
    
    @patch("safetensors.torch.save_file")
    def test_extract_and_save_layer(self, mock_save_file, sharder):
        """Test extracting and saving layer weights."""
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(2048, 768),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(768, 768),
        }
        layer_name = "model.layers.0"
        
        shard_path = sharder._extract_and_save_layer(state_dict, layer_name)
        
        # Check that only layer 0 weights were extracted
        saved_weights = mock_save_file.call_args[0][0]
        assert "model.layers.0.self_attn.q_proj.weight" in saved_weights
        assert "model.layers.0.self_attn.k_proj.weight" in saved_weights
        assert "model.layers.0.mlp.gate_proj.weight" in saved_weights
        assert "model.layers.1.self_attn.q_proj.weight" not in saved_weights
    
    @patch("safetensors.torch.save_file")
    def test_extract_and_save_layer_empty_warning(self, mock_save_file, sharder):
        """Test warning when no weights found for layer."""
        state_dict = {
            "model.layers.1.self_attn.q_proj.weight": torch.randn(768, 768),
        }
        layer_name = "model.layers.0"  # No matching weights
        
        with patch("ommi_llm.persistence.sharder.logger") as mock_logger:
            sharder._extract_and_save_layer(state_dict, layer_name)
            mock_logger.warning.assert_called_once()
    
    @patch("ommi_llm.persistence.sharder.clean_memory")
    @patch.object(ModelSharder, "_load_checkpoint")
    @patch.object(ModelSharder, "_extract_and_save_layer")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_shard_model_success(
        self, mock_from_config, mock_extract, mock_load, mock_clean, sharder
    ):
        """Test successful model sharding."""
        mock_config = Mock()
        mock_config.num_hidden_layers = 2
        mock_from_config.return_value = mock_config
        
        mock_state_dict = {"model.embed_tokens.weight": torch.randn(1000, 768)}
        mock_load.return_value = mock_state_dict
        
        mock_extract.return_value = Path("/fake/path/layer.safetensors")
        
        # Mock tqdm to just iterate
        with patch("ommi_llm.persistence.sharder.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, **kwargs: x
            
            result = sharder.shard_model()
        
        assert len(result) == 5  # embed + 2 layers + norm + lm_head
        mock_load.assert_called_once()
        mock_clean.assert_called_once()
    
    @patch("transformers.AutoConfig.from_pretrained")
    def test_shard_model_skip_if_exists(self, mock_from_config, sharder, tmp_path):
        """Test skipping when shards already exist."""
        mock_config = Mock()
        mock_config.num_hidden_layers = 1
        mock_from_config.return_value = mock_config
        
        # Create the expected output files
        output_path = tmp_path / "sharded"
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "model.embed_tokens.safetensors").touch()
        (output_path / "model.layers.0.safetensors").touch()
        (output_path / "model.norm.safetensors").touch()
        (output_path / "lm_head.safetensors").touch()
        
        with patch("ommi_llm.persistence.sharder.logger") as mock_logger:
            result = sharder.shard_model(skip_if_exists=True)
            mock_logger.info.assert_called_with("Model already sharded, skipping")


class TestModelSharderCompression:
    """Test ModelSharder compression functionality."""
    
    @pytest.fixture
    def sharder_4bit(self, tmp_path):
        """Create a 4bit sharder fixture."""
        return ModelSharder(
            model_name="test-model",
            output_path=tmp_path / "sharded",
            compression="4bit",
        )
    
    def test_compression_initialization(self, sharder_4bit):
        """Test sharder with compression enabled."""
        assert sharder_4bit.compression == "4bit"
    
    @patch("ommi_llm.persistence.sharder.logger")
    def test_compress_weights_without_bitsandbytes(self, mock_logger, sharder_4bit):
        """Test compression when bitsandbytes not installed."""
        weights = {"weight": torch.randn(10, 10)}
        
        result = sharder_4bit._compress_weights(weights)
        
        # Should return original weights when bitsandbytes not available
        assert result == weights
        mock_logger.warning.assert_called_once()
