"""
Tests for CLI interface.
"""

import pytest
from typer.testing import CliRunner
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Import the CLI app
from ommi_llm.cli import app, get_cache_dir


runner = CliRunner()


class TestCLIVersion:
    """Test version command."""
    
    def test_version_command(self):
        """Test version command shows version info."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "ommi-llm" in result.output
        assert "version" in result.output


class TestCLICache:
    """Test cache directory functions."""
    
    @patch.dict("os.environ", {}, clear=True)
    def test_get_cache_dir_default(self):
        """Test default cache directory."""
        cache_dir = get_cache_dir()
        
        assert isinstance(cache_dir, Path)
        assert ".cache" in str(cache_dir)
        assert "ommi-llm" in str(cache_dir)
    
    @patch.dict("os.environ", {"OMMI_CACHE_DIR": "/custom/cache"})
    def test_get_cache_dir_custom_env(self):
        """Test custom cache directory from environment."""
        cache_dir = get_cache_dir()
        
        assert str(cache_dir) == "/custom/cache"


class TestCLIMemory:
    """Test memory command."""
    
    @patch("ommi_llm.cli.MemoryManager")
    def test_memory_command(self, mock_mm_class):
        """Test memory stats display."""
        mock_mm = Mock()
        mock_mm.get_memory_stats.return_value = {
            "ram": {
                "total": 32.0,
                "available": 16.0,
                "percent": 50.0,
            },
            "vram": {
                "total": 8.0,
                "allocated": 2.0,
                "reserved": 3.0,
                "free": 6.0,
            },
        }
        mock_mm_class.return_value = mock_mm
        
        result = runner.invoke(app, ["memory"])
        
        assert result.exit_code == 0
        assert "RAM" in result.output
        assert "VRAM" in result.output
        assert "Total" in result.output


class TestCLIDownload:
    """Test download command."""
    
    @patch("huggingface_hub.snapshot_download")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_download_model_success(self, mock_config, mock_download):
        """Test successful model download."""
        mock_download.return_value = "/fake/download/path"
        
        mock_cfg = Mock()
        mock_cfg.architectures = ["LlamaForCausalLM"]
        mock_cfg.model_type = "llama"
        mock_cfg.hidden_size = 4096
        mock_cfg.num_hidden_layers = 32
        mock_cfg.num_attention_heads = 32
        mock_config.return_value = mock_cfg
        
        result = runner.invoke(app, ["download", "meta-llama/Llama-2-7b"])
        
        assert result.exit_code == 0
        assert "downloaded successfully" in result.output
        mock_download.assert_called_once()
    
    @patch("huggingface_hub.snapshot_download")
    def test_download_model_with_local_dir(self, mock_download):
        """Test download with custom local directory."""
        mock_download.return_value = "/custom/dir"
        
        result = runner.invoke(app, [
            "download", "meta-llama/Llama-2-7b",
            "--local-dir", "/custom/dir"
        ])
        
        assert result.exit_code == 0
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["local_dir"] == "/custom/dir"
    
    @patch("huggingface_hub.snapshot_download")
    def test_download_model_error(self, mock_download):
        """Test download error handling."""
        mock_download.side_effect = Exception("Download failed")
        
        result = runner.invoke(app, ["download", "invalid/model"])
        
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCLISearch:
    """Test search command."""
    
    @patch("huggingface_hub.list_models")
    def test_search_models_success(self, mock_list_models):
        """Test successful model search."""
        mock_model = Mock()
        mock_model.id = "meta-llama/Llama-2-7b"
        mock_model.downloads = 1000000
        mock_model.tags = ["llama", "text-generation"]
        
        mock_list_models.return_value = [mock_model]
        
        result = runner.invoke(app, ["search", "llama 7b"])
        
        assert result.exit_code == 0
        assert "Llama-2-7b" in result.output
        assert "1,000,000" in result.output
    
    @patch("huggingface_hub.list_models")
    def test_search_models_empty_results(self, mock_list_models):
        """Test search with no results."""
        mock_list_models.return_value = []
        
        result = runner.invoke(app, ["search", "nonexistent_model_xyz"])
        
        assert result.exit_code == 0
        assert "No models found" in result.output
    
    @patch("huggingface_hub.list_models")
    def test_search_models_with_filter(self, mock_list_models):
        """Test search with filter tag."""
        mock_model = Mock()
        mock_model.id = "some-model"
        mock_model.downloads = 100
        mock_model.tags = ["text-generation"]
        
        mock_list_models.return_value = [mock_model]
        
        result = runner.invoke(app, [
            "search", "model",
            "--filter", "text-generation"
        ])
        
        assert result.exit_code == 0
        call_kwargs = mock_list_models.call_args[1]
        assert call_kwargs["filter"] == "text-generation"


class TestCLIList:
    """Test list command."""
    
    @patch("ommi_llm.cli.get_cache_dir")
    def test_list_models_empty(self, mock_get_cache):
        """Test listing with no models."""
        mock_cache = Mock()
        mock_cache.exists.return_value = False
        mock_get_cache.return_value = mock_cache
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "No models" in result.output
    
    @patch("ommi_llm.cli.get_cache_dir")
    def test_list_models_with_models(self, mock_get_cache, tmp_path):
        """Test listing with models in cache."""
        # Create fake model structure
        model_dir = tmp_path / "models" / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").touch()
        (model_dir / "model.safetensors").touch()
        
        mock_get_cache.return_value = tmp_path / "models"
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "test-model" in result.output


class TestCLIRemove:
    """Test remove command."""
    
    @patch("ommi_llm.cli.get_cache_dir")
    def test_remove_model_not_found(self, mock_get_cache):
        """Test removing non-existent model."""
        mock_cache = Mock()
        mock_cache.__truediv__ = Mock(return_value=Path("/nonexistent"))
        mock_get_cache.return_value = mock_cache
        
        result = runner.invoke(app, ["rm", "nonexistent-model"])
        
        assert result.exit_code == 1
        assert "not found" in result.output
    
    @patch("shutil.rmtree")
    @patch("ommi_llm.cli.get_cache_dir")
    @patch("typer.confirm")
    def test_remove_model_with_confirmation(self, mock_confirm, mock_get_cache, mock_rmtree, tmp_path):
        """Test removing model with user confirmation."""
        mock_confirm.return_value = True
        
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        
        mock_cache = Mock()
        mock_cache.__truediv__ = Mock(return_value=model_dir)
        mock_cache.exists.return_value = True
        mock_get_cache.return_value = mock_cache
        
        result = runner.invoke(app, ["rm", "test-model"])
        
        assert result.exit_code == 0
        mock_rmtree.assert_called_once()
    
    @patch("shutil.rmtree")
    @patch("ommi_llm.cli.get_cache_dir")
    def test_remove_model_skip_confirmation(self, mock_get_cache, mock_rmtree, tmp_path):
        """Test removing model with --yes flag."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        
        mock_cache = Mock()
        mock_cache.__truediv__ = Mock(return_value=model_dir)
        mock_get_cache.return_value = mock_cache
        
        result = runner.invoke(app, ["rm", "test-model", "--yes"])
        
        assert result.exit_code == 0
        mock_rmtree.assert_called_once()


class TestCLILoad:
    """Test load command."""
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_load_model_success(self, mock_from_pretrained):
        """Test successful model loading."""
        mock_model = Mock()
        mock_model.get_model_info.return_value = {
            "model_name": "test-model",
            "architecture": "LlamaAdapter",
            "num_layers": 32,
            "device": "cuda",
        }
        mock_from_pretrained.return_value = mock_model
        
        result = runner.invoke(app, ["load", "test-model"])
        
        assert result.exit_code == 0
        assert "Model Loaded" in result.output
        mock_from_pretrained.assert_called_once()
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_load_model_with_options(self, mock_from_pretrained):
        """Test loading with custom options."""
        mock_model = Mock()
        mock_model.get_model_info.return_value = {}
        mock_from_pretrained.return_value = mock_model
        
        result = runner.invoke(app, [
            "load", "test-model",
            "--device", "cpu",
            "--dtype", "float32",
            "--no-prefetching",
            "--compression", "4bit"
        ])
        
        assert result.exit_code == 0
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["dtype"] == "float32"
        assert call_kwargs["prefetching"] is False
        assert call_kwargs["compression"] == "4bit"
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_load_model_error(self, mock_from_pretrained):
        """Test load error handling."""
        mock_from_pretrained.side_effect = Exception("Model not found")
        
        result = runner.invoke(app, ["load", "invalid-model"])
        
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCLIGenerate:
    """Test generate command."""
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_generate_success(self, mock_from_pretrained):
        """Test successful text generation."""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 10))
        }
        mock_tokenizer.decode.return_value = "Generated text output"
        mock_llm.tokenizer = mock_tokenizer
        mock_llm.device = "cuda"
        mock_llm.generate.return_value = torch.randint(0, 1000, (1, 20))
        
        mock_from_pretrained.return_value = mock_llm
        
        result = runner.invoke(app, [
            "generate", "test-model", "Hello world"
        ])
        
        assert result.exit_code == 0
        assert "Output:" in result.output
        mock_llm.generate.assert_called_once()
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_generate_with_options(self, mock_from_pretrained):
        """Test generation with custom options."""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.randint(0, 1000, (1, 10))}
        mock_tokenizer.decode.return_value = "Output"
        mock_llm.tokenizer = mock_tokenizer
        mock_llm.device = "cuda"
        mock_llm.generate.return_value = torch.randint(0, 1000, (1, 20))
        
        mock_from_pretrained.return_value = mock_llm
        
        result = runner.invoke(app, [
            "generate", "test-model", "Hello",
            "--max-tokens", "50",
            "--temperature", "0.5",
            "--device", "cpu"
        ])
        
        assert result.exit_code == 0
        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 50
        assert call_kwargs["temperature"] == 0.5
    
    @patch("ommi_llm.cli.AutoModel.from_pretrained")
    def test_generate_error(self, mock_from_pretrained):
        """Test generation error handling."""
        mock_from_pretrained.side_effect = Exception("Failed to load")
        
        result = runner.invoke(app, [
            "generate", "invalid-model", "Hello"
        ])
        
        assert result.exit_code == 1


class TestCLIShard:
    """Test shard command."""
    
    @patch("ommi_llm.cli.ModelSharder")
    def test_shard_model_success(self, mock_sharder_class):
        """Test successful model sharding."""
        mock_sharder = Mock()
        mock_sharder.shard_model.return_value = [Path("layer1.safetensors"), Path("layer2.safetensors")]
        mock_sharder_class.return_value = mock_sharder
        
        result = runner.invoke(app, [
            "shard", "test-model", "/output/dir"
        ])
        
        assert result.exit_code == 0
        assert "sharded" in result.output.lower()
        mock_sharder.shard_model.assert_called_once()
    
    @patch("ommi_llm.cli.ModelSharder")
    def test_shard_model_with_compression(self, mock_sharder_class):
        """Test sharding with compression."""
        mock_sharder = Mock()
        mock_sharder.shard_model.return_value = [Path("layer1.safetensors")]
        mock_sharder_class.return_value = mock_sharder
        
        result = runner.invoke(app, [
            "shard", "test-model", "/output/dir",
            "--compression", "4bit"
        ])
        
        assert result.exit_code == 0
        call_kwargs = mock_sharder_class.call_args[1]
        assert call_kwargs["compression"] == "4bit"
    
    @patch("ommi_llm.cli.ModelSharder")
    def test_shard_model_error(self, mock_sharder_class):
        """Test sharding error handling."""
        mock_sharder = Mock()
        mock_sharder.shard_model.side_effect = Exception("Sharding failed")
        mock_sharder_class.return_value = mock_sharder
        
        result = runner.invoke(app, [
            "shard", "invalid-model", "/output/dir"
        ])
        
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCLIArchitectures:
    """Test list-architectures command."""
    
    @patch("ommi_llm.cli.SUPPORTED_ARCHITECTURES", ["LlamaForCausalLM", "Qwen2ForCausalLM"])
    def test_list_architectures(self):
        """Test listing supported architectures."""
        result = runner.invoke(app, ["list-architectures"])
        
        assert result.exit_code == 0
        assert "Supported Architectures" in result.output
        assert "LlamaForCausalLM" in result.output
        assert "Qwen2ForCausalLM" in result.output


class TestCLIMain:
    """Test CLI main entry point."""
    
    def test_no_args_shows_help(self):
        """Test that running with no args shows help."""
        result = runner.invoke(app, [])
        
        assert result.exit_code == 0
        assert "Usage:" in result.output
    
    def test_help_command(self):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Commands:" in result.output
