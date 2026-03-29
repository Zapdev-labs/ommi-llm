"""
Tests for memory management utilities.
"""

import pytest
import torch
import psutil
from unittest.mock import Mock, patch, MagicMock
import gc
import ctypes

from ommi_llm.utils.memory import MemoryManager, clean_memory


class TestMemoryManager:
    """Test MemoryManager class."""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU memory manager fixture."""
        return MemoryManager(device="cpu")
    
    @pytest.fixture
    def cuda_manager(self):
        """Create a CUDA memory manager fixture."""
        with patch("torch.cuda.is_available", return_value=True):
            return MemoryManager(device="cuda")
    
    def test_initialization_cpu(self, cpu_manager):
        """Test CPU memory manager initialization."""
        assert cpu_manager.device == "cpu"
        assert cpu_manager._peak_memory == 0
    
    def test_initialization_cuda(self, cuda_manager):
        """Test CUDA memory manager initialization."""
        assert cuda_manager.device == "cuda"
    
    def test_get_memory_stats_cpu(self, cpu_manager):
        """Test getting memory stats on CPU."""
        stats = cpu_manager.get_memory_stats()
        
        assert "ram" in stats
        assert "total" in stats["ram"]
        assert "available" in stats["ram"]
        assert "percent" in stats["ram"]
        
        # Check values are reasonable
        assert stats["ram"]["total"] > 0
        assert 0 <= stats["ram"]["percent"] <= 100
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    def test_get_memory_stats_cuda(
        self, mock_reserved, mock_allocated, mock_props, mock_available, cuda_manager
    ):
        """Test getting memory stats on CUDA."""
        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024**3  # 8 GB
        mock_props.return_value = mock_device_props
        
        mock_allocated.return_value = 2 * 1024**3  # 2 GB
        mock_reserved.return_value = 3 * 1024**3  # 3 GB
        
        stats = cuda_manager.get_memory_stats()
        
        assert "ram" in stats
        assert "vram" in stats
        assert stats["vram"]["total"] == 8.0  # GB
        assert stats["vram"]["allocated"] == 2.0  # GB
        assert stats["vram"]["reserved"] == 3.0  # GB
        assert stats["vram"]["free"] == 6.0  # GB
    
    @patch("gc.collect")
    @patch("ctypes.CDLL")
    def test_clean_memory_cpu(self, mock_cdll, mock_gc, cpu_manager):
        """Test memory cleanup on CPU."""
        mock_malloc_trim = Mock()
        mock_libc = Mock()
        mock_libc.malloc_trim.return_value = 0
        mock_cdll.return_value = mock_libc
        
        cpu_manager.clean_memory()
        
        mock_gc.assert_called_once()
        mock_cdll.assert_called_once_with("libc.so.6")
        mock_libc.malloc_trim.assert_called_once_with(0)
    
    @patch("gc.collect")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_clean_memory_cuda(self, mock_empty_cache, mock_cuda_avail, mock_gc, cuda_manager):
        """Test memory cleanup on CUDA."""
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libc = Mock()
            mock_libc.malloc_trim.return_value = 0
            mock_cdll.return_value = mock_libc
            
            cuda_manager.clean_memory()
        
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()
    
    @patch("gc.collect")
    @patch("ctypes.CDLL", side_effect=Exception("No libc"))
    def test_clean_memory_libc_error(self, mock_cdll, mock_gc, cpu_manager):
        """Test cleanup handles libc errors gracefully."""
        # Should not raise
        cpu_manager.clean_memory()
        mock_gc.assert_called_once()
    
    @patch("psutil.virtual_memory")
    @patch("torch.cuda.is_available", return_value=False)
    def test_log_memory_stats(self, mock_cuda, mock_vm, cpu_manager):
        """Test logging memory statistics."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_vm.return_value = mock_memory
        
        with patch("ommi_llm.utils.memory.logger") as mock_logger:
            cpu_manager.log_memory_stats(prefix="Test: ")
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Test:" in log_message
            assert "RAM:" in log_message
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    @patch("psutil.virtual_memory")
    def test_log_memory_stats_with_vram(
        self, mock_vm, mock_allocated, mock_props, mock_cuda, cuda_manager
    ):
        """Test logging memory stats with VRAM."""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_vm.return_value = mock_memory
        
        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024**3
        mock_props.return_value = mock_device_props
        mock_allocated.return_value = 2 * 1024**3
        
        with patch("ommi_llm.utils.memory.logger") as mock_logger:
            cuda_manager.log_memory_stats()
            log_message = mock_logger.info.call_args[0][0]
            assert "VRAM:" in log_message
    
    def test_estimate_peak_memory(self, cpu_manager):
        """Test peak memory estimation."""
        peak = cpu_manager.estimate_peak_memory(
            num_layers=32,
            hidden_size=4096,
            batch_size=1,
            seq_length=1024,
        )
        
        # Peak should be positive
        assert peak > 0
        
        # Larger models should have higher estimates
        peak_small = cpu_manager.estimate_peak_memory(
            num_layers=12,
            hidden_size=768,
            batch_size=1,
            seq_length=512,
        )
        peak_large = cpu_manager.estimate_peak_memory(
            num_layers=80,
            hidden_size=8192,
            batch_size=1,
            seq_length=2048,
        )
        
        assert peak_large > peak_small
    
    def test_estimate_peak_memory_different_batch_sizes(self, cpu_manager):
        """Test memory estimation with different batch sizes."""
        peak_1 = cpu_manager.estimate_peak_memory(
            num_layers=32, hidden_size=4096, batch_size=1, seq_length=1024
        )
        peak_4 = cpu_manager.estimate_peak_memory(
            num_layers=32, hidden_size=4096, batch_size=4, seq_length=1024
        )
        
        assert peak_4 > peak_1


class TestGlobalCleanMemory:
    """Test global clean_memory function."""
    
    @patch("gc.collect")
    @patch("ctypes.CDLL")
    @patch("torch.cuda.is_available", return_value=False)
    def test_clean_memory_global_cpu(self, mock_cuda, mock_cdll, mock_gc):
        """Test global clean memory on CPU."""
        mock_libc = Mock()
        mock_libc.malloc_trim.return_value = 0
        mock_cdll.return_value = mock_libc
        
        clean_memory()
        
        mock_gc.assert_called_once()
        mock_cdll.assert_called_once_with("libc.so.6")
    
    @patch("gc.collect")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_clean_memory_global_cuda(self, mock_empty, mock_cuda, mock_gc):
        """Test global clean memory on CUDA."""
        with patch("ctypes.CDLL") as mock_cdll:
            mock_libc = Mock()
            mock_libc.malloc_trim.return_value = 0
            mock_cdll.return_value = mock_libc
            
            clean_memory()
        
        mock_gc.assert_called_once()
        mock_empty.assert_called_once()
    
    @patch("gc.collect")
    @patch("ctypes.CDLL", side_effect=Exception("Library error"))
    @patch("torch.cuda.is_available", return_value=False)
    def test_clean_memory_handles_errors(self, mock_cuda, mock_cdll, mock_gc):
        """Test clean memory handles errors gracefully."""
        # Should not raise
        clean_memory()
        mock_gc.assert_called_once()
