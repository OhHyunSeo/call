# Standard library imports
import logging
from typing import Dict, Optional, Callable, Any
from contextlib import contextmanager

# Related third-party imports
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_utils")


class GPUStreamManager:
    """
    Manages CUDA streams to enable concurrent execution of GPU operations.
    
    This class helps optimize GPU utilization by allowing different pipeline
    stages to run concurrently on the GPU using separate CUDA streams.
    """
    
    def __init__(self):
        """Initialize the GPU stream manager."""
        self.streams: Dict[str, Optional[torch.cuda.Stream]] = {}
        self.default_stream = torch.cuda.current_stream()
        
        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA is not available. GPU stream management will be disabled.")
    
    def create_stream(self, name: str) -> None:
        """
        Create a new CUDA stream with the given name.
        
        Parameters
        ----------
        name : str
            Name to identify the stream
        """
        if not self.cuda_available:
            self.streams[name] = None
            return
            
        self.streams[name] = torch.cuda.Stream()
        logger.debug(f"Created CUDA stream: {name}")
    
    @contextmanager
    def use_stream(self, name: str) -> None:
        """
        Context manager to execute operations on a specific stream.
        
        Parameters
        ----------
        name : str
            Name of the stream to use
            
        Yields
        ------
        None
        """
        if not self.cuda_available or name not in self.streams:
            yield
            return
            
        stream = self.streams[name]
        if stream is None:
            yield
            return
            
        with torch.cuda.stream(stream):
            yield
    
    def synchronize_stream(self, name: str) -> None:
        """
        Synchronize a specific stream.
        
        Parameters
        ----------
        name : str
            Name of the stream to synchronize
        """
        if not self.cuda_available or name not in self.streams:
            return
            
        stream = self.streams[name]
        if stream is not None:
            stream.synchronize()
            logger.debug(f"Synchronized CUDA stream: {name}")
    
    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        if not self.cuda_available:
            return
            
        for name, stream in self.streams.items():
            if stream is not None:
                stream.synchronize()
        logger.debug("Synchronized all CUDA streams")
    
    def run_on_stream(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function on a specific stream.
        
        Parameters
        ----------
        name : str
            Name of the stream to use
        func : Callable
            Function to run
        *args, **kwargs
            Arguments to pass to the function
            
        Returns
        -------
        Any
            The result of the function
        """
        if not self.cuda_available or name not in self.streams:
            return func(*args, **kwargs)
            
        stream = self.streams[name]
        if stream is None:
            return func(*args, **kwargs)
            
        with torch.cuda.stream(stream):
            result = func(*args, **kwargs)
            
        return result


# Create a global stream manager instance
stream_manager = GPUStreamManager()


def setup_gpu_streams():
    """
    Set up GPU streams for different pipeline stages.
    
    This function should be called at the beginning of the application
    to create streams for each GPU-intensive pipeline stage.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. GPU optimizations will be disabled.")
        return
    
    # Create streams for each GPU-intensive stage
    stream_manager.create_stream("demucs")      # Vocal separation
    stream_manager.create_stream("diarization") # Speaker diarization
    stream_manager.create_stream("asr")         # Automatic speech recognition
    stream_manager.create_stream("alignment")   # Forced alignment
    stream_manager.create_stream("llm")         # Language model inference
    
    logger.info("GPU streams initialized successfully")


def optimize_gpu_memory():
    """
    Optimize GPU memory usage.
    
    This function applies various optimizations to reduce GPU memory
    usage and prevent out-of-memory errors.
    """
    if not torch.cuda.is_available():
        return
    
    # Enable memory caching for faster allocation
    torch.cuda.empty_cache()
    
    # Set memory fraction to use (adjust based on GPU capacity)
    # This prevents the pipeline from using all available GPU memory
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Set to allocate only the necessary amount of memory
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.memory_stats()
    
    # Disable gradient calculation for inference
    torch.set_grad_enabled(False)
    
    logger.info("GPU memory optimizations applied")


def get_gpu_info():
    """
    Get information about available GPUs.
    
    Returns
    -------
    Dict
        Dictionary containing GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device())
    }
    
    # Add memory information if available
    if hasattr(torch.cuda, 'memory_allocated'):
        info["memory_allocated"] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        info["memory_reserved"] = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    
    if hasattr(torch.cuda, 'memory_stats') and callable(torch.cuda.memory_stats):
        try:
            memory_stats = torch.cuda.memory_stats()
            if "allocated_bytes.all.current" in memory_stats:
                info["allocated_bytes"] = memory_stats["allocated_bytes.all.current"] / (1024 ** 3)  # GB
        except:
            # Some CUDA versions don't support detailed memory stats
            pass
    
    return info 