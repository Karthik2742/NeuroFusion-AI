"""
NeuroFusion AI - GPU Monitoring & Benchmarking Utilities
Tracks GPU usage, memory consumption, and inference benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger


@dataclass
class GPUStats:
    """Container for GPU statistics."""
    device_name: str = "CPU"
    device_index: int = -1
    total_memory_gb: float = 0.0
    allocated_memory_gb: float = 0.0
    cached_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: Optional[float] = None
    cuda_version: str = "N/A"
    is_available: bool = False


@dataclass
class BenchmarkResult:
    """Container for inference/training benchmark results."""
    operation: str = ""
    total_time_sec: float = 0.0
    avg_time_sec: float = 0.0
    peak_memory_gb: float = 0.0
    throughput: float = 0.0
    device: str = "CPU"
    extra: dict = field(default_factory=dict)


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available compute device.
    
    Args:
        preference: 'auto', 'cpu', 'cuda', or 'cuda:N'
        
    Returns:
        torch.device object
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    elif preference.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(preference)
            idx = device.index or 0
            logger.info(f"Using GPU: {torch.cuda.get_device_name(idx)}")
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def get_gpu_stats(device_index: int = 0) -> GPUStats:
    """
    Get current GPU statistics.
    
    Args:
        device_index: GPU device index
        
    Returns:
        GPUStats object with current GPU state
    """
    stats = GPUStats()

    if not torch.cuda.is_available():
        return stats

    stats.is_available = True
    stats.device_index = device_index
    stats.device_name = torch.cuda.get_device_name(device_index)
    stats.cuda_version = torch.version.cuda or "N/A"

    # Memory stats (in GB)
    total = torch.cuda.get_device_properties(device_index).total_mem
    allocated = torch.cuda.memory_allocated(device_index)
    cached = torch.cuda.memory_reserved(device_index)

    stats.total_memory_gb = round(total / (1024**3), 2)
    stats.allocated_memory_gb = round(allocated / (1024**3), 2)
    stats.cached_memory_gb = round(cached / (1024**3), 2)
    stats.free_memory_gb = round((total - allocated) / (1024**3), 2)

    if total > 0:
        stats.utilization_percent = round((allocated / total) * 100, 1)

    return stats


def clear_gpu_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")


class InferenceTimer:
    """
    Context manager for timing GPU/CPU operations with memory tracking.
    
    Usage:
        with InferenceTimer("Enhancement") as timer:
            result = model(input)
        print(timer.result)
    """

    def __init__(self, operation_name: str = "inference", device: str = "auto"):
        self.operation = operation_name
        self.device_str = device
        self.result = BenchmarkResult(operation=operation_name)
        self._start_time: float = 0.0
        self._start_memory: float = 0.0

    def __enter__(self) -> InferenceTimer:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._start_memory = torch.cuda.memory_allocated()

        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._start_time
        self.result.total_time_sec = round(elapsed, 4)
        self.result.avg_time_sec = round(elapsed, 4)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
            self.result.peak_memory_gb = round(peak / (1024**3), 3)
            self.result.device = torch.cuda.get_device_name(0)
        else:
            self.result.device = "CPU"

        logger.info(
            f"[{self.operation}] Time: {self.result.total_time_sec:.3f}s | "
            f"Peak Memory: {self.result.peak_memory_gb:.3f} GB | "
            f"Device: {self.result.device}"
        )


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: Optional[torch.device] = None,
) -> BenchmarkResult:
    """
    Benchmark a PyTorch model's inference performance.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Shape of the input tensor
        num_runs: Number of timed inference runs
        warmup_runs: Number of warmup runs (not timed)
        device: Device to run on
        
    Returns:
        BenchmarkResult with timing and memory stats
    """
    if device is None:
        device = get_device()

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    result = BenchmarkResult(
        operation="model_benchmark",
        device=str(device),
    )

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    result.total_time_sec = round(sum(times), 4)
    result.avg_time_sec = round(sum(times) / len(times), 4)
    result.throughput = round(num_runs / sum(times), 2)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        result.peak_memory_gb = round(peak / (1024**3), 3)

    result.extra = {
        "num_runs": num_runs,
        "input_shape": list(input_shape),
        "min_time": round(min(times), 4),
        "max_time": round(max(times), 4),
        "std_time": round(
            (sum((t - result.avg_time_sec) ** 2 for t in times) / len(times)) ** 0.5,
            4,
        ),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    logger.info(
        f"Benchmark: avg={result.avg_time_sec:.4f}s | "
        f"throughput={result.throughput:.1f} samples/s | "
        f"peak_mem={result.peak_memory_gb:.3f} GB | "
        f"params={result.extra['num_parameters']:,}"
    )

    return result
