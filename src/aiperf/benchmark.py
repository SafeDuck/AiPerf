"""Isolated benchmark runner for AI inference workloads.

This module owns:
1. Warmup iteration execution (discarding initial execution transients).
2. Monotonic high-resolution timing of pure inference forward passes.
3. Invoking the metrics module to consolidate latency and throughput results.
"""

from __future__ import annotations

import time
from typing import Optional

from aiperf.metrics import BenchmarkMetrics, calculate_latency_stats, calculate_throughput
from aiperf.models import ModelWorkload, load_model


def run_benchmark(
    workload: Optional[ModelWorkload] = None,
    model_name: str = "mobilenet_v3_small",
    batch_size: int = 1,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 50,
    num_threads: Optional[int] = None,
) -> BenchmarkMetrics:
    """Execute the inference benchmarking protocol.

    Methodological Steps:
    1. Thread Configuration: If num_threads is specified, configure PyTorch's intra-op
       thread pool.
    2. Warmup Phase: Execute `warmup_iterations` passes.
       - Discards initial kernel initialization, memory allocation, and CPU cache warming.
       - These passes are NOT timed or included in the final statistics.
    3. Benchmark Phase: Execute `benchmark_iterations` passes.
       - Uses `time.perf_counter_ns()` for high-resolution monotonic wall-clock timing.
       - Executes inside `torch.inference_mode()` to eliminate autograd graph construction
         and tensor version counter updates.
    4. Metrics Aggregation: Raw latencies (converted from ns to ms) are passed to the
       pure `metrics` module to compute statistical moments and throughput.

    Args:
        workload: Pre-initialized ModelWorkload. If None, loaded using model_name & batch_size.
        model_name: Model name if workload is None.
        batch_size: Batch size if workload is None.
        warmup_iterations: Number of initial iterations to discard (default: 10).
        benchmark_iterations: Number of timed iterations to record (default: 50).
        num_threads: Optional number of intra-op threads to set for PyTorch.

    Returns:
        BenchmarkMetrics containing LatencyStats, throughput, batch_size, and raw samples.

    Raises:
        ValueError: If iteration counts are invalid (< 1).
    """
    if warmup_iterations < 0:
        raise ValueError(f"Warmup iterations must be >= 0, got {warmup_iterations}")
    if benchmark_iterations < 1:
        raise ValueError(f"Benchmark iterations must be >= 1, got {benchmark_iterations}")

    import torch

    # Configure PyTorch CPU intra-op threads if explicitly requested
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {num_threads}")
        torch.set_num_threads(num_threads)

    if workload is None:
        workload = load_model(model_name=model_name, batch_size=batch_size)

    model = workload.model
    dummy_input = workload.dummy_input

    # Step 1: Warmup Phase
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Step 2: Timed Benchmark Loop
    raw_latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(benchmark_iterations):
            t_start = time.perf_counter_ns()
            _ = model(dummy_input)
            t_end = time.perf_counter_ns()

            latency_ms = (t_end - t_start) / 1_000_000.0
            raw_latencies_ms.append(latency_ms)

    # Step 3: Compute Metrics
    latency_stats = calculate_latency_stats(raw_latencies_ms)
    throughput = calculate_throughput(workload.batch_size, latency_stats.mean_ms)

    return BenchmarkMetrics(
        latency=latency_stats,
        throughput_samples_per_sec=throughput,
        batch_size=workload.batch_size,
        raw_latencies_ms=tuple(raw_latencies_ms),
    )

