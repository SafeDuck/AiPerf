"""Statistical metrics calculation for inference performance.

This module is intentionally pure: it operates on raw numerical sequences
(latencies in milliseconds or seconds) and has zero dependencies on PyTorch,
hardware drivers, or presentation frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

@dataclass(frozen=True)
class LatencyStats:
    """Summary statistics for a series of latency measurements.

    All latency fields are reported in milliseconds (ms).
    """

    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    iterations: int

# frozen dataclass is used for data integrity (downstream code won't modify it) and thread saftey (threads can share it without locks)
@dataclass(frozen=True)
class BenchmarkMetrics:
    """Consolidated performance metrics for an inference benchmark run."""

    latency: LatencyStats
    throughput_samples_per_sec: float
    batch_size: int
    raw_latencies_ms: tuple[float, ...]


def calculate_latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Calculate descriptive statistics from a sequence of latency measurements in milliseconds.

    Args:
        latencies_ms: Sequence of measured iteration latencies in milliseconds.

    Returns:
        LatencyStats containing mean, min, max, standard deviation, and sample count.

    Raises:
        ValueError: If latencies_ms is empty or contains non-positive values.
    """
    if not latencies_ms:
        raise ValueError("Cannot calculate latency statistics for an empty sequence.")

    n = len(latencies_ms)
    min_val = min(latencies_ms)
    if min_val <= 0.0:
        raise ValueError("Latencies must be strictly positive numbers.")

    max_val = max(latencies_ms)
    mean_val = sum(latencies_ms) / n

    if n > 1:
        # Sample standard deviation (ddof=1) # ddof means delta degrees of freedom
        variance = sum((x - mean_val) ** 2 for x in latencies_ms) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    return LatencyStats(
        mean_ms=mean_val,
        min_ms=min_val,
        max_ms=max_val,
        std_ms=std_val,
        iterations=n,
    )


def calculate_throughput(batch_size: int, mean_latency_ms: float) -> float:
    """Calculate inference throughput in samples processed per second.

    Formula:
        throughput = batch_size / mean_latency_seconds
                   = batch_size / (mean_latency_ms / 1000.0)
                   = (batch_size * 1000.0) / mean_latency_ms

    Assumptions & Nuance:
        This calculates steady-state saturated throughput for synchronous,
        single-stream execution. It assumes back-to-back requests where the
        next batch begins immediately after the prior batch finishes.

    Args:
        batch_size: Number of concurrent input samples evaluated in each inference pass.
        mean_latency_ms: Arithmetic mean of batch latency in milliseconds.

    Returns:
        Throughput in samples per second.

    Raises:
        ValueError: If batch_size <= 0 or mean_latency_ms <= 0.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}")
    if mean_latency_ms <= 0.0:
        raise ValueError(f"Mean latency must be positive, got {mean_latency_ms}")

    return (batch_size * 1000.0) / mean_latency_ms

