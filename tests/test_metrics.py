"""Unit tests for the metrics calculation module."""

import math
import pytest

from aiperf.metrics import (
    LatencyStats,
    calculate_latency_stats,
    calculate_throughput,
)


def test_calculate_latency_stats_known_values() -> None:
    """Verify statistics against manually computed reference values."""
    # Samples: [10.0, 20.0, 30.0]
    # Mean: 20.0
    # Variance (ddof=1): ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = 200 / 2 = 100
    # Std: sqrt(100) = 10.0
    samples = [10.0, 20.0, 30.0]
    stats = calculate_latency_stats(samples)

    assert isinstance(stats, LatencyStats)
    assert stats.mean_ms == 20.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 30.0
    assert stats.std_ms == 10.0
    assert stats.iterations == 3


def test_calculate_latency_stats_single_sample() -> None:
    """A single sample should produce mean=value and std=0.0."""
    stats = calculate_latency_stats([15.5])
    assert stats.mean_ms == 15.5
    assert stats.min_ms == 15.5
    assert stats.max_ms == 15.5
    assert stats.std_ms == 0.0
    assert stats.iterations == 1


def test_calculate_latency_stats_empty_raises_error() -> None:
    """Passing an empty sequence must raise a ValueError."""
    with pytest.raises(ValueError, match="empty sequence"):
        calculate_latency_stats([])


def test_calculate_latency_stats_negative_or_zero_raises_error() -> None:
    """Non-positive latencies represent invalid physical measurements."""
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_latency_stats([10.0, 0.0, 5.0])

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_latency_stats([10.0, -1.0, 5.0])


def test_calculate_throughput_standard_case() -> None:
    """Verify throughput formula: (batch_size * 1000) / mean_latency_ms."""
    # Batch size 1, 10ms -> 100 inferences per second
    throughput = calculate_throughput(batch_size=1, mean_latency_ms=10.0)
    assert math.isclose(throughput, 100.0, rel_tol=1e-5)

    # Batch size 4, 20ms -> 200 samples per second
    throughput_b4 = calculate_throughput(batch_size=4, mean_latency_ms=20.0)
    assert math.isclose(throughput_b4, 200.0, rel_tol=1e-5)


def test_calculate_throughput_invalid_inputs() -> None:
    """Invalid batch size or non-positive latency must raise ValueError."""
    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        calculate_throughput(batch_size=0, mean_latency_ms=10.0)

    with pytest.raises(ValueError, match="Batch size must be a positive integer"):
        calculate_throughput(batch_size=-1, mean_latency_ms=10.0)

    with pytest.raises(ValueError, match="Mean latency must be positive"):
        calculate_throughput(batch_size=1, mean_latency_ms=0.0)

    with pytest.raises(ValueError, match="Mean latency must be positive"):
        calculate_throughput(batch_size=1, mean_latency_ms=-5.0)

