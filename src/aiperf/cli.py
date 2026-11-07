"""Command-Line Interface for AIPerf.

Provides an intuitive CLI for launching inference benchmarks and inspecting
performance metrics.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from aiperf.benchmark import run_benchmark


def create_parser() -> argparse.ArgumentParser:
    """Build the argument parser for AIPerf CLI."""
    parser = argparse.ArgumentParser(
        prog="aiperf",
        description="AIPerf: AI Inference Performance Benchmarking & Optimization Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # `benchmark` subcommand
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run an inference benchmark on a specified model",
    )
    bench_parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v3_small",
        help="Model architecture name (default: mobilenet_v3_small)",
    )
    bench_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    bench_parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of untimed warmup iterations (default: 10)",
    )
    bench_parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed benchmark iterations (default: 50)",
    )
    bench_parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of intra-op threads for PyTorch CPU execution (default: auto)",
    )

    return parser


def format_results_table(
    model_name: str,
    batch_size: int,
    threads: Optional[int],
    warmup: int,
    iterations: int,
    mean_ms: float,
    min_ms: float,
    max_ms: float,
    std_ms: float,
    throughput: float,
) -> str:
    """Format benchmark results into a clean, human-readable terminal table."""
    border = "=" * 62
    divider = "-" * 62

    lines = [
        "",
        border,
        "             AIPerf Inference Benchmark Results             ",
        border,
        f"  {'Model:':<26} {model_name}",
        f"  {'Batch Size:':<26} {batch_size}",
        f"  {'Threads:':<26} {threads if threads is not None else 'Default (Auto)'}",
        f"  {'Warmup Passes:':<26} {warmup}",
        f"  {'Measured Iterations:':<26} {iterations}",
        divider,
        "  LATENCY METRICS (Pure Inference)",
        divider,
        f"  {'Mean Latency:':<26} {mean_ms:>10.3f} ms",
        f"  {'Min Latency:':<26} {min_ms:>10.3f} ms",
        f"  {'Max Latency:':<26} {max_ms:>10.3f} ms",
        f"  {'Std Dev:':<26} {std_ms:>10.3f} ms",
        divider,
        "  THROUGHPUT METRIC",
        divider,
        f"  {'Throughput:':<26} {throughput:>10.2f} samples/sec",
        border,
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for AIPerf."""
    parser = create_parser()
    args = parser.parse_args(argv) 

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "benchmark":
        print(f"[AIPerf] Preparing workload: {args.model} (batch_size={args.batch_size})...")
        try:
            metrics = run_benchmark(
                model_name=args.model,
                batch_size=args.batch_size,
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
                num_threads=args.threads,
            )
        except Exception as e:
            print(f"[AIPerf] Error running benchmark: {e}", file=sys.stderr) 
            return 1

        output = format_results_table(
            model_name=args.model,
            batch_size=args.batch_size,
            threads=args.threads,
            warmup=args.warmup,
            iterations=args.iterations,
            mean_ms=metrics.latency.mean_ms,
            min_ms=metrics.latency.min_ms,
            max_ms=metrics.latency.max_ms,
            std_ms=metrics.latency.std_ms,
            throughput=metrics.throughput_samples_per_sec,
        )
        print(output)
        return 0

    return 0


def app() -> None:
    """Wrapper function referenced by pyproject.toml script entrypoint."""
    sys.exit(main())


if __name__ == "__main__":
    app()

