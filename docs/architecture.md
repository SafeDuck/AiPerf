# AIPerf System Architecture

`AIPerf` is an AI Inference Performance Benchmarking & Optimization Toolkit designed to systematically measure, analyze, and optimize deep learning model inference across hardware configurations, runtimes, and execution parameters.

---

## 1. High-Level Vision & Design Philosophy

Rather than treating AI inference as a black box where models are executed with naive wall-clock timing, `AIPerf` treats inference as a systems engineering workload governed by computer architecture principles: memory hierarchy, cache locality, compute vs. memory bounds, thread scheduling, and hardware utilization.

### Core Architectural Principle: Unidirectional Separation of Concerns

```
Benchmark Specification (Config)
          │
          ▼
Benchmark Orchestration (Runner / Engine)
          │
          ├──> Model Management (Loader, Input Generator)
          └──> Environment Control (Threads, Precision, Device)
          │
          ▼
Pure Execution & Raw Measurement (Latency Samples)
          │
          ▼
Statistical Metric Extraction (P50, P90, P95, P99, Throughput)
          │
          ▼
Structured Result Record (Validation & Schema Definition)
          │
          ▼
Storage & Persistence (DuckDB / Structured Records)
          │
          ▼
Analysis & Recommendation Engine (Latency/Throughput/Efficiency Objectives)
          │
          ▼
Presentation (CLI Tables, Markdown Reports, Future Dashboards)
```

The fundamental design invariant is: **Dependencies flow downward.**
- The **Benchmark Engine** knows nothing about SQL, DuckDB, HTML generation, or CLI formatting.
- The **Metrics Collector** calculates statistical moments and percentiles on raw numerical arrays; it knows nothing about PyTorch or threads.
- The **Analysis Engine** consumes validated tabular records; it does not trigger inference runs.
- No component imports a layer above it, preventing circular dependencies and spaghetti coupling.

---

## 2. Module Responsibilities & Non-Responsibilities

To avoid bloated god-objects and blurry boundaries, every module in `AIPerf` has explicit responsibilities and strict prohibitions.

### `src/aiperf/config.py`
- **Owns:**
  - Parsing, schema definition, and validation of user experiment configurations (YAML, CLI flags).
  - Validation of parameter constraints (e.g., `batch_size > 0`, `iterations >= 1`, `threads > 0`, valid precision flags).
  - Strong typing via dataclasses or Pydantic.
- **Does NOT own:**
  - Running benchmarks.
  - Loading models or inspecting runtime hardware capabilities.
  - Formatting results.

### `src/aiperf/system.py`
- **Owns:**
  - Collection of reproducible hardware and software environment metadata.
  - CPU model, physical vs. logical core counts, available RAM, OS platform, Python version, PyTorch version, accelerator information (CUDA device name, compute capability if present).
  - Standardizing system telemetry into an immutable metadata dictionary/object.
- **Does NOT own:**
  - Profiling process memory during inference (handled by telemetry sampling hooks).
  - Benchmark orchestration.

### `src/aiperf/models.py`
- **Owns:**
  - Encapsulating model acquisition, instantiation, evaluation mode setting (`eval()`), precision casting (`fp32`, `fp16`), and device placement.
  - Generating valid synthetic input tensors shaped strictly according to model expectations and configured batch sizes.
- **Does NOT own:**
  - Timing inference loops.
  - Managing thread pools.
  - Recording latency statistics.

### `src/aiperf/metrics.py`
- **Owns:**
  - Pure mathematical and statistical processing of raw latency arrays.
  - Percentiles (P50, P90, P95, P99), arithmetic mean, minimum, maximum, sample standard deviation.
  - Derivation of batch throughput (samples/second) given batch size and latency samples.
- **Does NOT own:**
  - Collecting timestamps (it receives an array of floating-point seconds).
  - File I/O, database writes, or terminal formatting.

### `src/aiperf/benchmark.py`
- **Owns:**
  - Executing the measurement protocol:
    1. Setting intra-op/inter-op thread counts.
    2. Model warmup phase (discarding warmup measurements).
    3. Monotonic high-resolution timing of pure inference iterations.
    4. Guarding against common benchmarking traps (e.g., Python overhead, memory allocation in loops, GPU async completion).
  - Packing raw latencies, environment metadata, and configuration into an intermediate raw result record.
- **Does NOT own:**
  - Direct database persistence.
  - Plotting or terminal table rendering.
  - CLI argument parsing.

### `src/aiperf/cli.py`
- **Owns:**
  - CLI entry point parsing (via Typer).
  - Translating user commands into configuration objects.
  - Presenting benchmark execution progress and displaying human-readable output summaries (using Rich tables).
- **Does NOT own:**
  - Statistical calculations.
  - Low-level PyTorch tensor handling.

---

## 3. Benchmarking Methodology vs. Profiling

A central tenet of performance engineering is understanding the distinct roles of **benchmarking** and **profiling**.

| Dimension | Benchmarking | Profiling |
| :--- | :--- | :--- |
| **Core Question** | *"How fast is it?"* | *"Why is it taking this long?"* |
| **Primary Metric** | Wall-clock latency (P50, P99), Throughput (samples/sec) | Instruction counts, cache misses, kernel execution breakdown, memory bandwidth |
| **Measurement Overhead** | **Must be minimal** (nanosecond-level timer calls) so the measurement does not distort execution | **Non-zero overhead** (sampling or instrumentation) acceptable to inspect internal mechanics |
| **Primary Goal** | Establish baselines, detect regressions, compare hardware/configurations | Identify hotspots, bottlenecks (compute vs memory bound), pipeline stalls |
| **Typical Tools** | High-resolution monotonic timers (`time.perf_counter_ns`), warmup loops, statistical aggregation | Linux `perf`, `taskset`, `torch.profiler`, Nsight Systems/Compute |

### Why Naive Timing Is Flawed
A naive timing loop like:
```python
# FLAWED BENCHMARKING
t0 = time.time()
output = model(input_tensor)
t1 = time.time()
latency = t1 - t0
```
suffers from severe distortions:
1. **Cold Start & JIT/Kernel Compilation:** The first inference pass includes memory allocations, CUDA context initialization, and kernel graph setups. Including this skews the mean.
2. **Timer Resolution & Clock Drift:** `time.time()` tracks system clock time, which can drift, jump with NTP adjustments, and has coarse resolution on some operating systems. A monotonic nanosecond timer (`time.perf_counter_ns()`) is mandatory.
3. **GPU Asynchronous Dispatch:** In PyTorch, GPU kernel launches are asynchronous non-blocking operations queued on CUDA streams. A naive CPU timer stops when the kernel is queued, not when execution finishes on the streaming multiprocessors. `torch.cuda.synchronize()` or CUDA events are required for accelerators.
4. **Memory Allocation Noise:** Allocating new input or output tensors inside the timed loop measures the memory allocator and garbage collector, not the model computation.
5. **Mean vs. Tail Latency:** Reporting only the arithmetic mean hides pathological latency spikes (e.g., OS thread descheduling, GC pauses, cache invalidation). Reporting P50, P90, P95, and P99 is critical for production SLAs.

---

## 4. Technology Decisions and Tradeoff Analysis

### Language: Python (Orchestration) with C++ (Targeted Microbenchmarks Later)
- **Choice:** Python for high-level experiment control, PyTorch API integration, DuckDB querying, and CLI tooling.
- **Why:** PyTorch's heavy compute kernels (ATen, BLAS, cuDNN, oneDNN) are already compiled C++/CUDA binaries. The execution time of deep neural networks is dominated by underlying GEMM and convolution kernels, not Python byte-code, provided the timed loop does not execute excessive Python logic.
- **Alternatives Considered:** Pure C++ from Phase 0.
  - *Tradeoff:* Starting in pure C++ with LibTorch increases compilation friction, slows down model download/testing cycles, and creates boilerplate without improving the core learning of inference tradeoffs (batching, threading, memory). We intentionally isolate C++ to Phase 8 as dedicated microbenchmarks (e.g., matrix multiply, OpenMP parallel scaling) where low-level cache and instruction behavior can be studied directly.

### Data Layer: DuckDB (Columnar Embedded Analytical Engine)
- **Choice:** DuckDB for persisting benchmark runs and analytical queries.
- **Why:** 
  1. **Embedded & Zero-Infrastructure:** Like SQLite, it runs in-process with zero daemon configuration, perfect for local CLI developer tooling and CI runs.
  2. **Columnar & Vectorized Execution:** Benchmark telemetry is inherently analytical (OLAP): calculating `AVG(latency_ms)`, `PERCENTILE_CONT(0.99)`, and groupings across `batch_size` and `thread_count`. Columnar storage is order-of-magnitude faster than row-based SQLite for analytical scans.
  3. **Data Interoperability:** Native export to Parquet, Arrow, and Pandas.
- **Alternatives Considered:**
  - *Flat JSON/CSV files:* Suffer from lack of schema enforcement, slow aggregation as dataset grows, and complex filtering logic.
  - *PostgreSQL:* Requires external service administration, network connection setup, and credentials, introducing unnecessary friction for an experimental toolkit.

### Interface: CLI-First (Typer + Rich)
- **Choice:** Terminal-first CLI using `Typer` and `Rich`.
- **Why:** Automated performance experimentation belongs in automated scripts, CI pipelines, and SSH remote servers. A GUI creates premature coupling, obscures the data flow, and slows iteration.

---

## 5. System Data Flow

```
[User Command: aiperf benchmark --config config.yaml]
                        │
                        ▼
           [Config Loader & Validator]
                        │ (Validated Config Object)
                        ▼
             [Benchmark Controller]
                        │
    ┌───────────────────┴───────────────────┐
    ▼                                       ▼
[System Inspector]                   [Model Loader]
(CPU, RAM, OS, PyTorch)              (MobileNetV3 / DistilBERT)
    │                                       │
    │                              (Model + Warmup Tensors)
    │                                       │
    └───────────────────┬───────────────────┘
                        ▼
             [Benchmark Runner]
      ├── Thread Environment Setup (torch.set_num_threads)
      ├── Warmup Iterations (N runs, unrecorded)
      └── Timed Iterations (M runs, high-res monotonic timing)
                        │
                        ▼
                [Raw Latency Vector]
                        │
                        ▼
               [Metrics Calculator]
         (P50, P90, P95, P99, Mean, Throughput)
                        │
                        ▼
             [Experiment Record Object]
                        │
    ┌───────────────────┴───────────────────┐
    ▼                                       ▼
[Results Store: DuckDB/JSON]          [Console Summary: Rich Table]
```

