"""Model loading and workload initialization.

This module owns:
1. Instantiating supported model architectures in evaluation mode (`eval()`).
2. Generating synthetic input tensors with deterministic dimensions.
3. Isolating model setup from the actual timing loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class ModelWorkload:
    """An initialized model and its synthetic input tensor, ready for benchmarking."""

    name: str
    model: Any
    dummy_input: Any
    batch_size: int
    input_shape: Tuple[int, ...]


def load_model(
    model_name: str = "mobilenet_v3_small",
    batch_size: int = 1,
    pretrained: bool = False,
    device: str = "cpu",
) -> ModelWorkload:
    """Load a model and generate matching synthetic input for inference benchmarking.

    Important Benchmarking Rationale:
    1. `pretrained=False` (default): Deep learning inference latency is governed by
       computational graph topology (GEMM dimensions, layer depth, tensor strides,
       and memory allocations). Latency is numerically identical whether weights
       are random or pretrained, but avoiding network downloads guarantees deterministic,
       offline, zero-friction reproducibility.
    2. `model.eval()`: Disables training-specific operations such as Dropout (which
       drops activations probabilistically) and forces BatchNorm to use running
       mean/variance rather than computing batch statistics over the inputs.
    3. Synthetic input tensors: Using synthetic in-memory tensors isolates model
       execution time from disk I/O, image decoding (e.g. libjpeg/Pillow), and
       preprocessing pipelines.

    Args:
        model_name: Identifier of the model (supported: 'mobilenet_v3_small').
        batch_size: Number of concurrent input samples.
        pretrained: If True, attempts to download official pretrained weights.
        device: Device to place the model on (default: 'cpu').

    Returns:
        ModelWorkload containing the configured model and input tensor.

    Raises:
        ValueError: If model_name is unsupported or batch_size < 1.
        ImportError: If PyTorch or TorchVision is not installed.
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be >= 1, got {batch_size}")

    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch (not installed) is required for running benchmarks."
        ) from e

    normalized_name = model_name.lower().strip()

    if normalized_name in ("mobilenet_v3_small", "mobilenet_v3"):
        try:
            import torchvision.models as tv_models
        except ImportError as e:
            raise ImportError(
                "torchvision is required for MobileNetV3."
            ) from e

        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)

        # Standard ImageNet input: (B, 3, 224, 224)
        input_shape = (batch_size, 3, 224, 224)
        dummy_input = torch.randn(input_shape, dtype=torch.float32, device=device)

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Phase 1 supports: ['mobilenet_v3_small']."
        )

    # Set evaluation mode and target device
    model.eval()
    model.to(device)

    return ModelWorkload(
        name=normalized_name,
        model=model,
        dummy_input=dummy_input,
        batch_size=batch_size,
        input_shape=input_shape,
    )

