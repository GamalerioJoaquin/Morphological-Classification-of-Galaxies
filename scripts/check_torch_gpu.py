"""Fail clearly when PyTorch cannot use an NVIDIA CUDA device."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime in PyTorch: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "ERROR: This PyTorch build cannot use CUDA. Install "
            "requirements-gpu.txt and restart the Jupyter kernel.",
            file=sys.stderr,
        )
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    sample = torch.ones((1024, 1024), device="cuda")
    print(f"CUDA tensor check: {float(sample.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
