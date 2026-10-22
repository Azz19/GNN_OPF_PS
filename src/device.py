"""Device selection shared by the experiment runners.

Same scripts run on this CPU box and on Hay's GPU workstation:
  --device auto   -> cuda if available, else cpu   (default)
  --device cuda   -> fail loudly if no CUDA device is present
  --device cpu    -> force CPU

Models stay device-agnostic: every temporary tensor is derived from an
input tensor (torch.zeros_like / index_add_), so moving the model and the
input tensors is sufficient.
"""
import torch


def resolve_device(choice="auto"):
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device choice {choice!r}; want auto|cpu|cuda")


def to_dev(arr, dev, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=dev)
