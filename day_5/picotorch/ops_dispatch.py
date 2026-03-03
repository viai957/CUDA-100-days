"""Dispatches tensor operations to the correct backend (CPU or CUDA)."""
import numpy as np
from picotorch import ops_cpu

# Conditional CUDA import
try:
    from picotorch import ops_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def _get_backend(device):
    """Return the ops module for the given device."""
    if device == 'cpu':
        return ops_cpu
    elif device == 'cuda':
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CUDA backend requested but CuPy is not installed. "
                "Install with: pip install cupy-cuda12x"
            )
        return ops_cuda
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'cuda'.")


def add(a_data, b_data, device, alpha=1.0):
    return _get_backend(device).add(a_data, b_data, alpha)


def mul(a_data, b_data, device):
    return _get_backend(device).mul(a_data, b_data)


def matmul(a_data, b_data, device, m, n, p):
    return _get_backend(device).matmul(a_data, b_data, m, n, p)


def matmul_backward_a(grad_out, b_data, device, m, n, p):
    return _get_backend(device).matmul_backward_a(grad_out, b_data, m, n, p)


def matmul_backward_b(a_data, grad_out, device, m, n, p):
    return _get_backend(device).matmul_backward_b(a_data, grad_out, m, n, p)


def pow_scalar(a_data, exponent, device):
    return _get_backend(device).pow_scalar(a_data, exponent)


def relu(a_data, device):
    return _get_backend(device).relu(a_data)


def sum_all(a_data, device):
    return _get_backend(device).sum_all(a_data)


def fill(n, value, device):
    return _get_backend(device).fill(n, value)


def zeros(n, device):
    return _get_backend(device).zeros(n)


def ones(n, device):
    return _get_backend(device).ones(n)


def add_inplace(out_data, a_data, device):
    return _get_backend(device).add_inplace(out_data, a_data)


def to_device(data, src_device, dst_device):
    """Transfer data between devices."""
    if src_device == dst_device:
        return data
    if src_device == 'cpu' and dst_device == 'cuda':
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        return ops_cuda.to_device(data)
    if src_device == 'cuda' and dst_device == 'cpu':
        return ops_cuda.to_numpy(data)
    raise ValueError(f"Cannot transfer from {src_device} to {dst_device}")
