"""CUDA backend: wraps the compiled ops_cuda.cu shared library via ctypes.

This module loads libops_cuda.so (compiled with nvcc from csrc/ops_cuda.cu)
and provides the same Python API as ops_cpu.py.  All data stays in GPU
device memory between operations — the Python side only holds opaque
ctypes pointers (no CuPy dependency).

Build the library:
    cd picotorch/csrc
    nvcc -shared -Xcompiler -fPIC -O2 -o libops_cuda.so ops_cuda.cu

If the .so is not found, this module raises ImportError so the dispatcher
can fall back to CPU gracefully.
"""

import ctypes
import os
import numpy as np

# ── Load the CUDA shared library ────────────────────────────────

_dir = os.path.dirname(os.path.abspath(__file__))
_so_path = os.path.join(_dir, "csrc", "libops_cuda.so")

if not os.path.isfile(_so_path):
    raise ImportError(
        f"CUDA ops library not found at {_so_path}. "
        f"Build it with: cd picotorch/csrc && "
        f"nvcc -shared -Xcompiler -fPIC -O2 -o libops_cuda.so ops_cuda.cu"
    )

_lib = ctypes.cdll.LoadLibrary(_so_path)

# ── ctypes aliases ──────────────────────────────────────────────

_c_float = ctypes.c_float
_c_int   = ctypes.c_int
_float_p = ctypes.POINTER(_c_float)
_void_p  = ctypes.c_void_p

# ── Declare argtypes / restype for every exported function ──────

# Element-wise
_lib.pt_cuda_add.argtypes = [_float_p, _float_p, _float_p, _c_float, _c_int]
_lib.pt_cuda_add.restype  = None

_lib.pt_cuda_mul.argtypes = [_float_p, _float_p, _float_p, _c_int]
_lib.pt_cuda_mul.restype  = None

_lib.pt_cuda_pow_scalar.argtypes = [_float_p, _float_p, _c_float, _c_int]
_lib.pt_cuda_pow_scalar.restype  = None

_lib.pt_cuda_relu.argtypes = [_float_p, _float_p, _c_int]
_lib.pt_cuda_relu.restype  = None

_lib.pt_cuda_relu_backward.argtypes = [_float_p, _float_p, _float_p, _c_int]
_lib.pt_cuda_relu_backward.restype  = None

# Reduction
_lib.pt_cuda_sum.argtypes = [_float_p, _float_p, _c_int]
_lib.pt_cuda_sum.restype  = None

# Matmul
_lib.pt_cuda_matmul.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
_lib.pt_cuda_matmul.restype  = None

_lib.pt_cuda_matmul_backward_a.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
_lib.pt_cuda_matmul_backward_a.restype  = None

_lib.pt_cuda_matmul_backward_b.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
_lib.pt_cuda_matmul_backward_b.restype  = None

# Utility
_lib.pt_cuda_fill.argtypes = [_float_p, _c_float, _c_int]
_lib.pt_cuda_fill.restype  = None

_lib.pt_cuda_add_inplace.argtypes = [_float_p, _float_p, _c_int]
_lib.pt_cuda_add_inplace.restype  = None

_lib.pt_cuda_add_scalar.argtypes = [_float_p, _float_p, _c_float, _c_int]
_lib.pt_cuda_add_scalar.restype  = None

# Memory management
_lib.pt_cuda_malloc.argtypes = [_c_int]
_lib.pt_cuda_malloc.restype  = _float_p

_lib.pt_cuda_free.argtypes = [_float_p]
_lib.pt_cuda_free.restype  = None

_lib.pt_cuda_memcpy_h2d.argtypes = [_float_p, _float_p, _c_int]
_lib.pt_cuda_memcpy_h2d.restype  = None

_lib.pt_cuda_memcpy_d2h.argtypes = [_float_p, _float_p, _c_int]
_lib.pt_cuda_memcpy_d2h.restype  = None


# ── Device memory wrapper ──────────────────────────────────────

class CudaArray:
    """Thin wrapper around a device float* pointer with size metadata.

    This replaces CuPy arrays — we manage GPU memory ourselves via
    the CUDA runtime calls exposed through ops_cuda.cu.
    """

    __slots__ = ('ptr', '_size')

    def __init__(self, ptr, size):
        self.ptr  = ptr     # ctypes float* pointing to device memory
        self._size = size

    @property
    def size(self):
        return self._size

    def __del__(self):
        if self.ptr is not None:
            try:
                _lib.pt_cuda_free(self.ptr)
            except Exception:
                pass
            self.ptr = None


def _device_alloc(n):
    """Allocate n floats on the GPU, return a CudaArray."""
    ptr = _lib.pt_cuda_malloc(_c_int(n))
    return CudaArray(ptr, n)


def _host_ptr(arr):
    """Get a ctypes float* from a contiguous numpy float32 array."""
    return arr.ctypes.data_as(_float_p)


# ── Public API (mirrors ops_cpu.py signatures) ─────────────────
#
# Every function takes/returns CudaArray objects (device memory).
# The dispatcher and Tensor class treat these as opaque "data" —
# same role as numpy arrays on the CPU side.

def add(a, b, alpha=1.0):
    """Element-wise: out = a + alpha * b"""
    out = _device_alloc(a.size)
    _lib.pt_cuda_add(a.ptr, b.ptr, out.ptr,
                     _c_float(alpha), _c_int(a.size))
    return out


def mul(a, b):
    """Element-wise: out = a * b"""
    out = _device_alloc(a.size)
    _lib.pt_cuda_mul(a.ptr, b.ptr, out.ptr, _c_int(a.size))
    return out


def matmul(a, b, m, n, p):
    """Matrix multiply: out[m,p] = a[m,n] @ b[n,p]"""
    out = _device_alloc(m * p)
    _lib.pt_cuda_matmul(a.ptr, b.ptr, out.ptr,
                        _c_int(m), _c_int(n), _c_int(p))
    return out


def matmul_backward_a(grad_out, b, m, n, p):
    """grad_a[m,n] = grad_out[m,p] @ B^T"""
    grad_a = _device_alloc(m * n)
    _lib.pt_cuda_matmul_backward_a(grad_out.ptr, b.ptr, grad_a.ptr,
                                   _c_int(m), _c_int(n), _c_int(p))
    return grad_a


def matmul_backward_b(a, grad_out, m, n, p):
    """grad_b[n,p] = A^T @ grad_out"""
    grad_b = _device_alloc(n * p)
    _lib.pt_cuda_matmul_backward_b(a.ptr, grad_out.ptr, grad_b.ptr,
                                   _c_int(m), _c_int(n), _c_int(p))
    return grad_b


def pow_scalar(a, exponent):
    """Element-wise: out = a ^ exponent"""
    out = _device_alloc(a.size)
    _lib.pt_cuda_pow_scalar(a.ptr, out.ptr,
                            _c_float(exponent), _c_int(a.size))
    return out


def relu(a):
    """Element-wise: out = max(0, a)"""
    out = _device_alloc(a.size)
    _lib.pt_cuda_relu(a.ptr, out.ptr, _c_int(a.size))
    return out


def relu_backward(a, grad_out):
    """ReLU backward: grad_in = (a > 0) ? grad_out : 0"""
    grad_in = _device_alloc(a.size)
    _lib.pt_cuda_relu_backward(a.ptr, grad_out.ptr, grad_in.ptr,
                               _c_int(a.size))
    return grad_in


def sum_all(a):
    """Reduction: sum of all elements."""
    out = _device_alloc(1)
    _lib.pt_cuda_sum(a.ptr, out.ptr, _c_int(a.size))
    return out


def fill(n, value):
    """Create device array filled with value."""
    out = _device_alloc(n)
    _lib.pt_cuda_fill(out.ptr, _c_float(value), _c_int(n))
    return out


def add_inplace(out, a):
    """In-place: out += a"""
    _lib.pt_cuda_add_inplace(out.ptr, a.ptr, _c_int(out.size))
    return out


def zeros(n):
    """Create zero-filled device array."""
    return fill(n, 0.0)


def ones(n):
    """Create ones-filled device array."""
    return fill(n, 1.0)


def to_device(np_array):
    """Transfer numpy array (host) → GPU (device). Returns CudaArray."""
    arr = np.ascontiguousarray(np_array, dtype=np.float32)
    n = arr.size
    ca = _device_alloc(n)
    _lib.pt_cuda_memcpy_h2d(ca.ptr, _host_ptr(arr), _c_int(n))
    return ca


def to_numpy(cuda_array):
    """Transfer CudaArray (device) → numpy array (host)."""
    n = cuda_array.size
    out = np.empty(n, dtype=np.float32)
    _lib.pt_cuda_memcpy_d2h(_host_ptr(out), cuda_array.ptr, _c_int(n))
    return out
