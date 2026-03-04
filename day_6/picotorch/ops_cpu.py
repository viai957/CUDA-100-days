"""CPU backend: calls the C shared library via ctypes (or cffi if available).

This module wraps every function declared in csrc/ops_cpu.h, returning
numpy float32 arrays.  The dispatcher (ops_dispatch.py) imports this module
for CPU tensor operations.
"""

import ctypes
import os
import numpy as np

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------

_lib = None

# Strategy 1: try cffi-compiled module
try:
    from picotorch._ops_cpu_cffi import lib as _cffi_lib, ffi as _cffi_ffi
    _use_cffi = True
except ImportError:
    _use_cffi = False

# Strategy 2: fall back to ctypes with the pre-built .dylib
if not _use_cffi:
    _dir = os.path.dirname(os.path.abspath(__file__))
    _dylib_path = os.path.join(_dir, "csrc", "libops_cpu.dylib")
    if not os.path.isfile(_dylib_path):
        raise RuntimeError(f"Cannot find CPU ops library at {_dylib_path}")
    _lib = ctypes.cdll.LoadLibrary(_dylib_path)

    # Convenience aliases
    _c_float = ctypes.c_float
    _c_int = ctypes.c_int
    _float_p = ctypes.POINTER(ctypes.c_float)

    # -- argtypes / restype declarations ------------------------------------

    # void pt_add(const float* a, const float* b, float* out, float alpha, int n)
    _lib.pt_add.argtypes = [_float_p, _float_p, _float_p, _c_float, _c_int]
    _lib.pt_add.restype = None

    # void pt_mul(const float* a, const float* b, float* out, int n)
    _lib.pt_mul.argtypes = [_float_p, _float_p, _float_p, _c_int]
    _lib.pt_mul.restype = None

    # void pt_matmul(const float* a, const float* b, float* out, int m, int n, int p)
    _lib.pt_matmul.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
    _lib.pt_matmul.restype = None

    # void pt_matmul_backward_a(const float* grad_out, const float* b, float* grad_a, int m, int n, int p)
    _lib.pt_matmul_backward_a.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
    _lib.pt_matmul_backward_a.restype = None

    # void pt_matmul_backward_b(const float* a, const float* grad_out, float* grad_b, int m, int n, int p)
    _lib.pt_matmul_backward_b.argtypes = [_float_p, _float_p, _float_p, _c_int, _c_int, _c_int]
    _lib.pt_matmul_backward_b.restype = None

    # void pt_pow_scalar(const float* a, float* out, float exponent, int n)
    _lib.pt_pow_scalar.argtypes = [_float_p, _float_p, _c_float, _c_int]
    _lib.pt_pow_scalar.restype = None

    # void pt_relu(const float* a, float* out, int n)
    _lib.pt_relu.argtypes = [_float_p, _float_p, _c_int]
    _lib.pt_relu.restype = None

    # void pt_sum(const float* a, float* out, int n)
    _lib.pt_sum.argtypes = [_float_p, _float_p, _c_int]
    _lib.pt_sum.restype = None

    # void pt_fill(float* out, float value, int n)
    _lib.pt_fill.argtypes = [_float_p, _c_float, _c_int]
    _lib.pt_fill.restype = None

    # void pt_add_inplace(float* out, const float* a, int n)
    _lib.pt_add_inplace.argtypes = [_float_p, _float_p, _c_int]
    _lib.pt_add_inplace.restype = None

    # void pt_add_scalar(const float* a, float* out, float scalar, int n)
    _lib.pt_add_scalar.argtypes = [_float_p, _float_p, _c_float, _c_int]
    _lib.pt_add_scalar.restype = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ptr(arr):
    """Get a ctypes float pointer from a contiguous numpy float32 array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ensure_f32(arr):
    """Ensure array is contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add(a, b, alpha=1.0):
    """Element-wise: out = a + alpha * b"""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    n = a.size
    out = np.empty(n, dtype=np.float32)
    _lib.pt_add(_ptr(a), _ptr(b), _ptr(out), ctypes.c_float(alpha), ctypes.c_int(n))
    return out


def mul(a, b):
    """Element-wise: out = a * b"""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    n = a.size
    out = np.empty(n, dtype=np.float32)
    _lib.pt_mul(_ptr(a), _ptr(b), _ptr(out), ctypes.c_int(n))
    return out


def matmul(a, b, m, n, p):
    """Matrix multiply: out[m,p] = a[m,n] @ b[n,p]  (row-major, flat arrays)"""
    a = _ensure_f32(a)
    b = _ensure_f32(b)
    out = np.empty(m * p, dtype=np.float32)
    _lib.pt_matmul(_ptr(a), _ptr(b), _ptr(out),
                   ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(p))
    return out.reshape(m, p)


def matmul_backward_a(grad_out, b, m, n, p):
    """grad_a[m,n] = grad_out[m,p] @ b^T[p,n]"""
    grad_out = _ensure_f32(grad_out)
    b = _ensure_f32(b)
    grad_a = np.empty(m * n, dtype=np.float32)
    _lib.pt_matmul_backward_a(_ptr(grad_out), _ptr(b), _ptr(grad_a),
                              ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(p))
    return grad_a.reshape(m, n)


def matmul_backward_b(a, grad_out, m, n, p):
    """grad_b[n,p] = a^T[n,m] @ grad_out[m,p]"""
    a = _ensure_f32(a)
    grad_out = _ensure_f32(grad_out)
    grad_b = np.empty(n * p, dtype=np.float32)
    _lib.pt_matmul_backward_b(_ptr(a), _ptr(grad_out), _ptr(grad_b),
                              ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(p))
    return grad_b.reshape(n, p)


def pow_scalar(a, exponent):
    """Element-wise: out = a ^ exponent"""
    a = _ensure_f32(a)
    n = a.size
    out = np.empty(n, dtype=np.float32)
    _lib.pt_pow_scalar(_ptr(a), _ptr(out), ctypes.c_float(exponent), ctypes.c_int(n))
    return out


def relu(a):
    """Element-wise: out = max(0, a)"""
    a = _ensure_f32(a)
    n = a.size
    out = np.empty(n, dtype=np.float32)
    _lib.pt_relu(_ptr(a), _ptr(out), ctypes.c_int(n))
    return out


def sum_all(a):
    """Reduction: sum of all elements, returns numpy array of size 1."""
    a = _ensure_f32(a)
    n = a.size
    out = np.zeros(1, dtype=np.float32)
    _lib.pt_sum(_ptr(a), _ptr(out), ctypes.c_int(n))
    return out


def fill(n, value):
    """Create array of size n filled with value."""
    out = np.empty(n, dtype=np.float32)
    _lib.pt_fill(_ptr(out), ctypes.c_float(value), ctypes.c_int(n))
    return out


def add_inplace(out, a):
    """In-place: out += a.  Returns out."""
    out = _ensure_f32(out)
    a = _ensure_f32(a)
    n = out.size
    _lib.pt_add_inplace(_ptr(out), _ptr(a), ctypes.c_int(n))
    return out


def zeros(n):
    """Create zero-filled numpy array of size n."""
    return fill(n, 0.0)


def ones(n):
    """Create ones-filled numpy array of size n."""
    return fill(n, 1.0)
