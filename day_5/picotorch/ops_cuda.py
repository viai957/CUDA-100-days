"""CUDA backend: GPU kernels via CuPy RawKernel.

This module is only usable on machines with an NVIDIA GPU and CuPy installed.
Import is conditional -- the dispatcher handles fallback to CPU.
"""
import cupy as cp
import numpy as np

# --- CUDA Kernel Source Code ---

_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_add(const float* a, const float* b, float* out, float alpha, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + alpha * b[i];
    }
}
''', 'pt_add')

_mul_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_mul(const float* a, const float* b, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}
''', 'pt_mul')

_pow_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_pow(const float* a, float* out, float exponent, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = powf(a[i], exponent);
    }
}
''', 'pt_pow')

_relu_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_relu(const float* a, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}
''', 'pt_relu')

_sum_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_sum(const float* a, float* out, int n) {
    // Shared memory parallel reduction with atomicAdd for multi-block support
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    sdata[tid] = (i < n) ? a[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}
''', 'pt_sum')

_fill_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_fill(float* out, float value, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] = value;
    }
}
''', 'pt_fill')

_add_inplace_kernel = cp.RawKernel(r'''
extern "C" __global__
void pt_add_inplace(float* out, const float* a, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[i] += a[i];
    }
}
''', 'pt_add_inplace')


# --- Launch Configuration Helper ---

def _launch_config(n, block_size=256):
    """Calculate grid dimensions for n elements."""
    grid_size = (n + block_size - 1) // block_size
    return (grid_size,), (block_size,)


# --- Public API (mirrors ops_cpu.py signatures) ---

def add(a, b, alpha=1.0):
    """Element-wise: out = a + alpha * b"""
    n = a.size
    out = cp.empty(n, dtype=cp.float32)
    grid, block = _launch_config(n)
    _add_kernel(grid, block, (a, b, out, cp.float32(alpha), np.int32(n)))
    return out


def mul(a, b):
    """Element-wise: out = a * b"""
    n = a.size
    out = cp.empty(n, dtype=cp.float32)
    grid, block = _launch_config(n)
    _mul_kernel(grid, block, (a, b, out, np.int32(n)))
    return out


def matmul(a, b, m, n, p):
    """Matrix multiply using CuPy's optimized implementation (delegates to cuBLAS)."""
    a_2d = a.reshape(m, n)
    b_2d = b.reshape(n, p)
    return cp.dot(a_2d, b_2d).ravel()


def matmul_backward_a(grad_out, b, m, n, p):
    """grad_a[m,n] = grad_out[m,p] @ b^T[p,n]"""
    g = grad_out.reshape(m, p)
    b_2d = b.reshape(n, p)
    return cp.dot(g, b_2d.T).ravel()


def matmul_backward_b(a, grad_out, m, n, p):
    """grad_b[n,p] = a^T[n,m] @ grad_out[m,p]"""
    a_2d = a.reshape(m, n)
    g = grad_out.reshape(m, p)
    return cp.dot(a_2d.T, g).ravel()


def pow_scalar(a, exponent):
    """Element-wise: out = a ^ exponent"""
    n = a.size
    out = cp.empty(n, dtype=cp.float32)
    grid, block = _launch_config(n)
    _pow_kernel(grid, block, (a, out, cp.float32(exponent), np.int32(n)))
    return out


def relu(a):
    """Element-wise: out = max(0, a)"""
    n = a.size
    out = cp.empty(n, dtype=cp.float32)
    grid, block = _launch_config(n)
    _relu_kernel(grid, block, (a, out, np.int32(n)))
    return out


def sum_all(a):
    """Reduction: sum of all elements using shared-memory parallel reduction."""
    n = a.size
    out = cp.zeros(1, dtype=cp.float32)
    block_size = 256
    grid, block = _launch_config(n, block_size)
    # shared_mem = block_size * sizeof(float) for the shared memory reduction
    _sum_kernel(grid, block, (a, out, np.int32(n)), shared_mem=block_size * 4)
    return out


def fill(n, value):
    """Create array filled with value."""
    out = cp.empty(n, dtype=cp.float32)
    grid, block = _launch_config(n)
    _fill_kernel(grid, block, (out, cp.float32(value), np.int32(n)))
    return out


def add_inplace(out, a):
    """In-place: out += a"""
    n = out.size
    grid, block = _launch_config(n)
    _add_inplace_kernel(grid, block, (out, a, np.int32(n)))
    return out


def zeros(n):
    """Create zero-filled array on GPU."""
    return fill(n, 0.0)


def ones(n):
    """Create ones-filled array on GPU."""
    return fill(n, 1.0)


def to_device(np_array):
    """Transfer numpy array to GPU (CuPy array)."""
    return cp.asarray(np_array, dtype=cp.float32)


def to_numpy(cp_array):
    """Transfer CuPy array back to CPU as numpy."""
    return cp.asnumpy(cp_array).astype(np.float32)
