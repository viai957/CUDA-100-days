#ifndef PICOTORCH_OPS_CUDA_H
#define PICOTORCH_OPS_CUDA_H

/*
 * picotorch CUDA kernel declarations.
 *
 * These are C-linkage host functions that allocate device memory,
 * launch CUDA kernels, and copy results back.  They mirror the
 * signatures in ops_cpu.h so the Python dispatcher can route to
 * either backend interchangeably.
 *
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -O2 -o libops_cuda.so ops_cuda.cu
 */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Element-wise operations ─────────────────────────────────── */

// out[i] = a[i] + alpha * b[i]
void pt_cuda_add(const float* a, const float* b, float* out,
                 float alpha, int n);

// out[i] = a[i] * b[i]
void pt_cuda_mul(const float* a, const float* b, float* out, int n);

// out[i] = a[i] ^ exponent
void pt_cuda_pow_scalar(const float* a, float* out,
                        float exponent, int n);

// out[i] = max(0, a[i])
void pt_cuda_relu(const float* a, float* out, int n);

// out[i] = (a[i] > 0) ? 1.0 : 0.0   (ReLU derivative mask)
void pt_cuda_relu_backward(const float* a, const float* grad_out,
                           float* grad_in, int n);

/* ── Reduction ───────────────────────────────────────────────── */

// *out = sum(a[0..n-1])   (parallel reduction with shared memory)
void pt_cuda_sum(const float* a, float* out, int n);

/* ── Matrix multiply (naive tiled) ───────────────────────────── */

// out[m,p] = a[m,n] @ b[n,p]   (row-major)
void pt_cuda_matmul(const float* a, const float* b, float* out,
                    int m, int n, int p);

// grad_a[m,n] = grad_out[m,p] @ b^T[p,n]
void pt_cuda_matmul_backward_a(const float* grad_out, const float* b,
                               float* grad_a, int m, int n, int p);

// grad_b[n,p] = a^T[n,m] @ grad_out[m,p]
void pt_cuda_matmul_backward_b(const float* a, const float* grad_out,
                               float* grad_b, int m, int n, int p);

/* ── Utility ─────────────────────────────────────────────────── */

// out[0..n-1] = value
void pt_cuda_fill(float* out, float value, int n);

// out[i] += a[i]   (in-place accumulate)
void pt_cuda_add_inplace(float* out, const float* a, int n);

// out[i] = a[i] + scalar
void pt_cuda_add_scalar(const float* a, float* out, float scalar, int n);

/* ── Device memory management (host↔device transfers) ────────── */

// Allocate n floats on GPU, return device pointer
float* pt_cuda_malloc(int n);

// Free device pointer
void pt_cuda_free(float* d_ptr);

// Copy n floats: host → device
void pt_cuda_memcpy_h2d(float* d_dst, const float* h_src, int n);

// Copy n floats: device → host
void pt_cuda_memcpy_d2h(float* h_dst, const float* d_src, int n);

#ifdef __cplusplus
}
#endif

#endif /* PICOTORCH_OPS_CUDA_H */
