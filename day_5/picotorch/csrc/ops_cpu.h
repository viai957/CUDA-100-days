#ifndef PICOTORCH_OPS_CPU_H
#define PICOTORCH_OPS_CPU_H

// Element-wise: out[i] = a[i] + alpha * b[i]
void pt_add(const float* a, const float* b, float* out, float alpha, int n);

// Element-wise: out[i] = a[i] * b[i]
void pt_mul(const float* a, const float* b, float* out, int n);

// Matrix multiply: out[m,p] = a[m,n] @ b[n,p]  (row-major)
void pt_matmul(const float* a, const float* b, float* out, int m, int n, int p);

// Matmul backward wrt A: grad_a[m,n] = grad_out[m,p] @ b^T[p,n]
void pt_matmul_backward_a(const float* grad_out, const float* b, float* grad_a, int m, int n, int p);

// Matmul backward wrt B: grad_b[n,p] = a^T[n,m] @ grad_out[m,p]
void pt_matmul_backward_b(const float* a, const float* grad_out, float* grad_b, int m, int n, int p);

// Element-wise: out[i] = a[i] ^ exp
void pt_pow_scalar(const float* a, float* out, float exponent, int n);

// Element-wise: out[i] = max(0, a[i])
void pt_relu(const float* a, float* out, int n);

// Reduction: *out = sum(a[0..n-1])
void pt_sum(const float* a, float* out, int n);

// Fill: out[0..n-1] = value
void pt_fill(float* out, float value, int n);

// Element-wise: out[i] += a[i]  (in-place accumulate)
void pt_add_inplace(float* out, const float* a, int n);

// Broadcast scalar to all elements: out[i] = a[i] + scalar
void pt_add_scalar(const float* a, float* out, float scalar, int n);

#endif
