// CPU operations
// This file contains the implementation of the CPU operations
// It is used to perform the operations on the CPU
#include "ops_cpu.h"
#include <math.h>

// Element-wise: out[i] = a[i] + alpha * b[i]
void pt_add(const float* a, const float* b, float* out, float alpha, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] + alpha * b[i];
    }
}

// Element-wise: out[i] = a[i] * b[i]
void pt_mul(const float* a, const float* b, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] * b[i];
    }
}

// Matrix multiply: out[m, p] = a[m, n] @ b[n, p]  (row-major)
void pt_matmul(const float* a, const float* b, float* out, int m, int n, int p){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < p; j++){
            float sum = 0.0f;
            for (int k = 0; k < n; k++){
                for (int l = 0; l < p; l++){
                    // a[i, n] @ b[n, p] = out[i, p]
                    // m -> no of rows in a
                    // n -> no of columns in a and no of rows in b
                    // p -> no of columns in b
                    sum += a[i * n + k] * b[k * p + l];
                }
                out[i * p + j] = sum;
            }
        }
    }
}

// Matmul backward wrt A: grad_a[m, n] = grad_out[m, p] @ b^T[p, n]
void pt_matmul_backward_a(const float* grad_out, const float* b, float* grad_a, int m, int n, int p){
    // grad_a[m, n] = grad_out[m, p] @ b^T[p,n]
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            float sum = 0.0f;
            for (int k = 0; k < p; k++){
                sum += grad_out[i * p + k] * b[j * p + k]; 
            }
            grad_a[i * n + j] = sum;
        }
    }
}

// Matmul backward wrt B: grad_b[n, p] = a^T[n, m] @ grad_out[m, p]
void pt_matmul_backward_b(const float* a, const float* grad_out, float* grad_b, int m, int n, int p){
    // grad_b[n, p] = a^T[n, m] @ grad_out[m, p]
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            float sum = 0.0f;
            for (int k = 0; k < m; k++){
                sum += a[k * n + i] * grad_out[k * p + j];
            }
            grad_b[i * p + j] = sum;
        }
    }
}

// Element-wise: out[i] = a[i] ^ exp
void pt_pow_scalar(const float* a, float* out, float exponent, int n){
    for (int i = 0; i < n; i++){
        out[i] = powf(a[i], exponent);
    }
}

// Element-wise: out[i] = max(0, a[i])
void pt_relu(const float* a, float* out, int n){
    for (int i = 0; i < n; i++){
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

// Reduction: *out = sum(a[0..n-1])
void pt_sum(const float* a, float* out, int n){
    float s = 0.0f;
    for (int i = 0; i < n; i++){
        s += a[i];
    }
    *out = s;
}

// Fill: out[0..n-1] = value
void pt_fill(float* out, float value, int n){
    for (int i = 0; i < n; i++){
        out[i] = value;
    }
}


// Element-wise: out[i] += a[i]  (in-place accumulate)
void pt_add_inplace(float* out, const float* a, int n){
    for (int i = 0; i < n; i++){
        out[i] += a[i];
    }
}