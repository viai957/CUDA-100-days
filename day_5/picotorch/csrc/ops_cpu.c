#include "ops_cpu.h"
#include <math.h>

void pt_add(const float* a, const float* b, float* out, float alpha, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + alpha * b[i];
    }
}

void pt_mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void pt_matmul(const float* a, const float* b, float* out, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

void pt_matmul_backward_a(const float* grad_out, const float* b, float* grad_a, int m, int n, int p) {
    // grad_a[m,n] = grad_out[m,p] @ b^T[p,n]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < p; k++) {
                sum += grad_out[i * p + k] * b[j * p + k];
            }
            grad_a[i * n + j] = sum;
        }
    }
}

void pt_matmul_backward_b(const float* a, const float* grad_out, float* grad_b, int m, int n, int p) {
    // grad_b[n,p] = a^T[n,m] @ grad_out[m,p]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += a[k * n + i] * grad_out[k * p + j];
            }
            grad_b[i * p + j] = sum;
        }
    }
}

void pt_pow_scalar(const float* a, float* out, float exponent, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = powf(a[i], exponent);
    }
}

void pt_relu(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

void pt_sum(const float* a, float* out, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    *out = s;
}

void pt_fill(float* out, float value, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = value;
    }
}

void pt_add_inplace(float* out, const float* a, int n) {
    for (int i = 0; i < n; i++) {
        out[i] += a[i];
    }
}

void pt_add_scalar(const float* a, float* out, float scalar, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + scalar;
    }
}
