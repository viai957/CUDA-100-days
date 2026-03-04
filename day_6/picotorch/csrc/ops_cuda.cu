/*
 * picotorch CUDA kernels — pico-scale GPU tensor operations.
 *
 * Mirrors ops_cpu.c but runs on NVIDIA GPUs.  Every public function
 * (pt_cuda_*) is a host-side wrapper that launches __global__ CUDA
 * kernels on DEVICE pointers.
 *
 * The Python layer (ops_cuda.py) manages device memory directly
 * through pt_cuda_malloc / pt_cuda_free / pt_cuda_memcpy_*, so
 * the kernel wrappers below operate on DEVICE pointers only.
 *
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -O2 -o libops_cuda.so ops_cuda.cu
 *
 * Inspired by PyTorch's ATen CUDA kernels (at::native).
 */

#include "ops_cuda.cuh"
#include <cuda_runtime.h>
#include <cstdio>

/* ================================================================
 *  Configuration
 * ================================================================ */

#define PT_BLOCK_SIZE   256
#define PT_TILE_SIZE     16    /* for tiled matmul & transpose */

/* Check CUDA call and abort on error. */
#define PT_CHECK(call)                                               \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "picotorch CUDA error at %s:%d – %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

static inline int grid1d(int n, int block = PT_BLOCK_SIZE) {
    return (n + block - 1) / block;
}

/* ================================================================
 *  __global__ Kernels
 * ================================================================ */

/* ── Element-wise ────────────────────────────────────────────── */

// out[i] = a[i] + alpha * b[i]
//
// Fused multiply-add: add (alpha=1), sub (alpha=-1), scaled add.
// Compare with PyTorch's add_clamp_kernel in BinaryOps.cpp which
// does the same with optional clamping via Vectorized<> on CPU.
// On GPU each thread handles one element — massive parallelism
// replaces SIMD vectorization.
__global__ void kernel_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float*       __restrict__ out,
                           float alpha, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + alpha * b[i];
}

// out[i] = a[i] * b[i]
//
// Mirrors PyTorch's mul_kernel which uses cpu_kernel_vec on CPU
// and gpu_kernel with a lambda on CUDA.  The pattern is identical:
//   gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) {
//       return a * b;
//   });
__global__ void kernel_mul(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float*       __restrict__ out,
                           int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}

// out[i] = a[i] ^ exponent
//
// Uses __powf (device intrinsic, fast but less precise than powf).
// PyTorch dispatches through AT_DISPATCH_FLOATING_TYPES for type
// flexibility; picotorch stays with float32.
__global__ void kernel_pow(const float* __restrict__ a,
                           float*       __restrict__ out,
                           float exponent, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = __powf(a[i], exponent);
}

// out[i] = max(0, a[i])    — ReLU forward
__global__ void kernel_relu(const float* __restrict__ a,
                            float*       __restrict__ out,
                            int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
}

// grad_in[i] = (a[i] > 0) ? grad_out[i] : 0    — ReLU backward
//
// Gradient flows through where the input was positive.
// In PyTorch this is threshold_backward_kernel.
__global__ void kernel_relu_bwd(const float* __restrict__ a,
                                const float* __restrict__ grad_out,
                                float*       __restrict__ grad_in,
                                int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        grad_in[i] = (a[i] > 0.0f) ? grad_out[i] : 0.0f;
}

/* ── Reduction ───────────────────────────────────────────────── */

// Parallel sum via shared-memory tree reduction + atomicAdd.
//
// Standard two-phase pattern (see NVIDIA reduction whitepaper):
//   Phase 1: each block reduces its chunk in shared memory
//   Phase 2: block leaders atomicAdd partial sums into output
//
// The output must be zeroed before launch (caller responsibility).
__global__ void kernel_sum(const float* __restrict__ a,
                           float* __restrict__ out,
                           int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i   = blockDim.x * blockIdx.x + threadIdx.x;

    sdata[tid] = (i < n) ? a[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, sdata[0]);
}

/* ── Transpose ───────────────────────────────────────────────── */

// Transpose a row-major matrix: out[cols, rows] = in[rows, cols]^T
//
// Uses shared memory tiles to coalesce global memory accesses.
// This is the standard tiled transpose pattern that avoids bank
// conflicts by padding the shared memory tile (+1 column).
__global__ void kernel_transpose(const float* __restrict__ in,
                                 float*       __restrict__ out,
                                 int rows, int cols)
{
    __shared__ float tile[PT_TILE_SIZE][PT_TILE_SIZE + 1]; // +1 avoids bank conflicts

    int x_in = blockIdx.x * PT_TILE_SIZE + threadIdx.x;  // column in input
    int y_in = blockIdx.y * PT_TILE_SIZE + threadIdx.y;  // row in input

    // Load tile from input (row-major): in[y_in, x_in]
    if (x_in < cols && y_in < rows)
        tile[threadIdx.y][threadIdx.x] = in[y_in * cols + x_in];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Write tile to output transposed
    int x_out = blockIdx.y * PT_TILE_SIZE + threadIdx.x;  // column in output
    int y_out = blockIdx.x * PT_TILE_SIZE + threadIdx.y;  // row in output

    if (x_out < rows && y_out < cols)
        out[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
}

/* ── Matrix multiply (tiled with shared memory) ──────────────── */

// C[m,p] = A[m,n] @ B[n,p]   (row-major)
//
// Shared memory tiles of PT_TILE_SIZE x PT_TILE_SIZE reduce global
// memory bandwidth by a factor of ~PT_TILE_SIZE.  This is the
// standard textbook tiled matmul (CUDA Programming Guide Ch. 3).
//
// PyTorch uses cuBLAS (vastly more optimized: double buffering,
// register tiling, tensor cores).  This kernel is educational —
// it shows the core idea before production optimizations.
__global__ void kernel_matmul(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int m, int n, int p)
{
    __shared__ float tileA[PT_TILE_SIZE][PT_TILE_SIZE];
    __shared__ float tileB[PT_TILE_SIZE][PT_TILE_SIZE];

    int row = blockIdx.y * PT_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * PT_TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + PT_TILE_SIZE - 1) / PT_TILE_SIZE; t++) {
        int a_col = t * PT_TILE_SIZE + threadIdx.x;
        int b_row = t * PT_TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < m && a_col < n) ? A[row * n + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < n && col < p) ? B[b_row * p + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < PT_TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = sum;
}

/* ── Utility kernels ─────────────────────────────────────────── */

__global__ void kernel_fill(float* __restrict__ out,
                            float value, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = value;
}

__global__ void kernel_add_inplace(float*       __restrict__ out,
                                   const float* __restrict__ a,
                                   int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] += a[i];
}

__global__ void kernel_add_scalar(const float* __restrict__ a,
                                  float*       __restrict__ out,
                                  float scalar, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + scalar;
}


/* ================================================================
 *  Host wrappers (extern "C")
 *
 *  All float* parameters are DEVICE pointers.
 * ================================================================ */

/* Helper: launch the tiled transpose kernel on GPU */
static void transpose_on_gpu(const float* d_in, float* d_out,
                              int rows, int cols)
{
    dim3 block(PT_TILE_SIZE, PT_TILE_SIZE);
    dim3 grid((cols + PT_TILE_SIZE - 1) / PT_TILE_SIZE,
              (rows + PT_TILE_SIZE - 1) / PT_TILE_SIZE);
    kernel_transpose<<<grid, block>>>(d_in, d_out, rows, cols);
    PT_CHECK(cudaGetLastError());
}

extern "C" {

/* ── Element-wise ────────────────────────────────────────────── */

void pt_cuda_add(const float* a, const float* b, float* out,
                 float alpha, int n)
{
    kernel_add<<<grid1d(n), PT_BLOCK_SIZE>>>(a, b, out, alpha, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_mul(const float* a, const float* b, float* out, int n)
{
    kernel_mul<<<grid1d(n), PT_BLOCK_SIZE>>>(a, b, out, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_pow_scalar(const float* a, float* out,
                        float exponent, int n)
{
    kernel_pow<<<grid1d(n), PT_BLOCK_SIZE>>>(a, out, exponent, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_relu(const float* a, float* out, int n)
{
    kernel_relu<<<grid1d(n), PT_BLOCK_SIZE>>>(a, out, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_relu_backward(const float* a, const float* grad_out,
                           float* grad_in, int n)
{
    kernel_relu_bwd<<<grid1d(n), PT_BLOCK_SIZE>>>(
        a, grad_out, grad_in, n);
    PT_CHECK(cudaGetLastError());
}

/* ── Reduction ───────────────────────────────────────────────── */

void pt_cuda_sum(const float* a, float* out, int n)
{
    PT_CHECK(cudaMemset(out, 0, sizeof(float)));
    int shared = PT_BLOCK_SIZE * sizeof(float);
    kernel_sum<<<grid1d(n), PT_BLOCK_SIZE, shared>>>(a, out, n);
    PT_CHECK(cudaGetLastError());
}

/* ── Matrix multiply ─────────────────────────────────────────── */

void pt_cuda_matmul(const float* a, const float* b, float* out,
                    int m, int n, int p)
{
    dim3 block(PT_TILE_SIZE, PT_TILE_SIZE);
    dim3 grid((p + PT_TILE_SIZE - 1) / PT_TILE_SIZE,
              (m + PT_TILE_SIZE - 1) / PT_TILE_SIZE);
    kernel_matmul<<<grid, block>>>(a, b, out, m, n, p);
    PT_CHECK(cudaGetLastError());
}

// grad_a[m,n] = grad_out[m,p] @ B^T[p,n]
//
// Strategy: transpose B on GPU, then call matmul.
// All done in device memory — no host round-trip.
void pt_cuda_matmul_backward_a(const float* grad_out, const float* b,
                               float* grad_a, int m, int n, int p)
{
    float* b_t;
    PT_CHECK(cudaMalloc(&b_t, n * p * sizeof(float)));

    // B[n,p] → B^T[p,n]
    transpose_on_gpu(b, b_t, n, p);

    // grad_a[m,n] = grad_out[m,p] @ b_t[p,n]
    pt_cuda_matmul(grad_out, b_t, grad_a, m, p, n);

    PT_CHECK(cudaFree(b_t));
}

// grad_b[n,p] = A^T[n,m] @ grad_out[m,p]
void pt_cuda_matmul_backward_b(const float* a, const float* grad_out,
                               float* grad_b, int m, int n, int p)
{
    float* a_t;
    PT_CHECK(cudaMalloc(&a_t, m * n * sizeof(float)));

    // A[m,n] → A^T[n,m]
    transpose_on_gpu(a, a_t, m, n);

    // grad_b[n,p] = a_t[n,m] @ grad_out[m,p]
    pt_cuda_matmul(a_t, grad_out, grad_b, n, m, p);

    PT_CHECK(cudaFree(a_t));
}

/* ── Utility ─────────────────────────────────────────────────── */

void pt_cuda_fill(float* out, float value, int n)
{
    kernel_fill<<<grid1d(n), PT_BLOCK_SIZE>>>(out, value, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_add_inplace(float* out, const float* a, int n)
{
    kernel_add_inplace<<<grid1d(n), PT_BLOCK_SIZE>>>(out, a, n);
    PT_CHECK(cudaGetLastError());
}

void pt_cuda_add_scalar(const float* a, float* out,
                        float scalar, int n)
{
    kernel_add_scalar<<<grid1d(n), PT_BLOCK_SIZE>>>(a, out, scalar, n);
    PT_CHECK(cudaGetLastError());
}

/* ── Device memory management ────────────────────────────────── */

float* pt_cuda_malloc(int n)
{
    float* d_ptr;
    PT_CHECK(cudaMalloc(&d_ptr, n * sizeof(float)));
    return d_ptr;
}

void pt_cuda_free(float* d_ptr)
{
    PT_CHECK(cudaFree(d_ptr));
}

void pt_cuda_memcpy_h2d(float* d_dst, const float* h_src, int n)
{
    PT_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void pt_cuda_memcpy_d2h(float* h_dst, const float* d_src, int n)
{
    PT_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

} /* extern "C" */
