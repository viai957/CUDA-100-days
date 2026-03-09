#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "cuda_common.cuh"

typedef float EL_TYPE;

// A is [M x N] row-major, OUT is [N x M] row-major
// A[m, n] -> A[m * stride_A0 + n * stride_A1], strides = (N,1)
// OUT[n, m] -> OUT[n * stride_OUT0 + m * stride_OUT1], strides = (M, 1)
__global__ void matrix_transpsoe(EL_TYPE *OUT, EL_TYPE *A, int N, int M)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // n in [0, N]
    const int col = blockIdx.x * blockDim.y + threadIdx.x; // m in [0, M]

    if (row < M && col < N)
    {
        // input strides (row-major): (N, 1)
        const int in_idx = row * N + col; // A[m, n]

        // output strides (row-major on transposed shape [N x M]): (M, 1)
        const int out_idx = col * M + row; // OUT[N, M]

        OUT[out_idx] = A[in_idx];
    }
}

void test_matrix_transpsoe(int M, int N)
{
    EL_TYPE *A, *OUT;
    EL_TYPE *d_A, *d_OUT;

    const size_t bytes_A = (size_t)M * (size_t)N * sizeof(EL_TYPE);
    const size_t bytes_OUT = (size_t)N * (size_t)M * sizeof(EL_TYPE);

    // Host allocation
    A = (EL_TYPE *)malloc(bytes_A);
    OUT = (EL_TYPE *)malloc(bytes_OUT);
    assert(A && OUT);

    // Initialize A with random data
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            const int idx = m * N + n;
            A[idx] = (EL_TYPE)((rand() % 100) / 1000.0f);
        }
    }

    // Device allocation
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, bytes_OUT));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    // Launch configuration (2D)
    const dim3 threads_per_block(16, 16);
    const dim3 num_blocks(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );

    CUDA_CHECK(cudaEventRecord(start_kernel));

    matrix_transpsoe<<<num_blocks, threads_per_block>>>(d_OUT, d_A, N, M);

    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Transpose - elapsed time: %f ms\n", milliseconds_kernel);

    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    float milliseconds_kernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Transpose [%d x %d] -> [%d x %d] - elapsed time: %f ms\n",
           M, N, N, M, milliseconds_kernel);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, bytes_OUT, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_OUT));

    // CPU check (reference transpose with same stride math)
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    int errors = 0;
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            const int in_idx  = m * N + n;  // A[m, n]
            const int out_idx = n * M + m;  // OUT[n, m]
            EL_TYPE ref = A[in_idx];
            if (OUT[out_idx] != ref)
            {
                if (errors < 10)
                {
                    printf("Error at (m=%d, n=%d): OUT[%d]=%f, ref=%f\n",
                           m, n, out_idx, (float)OUT[out_idx], (float)ref);
                }
                errors++;
            }
        }
    }

    gettimeofday(&end_check, NULL);
    float elapsed_check = (end_check.tv_sec - start_check.tv_sec) * 1000.0f +
                          (end_check.tv_usec - start_check.tv_usec) / 1000.0f;

    if (errors == 0)
    {
        printf("Matrix Transpose - result OK, check time: %f ms\n", elapsed_check);
    }
    else
    {
        printf("Matrix Transpose - %d errors, check time: %f ms\n", errors, elapsed_check);
    }

    // Free host memory
    free(A);
    free(OUT);
}

int main()
{
    srand(0);

    // You can change these to experiment
    test_matrix_transpose(512, 512);
    test_matrix_transpose(256, 1024);
    test_matrix_transpose(777, 333);

    return 0;
}