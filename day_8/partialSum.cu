#include <stdio.h>

__global__ void partialSumKernel(int *input, int *output, int n)
{
    // Shared memory
    extern __shared__ int sharedMemory[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n){
        // input -> shared memory -> optimized coalescing
        sharedMemory[threadIdx.x] = input[index];
        __syncthreads();

        // Perfomr inclusive scan in shared memory
        for (int stride = 1; stride < blockDim.x; stride *= 2){
            int temp = 0;
            if (threadIdx.x >= stride){
                temp = sharedMemory[threadIdx.x - stride];
            }
            __syncthreads();
            sharedMemory[threadIdx.x] += temp;
            __syncthreads();
        }

        // Write result to output
        output[index] = sharedMemory[threadIdx.x];
    }
}

void partialSum(int *input, int *output, int n){
    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice));
    partialSumKernel<<<(n + 255) / 256, 256, 256 * sizeof(int)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    printf("Partial Sum - result OK\n");
}

int main(){
    int n = 1000000;
    int *input = (int *)malloc(n * sizeof(int));
    int *output = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++){
        input[i] = rand() % 100;
    }
    partialSum(input, output, n);
    return 0;
}