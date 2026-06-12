#include <iostream>
#include "gpu_logic.cuh"

// This is the actual CUDA kernel that runs on the GPU
__global__ void test_kernel() {
    int thread_id = threadIdx.x;
    if (thread_id == 0) {
        printf("[GPU] Calculating magnon-polaron hybridization from device!\n");
    }
}

// This is the host-side wrapper function that launches the kernel
void launch_cuda_test() {
    std::cout << "[Host] Preparing to launch CUDA kernel..." << std::endl;
    
    // Launch the kernel with 1 block of 1 thread
    test_kernel<<<1, 1>>>();
    
    // Crucial: Wait for the GPU to finish before the CPU continues.
    // If you forget this, the program will exit before the GPU can print!
    cudaDeviceSynchronize();
}