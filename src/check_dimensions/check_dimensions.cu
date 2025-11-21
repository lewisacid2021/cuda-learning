#include<iostream>

__global__ void checkDimensions(void)
{
    printf("Kernel Check:\n Grid Dim: (%d, %d, %d) Block Dim: (%d, %d, %d) Block Idx: (%d, %d, %d) Thread Idx: (%d, %d, %d)\n", 
        gridDim.x, gridDim.y, gridDim.z,blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x, blockIdx.y, blockIdx.z,threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    int nElem = 6;

    dim3 block(3);
    dim3 grid((nElem+block.x-1)/block.x);

    printf("Host Check:\n");
    std::cout << "Grid Dim: (" << grid.x << ", " << grid.y << ", " << grid.z << ") "
                << "Block Dim: (" << block.x << ", " << block.y << ", " << block.z << ")\n";

    checkDimensions<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}