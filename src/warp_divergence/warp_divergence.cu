#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

inline void cudaCheckImpl(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at "
                  << file << ":" << line << "\n"
                  << "Code: " << error << ", Reason: "
                  << cudaGetErrorString(error) << std::endl;

        throw cudaError(error);
    }
}

#define cudaCheck(call)  cudaCheckImpl((call), __FILE__, __LINE__)

__global__ void mathKernel1(float *c){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a=b=0.0f;
    if(tid%2==0){
        a=100.0f;
    }
    else{
        b=200.0f;
    }
    c[tid]=a+b;
}

__global__ void mathKernel2(float *c){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a=b=0.0f;
    if((tid/warpSize)%2==0){
        a=100.0f;
    }
    else{
        b=200.0f;
    }
    c[tid]=a+b;
}

__global__ void warmingup(float *c){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a=b=0.0f;
    if((tid/warpSize)%2==0){
        a=100.0f;
    }
    else{
        b=200.0f;
    }
    c[tid]=a+b;
}

int main()
{
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaCheck(cudaSetDevice(dev));
    cudaCheck(cudaGetDeviceProperties(&deviceProp,dev));
    std::cout<< "Using Device "<<dev<<": "<<deviceProp.name<<std::endl;

    int size = 64 * 1024;       // 64K threads
    int blocksize = 128;        // 128 threads per block

    dim3 block(blocksize,1,1);
    dim3 grid((size + block.x - 1) / block.x, 1, 1);

    std::cout << "Data size: " << size 
            << ", Blocks: " << grid.x 
            << ", Threads/block: " << block.x << std::endl;


    float *d_c;
    cudaCheck(cudaMalloc((float**)&d_c, sizeof(float)*size));

    // create CUDA events
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    float elapsed_ms;

    // --- warming up ---
    cudaCheck(cudaEventRecord(start));
    warmingup<<<grid, block>>>(d_c);
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Warming up time elapsed " << elapsed_ms * 1000 << " us" << std::endl;

    // --- mathKernel1 ---
    cudaCheck(cudaEventRecord(start));
    mathKernel1<<<grid, block>>>(d_c);
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "mathKernel1 time elapsed " << elapsed_ms * 1000 << " us" << std::endl;

    // --- mathKernel2 ---
    cudaCheck(cudaEventRecord(start));
    mathKernel2<<<grid, block>>>(d_c);
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "mathKernel2 time elapsed " << elapsed_ms * 1000 << " us" << std::endl;

    cudaCheck(cudaFree(d_c));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    return 0;
}
