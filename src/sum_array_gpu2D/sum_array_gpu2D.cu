#include<iostream>

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

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            std::cerr << "Result verification failed at element " << i << "!\n";
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Result verification passed!\n";
}

void initialData(float *ip, const int N) {
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < N; i++) {
        ip[i] = static_cast<float>(rand()&0xFF) / 10.0f;
    }
}

//指针偏移
void sumArrayOnHost(float *A, float *B, float *C, int nx, int ny) {
    float *a = A;
    float *b = B;
    float *c = C;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            c[ix] = a[ix] + b[ix];
        }
        a+=nx,b+=nx,c+=nx;
    }
}

__global__ void sumArrayOnGPU2D(float *A, float *B, float *C, int nx, int ny) {
    // 二维索引
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        int idx = iy * nx + ix; // 线性化索引访问数组
        C[idx] = A[idx] + B[idx];
    }
}

void printDeviceInfo(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device %d: %s\n", dev, prop.name);
    printf("  Total SMs: %d\n", prop.multiProcessorCount);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dim: (%d %d %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size: (%d %d %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Shared mem per block: %zu\n", prop.sharedMemPerBlock);
    printf("  Registers per block: %d\n", prop.regsPerBlock);
    printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("  Total global memory: %zu MB\n",
           prop.totalGlobalMem / (1024 * 1024));
}

double cpu_second(){
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return ((double)tp.tv_sec +(double)tp.tv_nsec * 1.e-9);
}



int main()
{
    std::cout<< "Starting sumArrayOnGPU example..." << std::endl;

    int dev=0;
    cudaCheck(cudaSetDevice(dev));
    printDeviceInfo(dev);

    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGrid,
        &blockSize,
        sumArrayOnGPU2D,
        0,           // 动态共享内存
        0);          // 最小块数

    printf("Recommended block size = %d\n", blockSize);

    int nx=1<<14;
    int ny=1<<14;
    int nxy=nx*ny;
    std::cout<<"Vector size: "<< nxy << std::endl;
    size_t nBytes = nxy * sizeof(float);

    float* h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc((float**)&d_a, nBytes));
    cudaCheck(cudaMalloc((float**)&d_b, nBytes));
    cudaCheck(cudaMalloc((float**)&d_c, nBytes));

    cudaCheck(cudaMemcpy(d_a, h_A, nBytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16); // 768 线程/block
    dim3 grid((nx + block.x - 1)/block.x,
            (ny + block.y - 1)/block.y);


    double iStart = cpu_second();
    sumArrayOnGPU2D<<<grid, block>>>(d_a, d_b, d_c, nx,ny);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    double iElaps = cpu_second() - iStart;
    std::cout << "GPU execution time: " << iElaps * 1000.0 << " ms" << std::endl;

    cudaCheck(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

    iStart = cpu_second();
    sumArrayOnHost(h_A, h_B, hostRef,nx,ny);
    iElaps = cpu_second() - iStart;
    std::cout << "CPU execution time: " << iElaps * 1000.0 << " ms" << std::endl;
        
    checkResult(hostRef, gpuRef, nxy);

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}