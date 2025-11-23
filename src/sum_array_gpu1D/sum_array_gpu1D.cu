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

__global__ void sumArrayOnGPU1D(float *A, float *B, float *C, int nx, int ny) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if(ix<nx){
        for(int iy=0; iy<ny; iy++){
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
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

    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGrid,
        &blockSize,
        sumArrayOnGPU1D,
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

    dim3 block(768, 1); // 768 线程/block
    dim3 grid((nx + block.x - 1)/block.x,
            1);


    double iStart = cpu_second();
    sumArrayOnGPU1D<<<grid, block>>>(d_a, d_b, d_c, nx,ny);
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