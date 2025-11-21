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

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}



int main()
{
    std::cout<< "Starting sumArrayOnGPU example..." << std::endl;

    int dev=0;
    cudaSetDevice(dev);

    int nElem=1024;
    std::cout<<"Vector size: "<< nElem << std::endl;
    size_t nBytes = nElem * sizeof(float);

    float* h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc((float**)&d_a, nBytes));
    cudaCheck(cudaMalloc((float**)&d_b, nBytes));
    cudaCheck(cudaMalloc((float**)&d_c, nBytes));

    cudaCheck(cudaMemcpy(d_a, h_A, nBytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(nElem);
    dim3 grid((nElem + block.x - 1)/ block.x);

    sumArrayOnGPU<<<grid, block>>>(d_a, d_b, d_c, nElem);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost));

    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}