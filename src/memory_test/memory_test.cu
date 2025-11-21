#include <ctime>
#include <cuda_runtime_api.h>
#include<iostream>

__global__ void sumOnDevice(float *a, float *b, float *c,const int N) {
    for(int idx=0; idx<N; idx++){
        c[idx] = a[idx] + b[idx];
    }
}

void initialData(float *ip, const int N) {
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < N; i++) {
        ip[i] = static_cast<float>(rand()&0xFF) / 10.0f;
    }
}

void printData(float *ip, const int N) {
    for (int i = 0; i < N; i++) {
        std::cout << ip[i] << " ";
    }
    std::cout << std::endl;
}

int main()
{
    int nElem=64;
    size_t nBytes = nElem * sizeof(float);
    float *h_a, *h_b, *h_c;
    h_a = (float *)malloc(nBytes);
    h_b = (float *)malloc(nBytes);
    h_c = (float *)malloc(nBytes);

    initialData(h_a, nElem);
    initialData(h_b, nElem);
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, nBytes);
    cudaMalloc((float**)&d_b, nBytes);
    cudaMalloc((float**)&d_c, nBytes);

    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
  
    sumOnDevice<<<1,1>>>(d_a, d_b, d_c, nElem);
    
    // cudaMemcpy本身就为同步API 会等待kernel执行完成
    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);
    printData(h_c, nElem);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}