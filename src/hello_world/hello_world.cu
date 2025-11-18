#include<iostream>
__global__ void helloFromGPU(void) {
    int threadId = threadIdx.x;
    printf("Hello, World! From GPU %d\n",threadId);
}

int main(){
    printf("Hello, World! From CPU\n");
    helloFromGPU <<<1, 10>>>();
    //cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}