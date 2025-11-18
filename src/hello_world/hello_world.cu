#include<stdio.h>
__global__ void helloFromGPU(void) {
    printf("Hello, World! From GPU\n");
}

int main(){
    printf("Hello, World! From CPU\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset(); // Ensure all GPU operations are complete before exiting
    return 0;
}