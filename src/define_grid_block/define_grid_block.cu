#include<iostream>

int main()
{
    int nElem = 1024;

    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Grid Dim: (%d, %d, %d) Block Dim: (%d, %d, %d)\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);
    
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid Dim: (%d, %d, %d) Block Dim: (%d, %d, %d)\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);

    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid Dim: (%d, %d, %d) Block Dim: (%d, %d, %d)\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);

    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("Grid Dim: (%d, %d, %d) Block Dim: (%d, %d, %d)\n", 
        grid.x, grid.y, grid.z, block.x, block.y, block.z);

    return 0;
}