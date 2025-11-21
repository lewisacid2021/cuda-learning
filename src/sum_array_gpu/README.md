# 思考
## Q1.如果不使用cudaSetDevice，默认时核函数的执行是如何分配设备的？
- CUDA 不调用 cudaSetDevice() 时，会使用“默认设备”，默认设备的选择规则是完全确定的，不会随机，也不会动态变化。如果你从未调用 cudaSetDevice()，当前设备被设定为 0 号 GPU（device 0）
- 核函数 <<<>>> 的执行设备：永远是“当前设备”，如果没有调用过 cudaSetDevice：所有 kernel 都会在 GPU 0 上执行。
- 默认设备不会根据负载、free memory、性能由 CUDA 自动调度，需要程序员自己调度。
## Q2.我在增大nElem的过程中出现了checkResult failed的情况，原因是？
- CUDA Block 内线程数限制：每个 block 的线程数 必须满足硬件最大 threads per block 限制。如RTX 3090 maxThreadsPerBlock = 1024，必须满足：**block.x * block.y * block.z ≤ 1024**
- Grid 大小限制（几乎不会遇到）：典型 Ampere GPU的限制如下：
```
    grid.x ≤ 2^31 - 1
    grid.y ≤ 65535
    grid.z ≤ 65535
```
一般根本不会超。
- CUDA kernel 的限制不是「线程总数」，而是：限制单个 block 的线程数不能超过硬件上限

## 获取硬件约束的API
- 最全面cudaGetDeviceProperties()
一次性获取所有属性
```cpp
void printDeviceInfo(int dev)
{
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
```

- 更细粒度cudaDeviceGetAttribute()
| 属性                                   | 含义              |
| ------------------------------------ | --------------- |
| `cudaDevAttrMaxThreadsPerBlock`      | 每个 block 的最大线程数 |
| `cudaDevAttrMaxBlockDimX/Y/Z`        | block 维度最大值     |
| `cudaDevAttrMaxGridDimX/Y/Z`         | grid 维度最大值      |
| `cudaDevAttrWarpSize`                | warp 大小         |
| `cudaDevAttrMaxSharedMemoryPerBlock` | shared memory   |
| `cudaDevAttrMultiprocessorCount`     | SM 数量           |
| `cudaDevAttrClockRate`               | SM 时钟           |
| …                                    | （几乎所有硬件资源）      |

- 最自动化的方式cudaOccupancyMaxPotentialBlockSize()
CUDA 提供自动计算最佳 block 大小的 API：
```cpp
    int minGrid, blockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGrid,
        &blockSize,
        sumArrayOnGPU,
        0,           // 动态共享内存
        0);          // 最小块数

    printf("Recommended block size = %d\n", blockSize);
```

## cuda block的线程数和执行速度是如何tradeoff的，是否是线程数越多，执行速度越快？
- GPU 不是 CPU，它是 SIMT 架构。每个 SM（Streaming Multiprocessor）同时执行 多个 warp（每 warp 32 个线程），block 内线程数决定了一个 block 内 warp 的数量：
```cpp
numWarpsPerBlock=⌈threadsPerBlock/32⌉
```
SM 资源有限，且每个 SM 能同时挂载的线程数、block 数、register 数、shared memory 都有限。


- 小 block 的情况
    - block 内 warp 数少
    - 一个 SM 同时可以运行更多 block（因为 SM 资源足够）
    - 可能导致 warp 数不足，SM 不能充分隐藏内存延迟
    - GPU 利用率低 → kernel 慢

- 大 block 的情况
    - 线程数大（比如 512、1024）：
    - block 内 warp 多 → 每个 block 占用更多 SM 资源（register / shared memory）
    - 每个 SM 能挂载的 block 数可能减少
    - 如果 block 太大 → 少 block 挂载 → GPU warp 总数不足 → 仍然可能降低吞吐量