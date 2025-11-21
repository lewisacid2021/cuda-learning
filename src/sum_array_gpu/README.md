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