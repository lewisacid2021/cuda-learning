# 思考

## 1.参考图1-5,分析以下几种数据划分形式:  
- 对于二维数据,沿x轴进行块划分  
- 对于二维数据,沿y轴进行周期划分  
- 对于三维数据,沿z轴进行周期划分

| 划分方式       | 内存连续性 | 缓存利用 | 负载均衡 | 索引计算复杂度 | 适用场景 |
|----------------|------------|----------|-----------|----------------|----------|
| 2D x 轴块      | 高         | 高       | 可能不均衡 | 低             | 行方向访问密集、计算均匀 |
| 2D y 轴周期    | 低         | 低       | 均衡       | 中             | 列方向计算量差异大 |
| 3D z 轴周期    | 中低       | 中低     | 均衡       | 中高           | 三维数据层计算差异大或总量大 |

- 索引计算复杂度：

    - 低：直接按块连续计算索引

    - 中：需要计算 col % num_blocks 或 z % num_blocks

    - 中高：三维周期划分，需要多维 modulo / offset 计算


## 2.从hello.cu中移除cudaDeviceReset函数,然后编译运行,看看会发生什么。

如果不加入cudaDeviceReset函数，仅输出Hello From CPU，在内核函数执行结束前，程序就已经退出

但是，cudaDeviceReset() 不是**线程阻塞**函数，它的作用也**不是用来同步或等待 GPU 任务**，而是释放当前进程所占用的GPU资源、销毁CUDA上下文、清空所有内存分配状态、清楚所有CUDA内部状态，重置GPU到刚启动程序的状态

实际上，从执行效果来看，cudaDeviceReset()内部在GPU资源被其他kernel占用的时候，不会强制释放相关资源和销毁上下文，而是等待kernel执行完毕之后再进行，也起到了阻塞的效果

- 开销很高，对于长时间运行的进程不建议频繁使用

## 用cudaDeviceSynchronize函数来替换hello.cu中的cudaDeviceReset函数,然后编译运行,看看会发生什么

与cudaDeviceReset()执行效果没有区别

cudaDeviceSynchronize()是标准的同步函数，阻塞调用线程，直到当前设备上所有先前启动的kernel和异步操作完成

- 只负责同步和等待，不释放资源

## 4.参考1.3节,从编译器命令行中移除设备架构标志,然后按照下面 的方式进行编译,看看会发生什么。  

如果不指定架构：

- nvcc 会生成兼容性 PTX（可能是 compute_52 或 compute_60 之类的通用 PTX）

- 运行时驱动会 JIT 编译到 SM 8.6

潜在问题：

- 首次 kernel launch 会稍慢（JIT PTX 编译）

- 无法利用新硬件特性（如 Tensor Core FP16/INT8）

- 寄存器分配 / block size 自动优化可能不理想 → 性能下降

## 5.参阅CUDA在线文档(http://docs.nvidia.com/cuda/index.html)。基于“CUDA编译器驱动NVCC”一节,谈谈nvcc对带有哪些后缀的文件支持编译?  

| Input File Suffix | Description |
|------------------|------------|
| .cu              | CUDA source file, containing host code and device functions |
| .c               | C source file |
| .cc, .cxx, .cpp  | C++ source file |
| .ptx             | PTX intermediate assembly file (see Figure 1) |
| .cubin           | CUDA device code binary file (CUBIN) for a single GPU architecture (see Figure 1) |
| .fatbin          | CUDA fat binary file that may contain multiple PTX and CUBIN files (see Figure 1) |
| .o, .obj         | Object file |
| .a, .lib         | Library file |
| .res             | Resource file |
| .so              | Shared object file |


## 6.为执行核函数的每个线程提供了一个唯一的线程ID,通过内置变 量threadIdx.x可以在内核中对线程进行访问。在hello.cu中修改核函数的线程索引,使输出如下:

简单计算线程索引