# 样例输出

- NVIDIA GeForce RTX 3090 * 4
- NVIDIA-SMI 565.57.01             
- Driver Version: 565.57.01      
- CUDA Version: 12.7


---

# ✅ **1. CUDA 版本**

### **Driver / Runtime: 12.7 / 12.0**

* **Driver**：底层支持，版本可以高于 Runtime。驱动更新后老程序仍可运行（向下兼容）。
* **Runtime**：你编译程序时链接的库版本。

💡 **使用场景**：

* 新特性（如 Cooperative Groups、Tensor Core）需要新 Runtime 支持
* 向下兼容可以在旧程序上运行

---

# ✅ **2. 计算能力（Compute Capability）**

### **CUDA Capability: 8.6 (Ampere 架构)**

决定 GPU 的硬件特性：

* **Warp size**：32
* **Max threads/block**：1024
* **Shared memory per SM**：大小
* **Tensor Core 支持**：用于 FP16、INT8 等加速
* **原子操作、指令集**

💡 **使用场景**：

* 编写通用矩阵乘法、卷积运算、深度学习推理程序
* 决定 block/thread 配置的硬件上限

---

# ✅ **3. 全局显存**

### **Global memory: 24 GB**

用于存储：

* 大矩阵、图像
* 模型参数
* GPU kernel 数据交换

💡 **使用场景**：

* 大 batch size 的训练
* 图像或视频处理的大型数据块

---

# 🔥 **4. SM 数量 (Multiprocessor Count)**

### **82 SM**

每个 SM 可以执行多个 warps 并行：

* **更多 SM → 更高并行度**
* **深度学习训练**：大量 warps 同时执行 tensor core 操作
* **图像/矩阵处理**：分块并行计算

---

# 🔥 **5. 核心与显存时钟**

### GPU Clock: 1755 MHz

### Memory Clock: 9751 MHz, Bus Width: 384-bit

可算带宽：

[
BW = 9751 \text{ MHz} \times 2 (\text{双倍速}) \times 384/8 = 936 \text{ GB/s}
]

💡 **使用场景**：

* **Memory-bound 内核**：如大矩阵逐元素加法，带宽决定性能上限
* **Compute-bound 内核**：如矩阵乘法，核心频率决定性能

---

# 🔥 **6. L2 Cache 大小**

### **6 MB**

* 减少 global memory 访问
* 重要于 **重复访问相同数据** 的 kernel

💡 **场景**：tile-based 矩阵乘法、图像卷积、共享纹理访问

---

# ⭐ **7. 纹理尺寸限制**

### **Max Texture Dimensions**

| 类型         | 限制                   | 场景          |
| ---------- | -------------------- | ----------- |
| 1D         | 131072               | 一维向量/数组     |
| 2D         | 131072 x 65536       | 图像/矩阵       |
| 3D         | 16384³               | 体积数据/3D卷积   |
| Layered 2D | 32768 x 32768 x 2048 | 多帧图像、深度学习权重 |

💡 **使用场景**：

* GPU 图像处理、体积渲染
* Tensor Core 加速时，纹理可缓存输入特征图

---

# 🔥 **8. Constant Memory**

### 64 KB

* 低延迟，broadcast 到 warp
* 适合**常量参数**或**卷积核权重**

---

# 🔥 **9. Shared Memory & Registers**

* **Shared memory per block**: 48 KB → 用于 tile 化加速矩阵运算
* **Registers per block**: 65536 → 每线程使用寄存器越多，SM 上能驻留的 block 越少

💡 **使用场景**：

* Tile-based 矩阵乘法、卷积
* 避免寄存器/共享内存过量导致 occupancy 下降

---

# 🔥 **10. Warp size**

### 32

* CUDA 的最小执行单位
* 决定 block thread 最好为 32 的倍数
* 影响 **memory coalescing** 和分支发散

---

# 🔥 **11. 最大线程数**

* **Max threads per SM**: 1536
* **Max threads per block**: 1024

💡 **使用场景**：

* 一维内核可以使用 1024 threads/block
* 二维内核 block.x * block.y ≤ 1024

---

# 🔥 **12. Block / Grid 最大维度**

* **BlockDim**: x=1024, y=1024, z=64
* **GridDim**: x=2B, y=65535, z=65535

💡 **使用场景**：

* 大矩阵、图像处理 → 二维 grid 配合二维 block
* 高分辨率视频/深度图 → 多层 grid

---

# ⭐ **13. Maximum Memory Pitch**

### 2147483647 bytes (~2GB)

* 使用 `cudaMallocPitch` 时单行最大长度
* 对二维数组/图像特别重要

---

# 在运行时设置设备  
支持多GPU的系统是很常见的。对于一个有N个GPU的系统,  `nvidia-smi`从0到N―1标记设备ID。使用环境变量  `CUDA_VISIBLE_DEVICES`,就可以在运行时指定所选的GPU且无须更改应用程序。  

设置运行时环境变量`CUDA_VISIBLE_DEVICES=2`。nvidia驱动程序会屏蔽其他GPU,这时设备2作为设备0出现在应用程序中。  

也可以使用`CUDA_VISIBLE_DEVICES`指定多个设备。例如,如果 想测试GPU 2和GPU 3,可以设置`CUDA_VISIBLE_DEVICES=2,3`。然后,在运行时,nvidia驱动程序将只使用ID为2和3的设备,并且会将设备ID分别映射为0和1。