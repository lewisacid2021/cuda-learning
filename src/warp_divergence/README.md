# 理解线程束的本质

线程束是SM中基本的执行单元。当一个线程块的网格被启动后,  网格中的线程块分布在SM中。一旦线程块被调度到一个SM上,线程块中的线程会被进一步划分为线程束。一个线程束由32个连续的线程组  成,在一个线程束中,所有的线程按照单指令多线程(SIMT)方式执行;也就是说,**所有线程都执行相同的指令**,每个线程在私有数据上进行操作。

## 线程束分化

我们在编写cuda核函数时，可能会出现复杂的分支语句

但GPU不像CPU拥有复杂的硬件以执行分支预测，CPU擅长处理复杂的控制流

如果我们的核函数中出现：一个线程束中16个线程条件为真，16个线程条件为假的分支语句,就会出现在同一线程束中的线程执行不同的指令,被称为**线程束分化**

如果一个线程束中的线程产生分化,线程束将连续执行每一个分支路径,而禁用不执行这一路径的线程。线程束分化会导致性能明显地下降。条件分支越多, 并行性削弱越严重。

> 这种情况下在硬件层面依然符合"SIMT"，实际上GPU会跑IF-ELSE两个分支的路径，再根据谓词寄存器存储的条件真值mask-out，决定结果是否写回，但是这样就会导致条件不同的线程空转，造成了资源浪费。为了获得最佳的性能，应该避免在同一线程束中有不同的执行路径。


# ✅ 一、cudaEvent_t 与 ncu 的测量对象不同（这是差异的根本原因）

## **1. cudaEvent_t 计时（时间较大）**

cudaEvent 计的是 **端到端（host 侧）消耗的时间**：

✔ kernel launch 到 queue 的时间
✔ CUDA runtime/driver 调度
✔ CPU/GPU 同步（event record/join）
✔ GPU 执行 kernel 的时间

你的 kernel 极小，只耗 **2 微秒左右**，
但 **launch + runtime 阶段通常需要 10–30 微秒**。

因此你测到：

```
mathKernel1 ≈ 25.6 us
mathKernel2 ≈ 20.48 us
```

→ **大部分时间不是 kernel 在跑，而是 launch/sync 的固定成本。**

---

## **2. ncu Duration（时间更小且差异更小）**

ncu 只统计：

### **纯 GPU 设备端的 kernel 执行时间（SM 执行指令的时间）**

不包含：

✘ launch
✘ driver 调度
✘ stream enqueue
✘ CPU 同步
✘ event 记录

你的代码实际 GPU 内部执行只有：

```
Duration ≈ 2.2~2.3 us
```

这才是 kernel 真正执行的时间。

→ **所以 ncu 时间自然比 cudaEvent 小得多**。

---

# ✅ 二、为什么 ncu 中 mathKernel1 与 mathKernel2 几乎一样？

因为你的 kernel **太小、太轻量**。

每个线程只有几条指令：

```cpp
a=100; or b=200;
c[tid]=a+b;
```

共 64 个线程（刚好 2 个 warp），真正执行指令的 cycles 只有约 2000–3000 个周期。

### ✔ warp divergence 本来就会让同一个 warp 顺序执行两个分支

但这里每个分支只是几条指令，因此：

→ 分化额外带来的 cycles 几乎可以忽略
→ 所以 ncu 中两者 Duration 都是约 2.2 μs

**因此不可能在 ncu 看到明显差异。**

---

# ✅ 三、为什么 cudaEvent 中 mathKernel1 > mathKernel2？

你看到：

```
25.6 us  vs  20.48 us
```

这不是因为 kernel1 真的慢 **5 微秒**。

真实 GPU 运行差只有不到 **0.1 微秒**。

真正导致 20–25 微秒差异的是：

### **launch/sync 抖动（调度噪声）**

* runtime 调度延迟
* 驱动内部队列波动
* event 记录同步的系统开销
* 首次调用后的缓存差异

这些抖动轻松能达到 ±5–10 微秒。

在这么小的 kernel 下：

→ **launch 噪声 >> kernel 本体差异**

因此 cudaEvent 下的差值根本不能用来判断哪个 kernel 更快。

---

# ✅ 四、warp divergence 在你的测试中为什么看不出来？

因为：

### 1. 线程数太少（仅 64）

GPU 空闲 99% 以上，SM 只执行极少一组 warp。

### 2. 分支中的计算太少（极短）

两个分支都是几条算术指令，总执行 cycles 非常小。

### 3. divergence 成本被「启动开销噪声」完全淹没

你测量的 20–25 微秒中，

> 真正 kernel 执行只占 **2 微秒**
> 启动和同步占 **18–23 微秒**

→ 真实 warp 分化开销被完全「盖掉」。

---

# ✅ 五、总结

### **cudaEvent_t 计时 vs ncu Duration**

| 项目                     | cudaEvent（你测到 20–25 μs） | ncu Duration（你测到 2.2 μs） |
| ---------------------- | ----------------------- | ------------------------ |
| 是否包含 kernel 启动开销       | ✔ 包含                    | ✘ 不包含                    |
| 是否包含 CPU/GPU 同步        | ✔ 包含                    | ✘ 不包含                    |
| 是否包含 driver/runtime 调度 | ✔ 包含                    | ✘ 不包含                    |
| 是否是真正的 kernel 执行时间     | ❌ 被 launch 噪声污染         | ✔ 纯 SM 执行时间              |
| 数量级                    | 10–30 μs                | 2–3 μs                   |
| 分化差异能否体现？              | ✘ 完全被噪声覆盖               | ✔ 但你这个 kernel 太小导致差异极低   |

---


# 实例

## Branch Efficiency

ncu 统计数据，发现kernel1和kernel2的branch efficiency均为0，分支根本没有真正产生 warp divergence

这可能是ncu 把 if(a)/else(b) 全部编译成 predicated execution！不是 branch！

反汇编出SASS指令之后确认

产生结果后使用谓词寄存器mask掉其他分支的语句，在分支语句较少的时候很容易被优化成这种形式

| 情况                           | warp 内线程行为 | branch efficiency |
| ---------------------------- | ---------- | ----------------- |
| 所有线程走同一分支                    | 完全一致       | 100%              |
| warp 半半分开                    | 50% / 50%  | 50%               |
| 每个线程都不同                      | 完全不同       | 0%                |
| 只有 predicated 指令，没有真正 BRA/IF | 没有真实分支     | 0% （profiler）     |


## IMC Stall

程序的瓶颈在Immediate Constant Cache (IMC) Stall 超过 70%–85%！

每条指令里的常数（如 100.0f / 200.0f）实际上以 immediate constant 的形式读入 SASS。

当 warp 内每个线程使用不同 immediate 时 → 会序列化。

grid 太小（1 block），occupancy 极低 → 低 latency hiding

```
Active Warps Per SM ≈ 1
Achieved Occupancy ≈ 4%
```