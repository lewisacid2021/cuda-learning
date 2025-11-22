| 指标                      | 1D kernel | 2D kernel   |
| ----------------------- | --------- | ----------- |
| Grid Size               | 128       | 1,048,576   |
| Block Size              | 128       | 256         |
| Threads 总数              | 16,384    | 268,435,456 |
| Waves per SM            | 0.13      | 2,131       |
| Achieved Occupancy      | 13%       | 66.5%       |
| Memory Throughput       | 48.77%    | 94.24%      |
| Compute (SM) Throughput | 6.24%     | 20.11%      |
| Duration                | 7.08 ms   | 3.64 ms     |

# 分析

## 线程总数与 GPU 资源利用率

- 1D kernel：仅 16,384 个线程 → 每个 SM 只有 0.13 个完整 warp 运行 → 大部分硬件资源空闲 → Achieved Occupancy 13%

- 2D kernel：总线程数 2.68 亿 → 每个 SM 运行 2,131 个 warp → GPU 资源充分利用 → Achieved Occupancy 66.5%

2D kernel 在线程数上大幅度超过 1D kernel，使 SM 活跃度和吞吐率大幅提高。

## 内存吞吐率

- 1D kernel：48.77%，没有饱和 GPU 内存带宽

- 2D kernel：94.24%，几乎满速访问 DRAM

2D kernel 的线程排列使得内存访问更加连续/并行，warp 内访问更好地 coalesced，从而提高内存吞吐率。

## 计算吞吐率

- 1D kernel：6.24%，计算单元几乎没利用

- 2D kernel：20.11%，计算吞吐提高，但仍有空间

- 说明计算量本身还是很小，每线程只做一次加法，所以 SM 计算资源没有被完全占满（Memory-bound）。

## 执行时间

- 1D kernel：7.08 ms

- 2D kernel：3.64 ms

尽管线程总数多了几个数量级，执行时间减半，说明 1D kernel 的低 occupancy 严重拖慢了整体执行速度，而 2D kernel 的高 occupancy 和内存利用率大大提升了效率。

# 总结

核心原因：1D kernel 线程太少 → SM 空闲 → 内存吞吐不高 → 执行慢；2D kernel 线程充足 → SM 利用率提升 → 内存吞吐高 → 执行快。