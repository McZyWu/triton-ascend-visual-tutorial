# Triton Ascend Visual Lab

一个中文交互式教程，覆盖 Triton-Ascend 官方 Quick Start，并扩展到：

- `program_id / grid / offset / mask` 的逐槽位可视化
- GM → UB → 计算 → GM 的搬运模拟
- persistent grid 并行任务分配
- UB 峰值工作集估算器与溢出排查
- 来自 `sgl-project/sgl-kernel-npu` 的 RMSNorm、SwiGLU、融合 Argmax+Softmax、KV cache 写入案例
- Triton API 到 PyTorch 心智模型的对照表

## 本地运行

```bash
pnpm install
pnpm dev
```

## 内容来源

- https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md
- https://triton-ascend.readthedocs.io/en/latest/index.html
- https://github.com/sgl-project/sgl-kernel-npu

本教程为学习辅助材料；版本和硬件支持请以上游最新文档为准。
