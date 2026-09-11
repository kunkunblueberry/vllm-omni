# MammothModa2 AR → DiT Layer-1 收口记录

日期：2026-09-11
状态：本地验证完成；不再继续实现流式 D2H/H2D；未提交、未推送。

## 本地代码状态

当前本地分支仍为 `kunkun/expend_MammothModa2_lifestyle`，HEAD 为
`422a63fd`。验证后的低精度 candidate 仍是未提交修改，涉及：

- `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py`
- `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py`
- 对应单元测试和 `benchmarks/mammoth_moda2/`

candidate 源码已经作为远端验证快照保存在归档的
`verified_source/` 下；两个关键源码文件的本地 SHA256 与归档内副本一致。

## 已保留的远端证据

压缩归档：

`benchmarks/mammoth_moda2/mammoth-moda2-artifacts-20260911.tar.gz`

- 大小：约 42 MB
- SHA256：
  `081651e4e08a54fe6a2b20833a84be5d0d132e16fd9d9a1b7ebbd877214734ef`

已展开目录：

`benchmarks/mammoth_moda2/remote_results_20260911/mammoth-moda2-artifacts-20260911/`

展开后约 117 MB，共 73 个文件，包含：

- 真实 MammothModa2 baseline/candidate 的 3-run 和 10-run A/B；
- Layer-1 payload microbenchmark 的 baseline/candidate 原始 JSON；
- baseline/candidate 的有效 Nsight `trace.nsys-rep` 和 `trace.sqlite`；
- Nsight 汇总报告；
- V100 precision compatibility proxy 的报告、JSON 和日志；
- 运行日志、输出图片、退出码；
- profile shim；
- candidate 验证源码、benchmark 脚本和部署配置；
- 远端 worktree commit/status 和逐文件 SHA256 清单。

模型权重、Python 虚拟环境和完整 worktree 没有下载：它们体积大，且不是复核这次结论所必需的证据。

## 已验证结论

本次实际保留的是 Layer 1 的低精度 payload 优化：

```text
AR hidden states
→ 保持 BF16/FP16
→ ar2dit
→ EngineCore payload
→ DiT 最终消费前转换为 model dtype
```

移除了中间的 FP32 扩展。结果：

| 指标 | Baseline | Candidate |
| --- | ---: | ---: |
| Layer-1 wire payload | 29.88 MB | 14.94 MB |
| 序列化 p50 | 4.896 ms | 2.402 ms |
| 反序列化 p50 | 5.099 ms | 2.372 ms |
| pinned H2D p50 | 2.492 ms | 1.226 ms |
| 端到端平均耗时（10-run） | 6.3675 s | 6.3067 s |

真实端到端 A/B 的平均差约为 `-0.96%`，但样本波动较大，不能声称统计显著的端到端加速。所有对应输出图片 hash 一致，没有观察到精度/结果回归。

Nsight 进一步确认：

- Stage-1 H2D payload：`-50.0%`
- 总 H2D bytes：`-44.0%`
- GPU H2D 时间：`-10.9%`
- `cudaMemcpyAsync` API 时间：`-9.85%`
- D2H bytes：不变
- Stage-0/Stage-1 kernel 时间：基本不变
- GPU allocation events：不变
- `cudaStreamSynchronize` 次数：仍为 1084

因此这不是流式传输实现，而是已经有证据支持的 payload-volume 优化。

## 明确停止的方向

以下内容没有实现，也不应在这次收口中被误认为已实现：

- 生产路径的 pinned staging；
- 跨 stage CUDA event handoff；
- 真正的异步 D2H/H2D 流式传输；
- cross-request buffer pool/reuse；
- cancellation、backpressure、stale-event 防护；
- prefix cache hidden-state reconstruction；
- chunked prefill 对齐；
- SharedMemoryConnector/MooncakeConnector 替换；
- GPU-direct payload handle。

原因是当前 Mammoth hidden-state 条件路径经过 EngineCore bytes payload，而不是
`SharedMemoryConnector`。继续做流式传输会跨入 Layer 2 的 request/buffer
lifecycle 合同，已经超出本次 Layer 1 优化的安全边界。

## 建议的 review 顺序

1. 先看 `verified_source/` 中的两个代码文件和当前本地 diff；
2. 再看 `benchmarks/mammoth_moda2/cross_stage_dataflow_20260910.md`；
3. 用 `results/mammoth-moda2-nsys-ab-20260911.md` 对照两份 Nsight trace；
4. 最后查看 `layer1-*metrics_100.json` 和 `ab-*-10r-20260911/metrics.json`。

服务器端实验进程已经结束，最后确认 GPU 空闲；本轮不再对服务器做代码或运行时改动。
