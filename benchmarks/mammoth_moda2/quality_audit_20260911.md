# MammothModa2 代码质量与边界审计

日期：2026-09-11

## 结论先行

如果审查对象是当前未提交的 low-precision candidate diff，那么它没有修改
vLLM-Omni 的通用架构。生产代码只涉及：

- `stage_input_processors/mammoth_moda2.py`
- `diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py`

另外修改了 MammothModa2 专属测试和 benchmark 文件。没有改动 scheduler、
GPU runner、connector、EngineCore、通用 serializer、request lifecycle、
cancellation、queue 或 stage topology。

但如果把整个分支历史都算进去，就不能说“从未碰过架构”：更早的
`e9634659` / `c68a9a5a` request-end payload 探索曾修改
`gpu_ar_model_runner.py`、`omni_connector_model_runner_mixin.py`、
`omni_ar_scheduler.py` 等通用路径。`422a63fd` 的作用是恢复标准
AR-to-DiT forwarding，并删除/收回那条 request-end 路径。当前待 review 的
candidate diff 不再包含这些通用架构修改，但历史事实应明确保留。

## 对照最初要求

| 原始要求 | 状态 | 证据或边界 |
| --- | --- | --- |
| 基于现有 #7087 分支工作 | 已做到 | 当前 HEAD 为 `422a63fd`，未重新从 main 开始 |
| 先调查真实 AR → DiT 路径 | 已做到 | `cross_stage_dataflow_20260910.md` |
| 不一次性实现整个 cross-stage runtime | 已做到 | 当前 candidate 没有 Layer-2/3 代码 |
| 架构修改前问“是否必要、是否有局部替代” | 对当前 candidate 已做到 | 选择 payload 表示层局部改动，没有引入 connector/handle/pool |
| 删除代码必须有强证据 | 基本做到 | 只替换 FP32 扩展；没有删除同步、生命周期或清理逻辑 |
| 正确性优先 | 已做到 | shape、row alignment、dtype、序列化 round-trip 和输出 hash 已验证 |
| 实现生产级异步 D2H/H2D | 未做到 | 只有 V100 transport feasibility microbenchmark；没有接入 runtime |
| 只保留必要同步 | 未做到/未进入生产代码 | 生产路径同步结构保持不变，Nsight 仍观察到原有同步 |
| 多 request buffer lifetime / cancellation 安全 | 未做到 | 没有新增 staging slot 或 payload handle，因此也没有声称支持 |
| prefix cache / chunked prefill | 未做到且有意保持不动 | 属于 Layer 0 语义问题，本次没有跨层处理 |
| connector 替换或 GPU-direct handoff | 未做到且不应做 | 当前 Mammoth hidden-state path 不经过 SharedMemoryConnector |
| 最小侵入式开发 | 对当前 candidate 已做到 | 2 个模型专属生产文件，diff 为 5 insertions / 11 deletions |
| 不影响其他模型 | 基本做到 | 没有修改共享接口；`ar2dit` 只有 MammothModa2 pipeline 使用 |
| 静态与功能验证 | 部分做到 | `compileall`、`diff --check` 通过；远端功能/A-B/Nsight 通过 |
| 完整项目测试/CI 认证 | 未做到 | 当前本地环境缺少 `vllm`，pytest 无法启动；不能宣称全套 CI 通过 |

## 当前 patch 的实际语义

```text
AR hidden states
→ 保持 BF16/FP16
→ EngineCore payload
→ DiT 按逻辑 token mask 选行
→ 进入 DiT 前转换为 consumer model dtype
```

这不是请求生命周期优化，也不是传输协议重构。它只避免了：

```text
BF16/FP16 → FP32 → bytes → CPU tensor → BF16/FP16
```

## 兼容性判断

当前 patch 的兼容性风险是受控的：

1. `full_hidden_states` 是 MammothModa2 专属 payload key；
2. 通用 `data_entry_keys` serializer 原本就按 raw bytes 保存 BF16/FP16；
3. DiT `forward` 仍显式转换到 refiner/transformer 的实际 dtype；
4. token-id、answer boundary、look-ahead 规则没有改变；
5. 其他模型、scheduler、connector 和 request cleanup 没有进入 diff。

仍未证明的范围：

- prefix-cache 命中后的 hidden-state reconstruction；
- chunked-prefill 的逻辑序列重组；
- 新增异步 buffer 生命周期；
- 完整项目 CI 和所有后端组合；
- 生产环境多请求流式传输。

## 质量判定

作为一个已经收束的 Layer-1 payload-volume patch，代码质量达到“可以供用户
review 和后续决定是否提交”的程度；它不是一个已经完成原始全部目标的
cross-stage runtime patch。后续如果转向模型整体 cross-lifecycle 问题，应把
本次 patch 当作独立、可回退的 Layer-1 变更，不把早期 request-end 探索和
未实现的流式设计混入新的问题。
