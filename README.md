# AgentCacheLab

面向 BFCL 多轮 Agent 工作负载的 SGLang 推理系统性能实验，研究部署方式、KV Cache 容量和 HiCache Offload 对吞吐、端到端任务延时与 Prefix Cache 命中率的影响。

## 已实现内容

- 五组实验配置：单 GPU 三档 KV Cache、TP=2、单 GPU + HiCache。
- 使用官方 BFCL multi-turn 数据，可固定随机子集和并发数。
- 通过运行时包装 BFCL 生成器，记录每个完整 Agent 任务的端到端耗时，不修改 BFCL 源码。
- 自动启动/停止 SGLang，保存启动命令、版本环境、服务日志和返回码。
- 在负载前后抓取 SGLang Prometheus metrics，并每秒采样 GPU 显存、利用率和功耗。
- 自动计算 Input/Output/Total token throughput、平均/P90 任务延时、Prefix Cache 命中率、实际 KV Cache token 容量、峰值 GPU 显存和任务成功率。

## 目录

```text
configs/                 实验矩阵
scripts/                 BFCL 计时包装、子集选择和服务器安装脚本
src/agent_serving_study/ 编排、指标解析和汇总代码
tests/                   不需要 GPU 的轻量测试
artifacts/               每次实验的原始结果（gitignored）
```

## 服务器准备

建议在 Linux + CUDA 12.4 + A100 环境执行。安装前先确认课程镜像是否已经包含 SGLang；若已包含，应保留镜像中经过验证的版本，避免盲目升级 CUDA/PyTorch。

```bash
git clone https://github.com/EriccirEgyz/agent-serving-performance-study.git
cd agent-serving-performance-study

# 全新环境可使用；已有课程环境建议手动安装缺失包。
bash scripts/server_setup.sh
source .venv/bin/activate
```

若使用现有环境，最低限度为：

```bash
python -m pip install -e .
python -m pip install bfcl-eval
# 仅在环境没有 SGLang 时安装，并根据服务器 CUDA 版本选择兼容版本。
python -m pip install "sglang[all]"
```

记录版本，报告中需要使用：

```bash
python --version
python -m pip show sglang bfcl-eval torch
nvidia-smi
```

## 选择固定 BFCL 子集

默认从四个 BFCL v4 multi-turn 类别各抽取 12 条，共 48 条。选择过程固定随机种子，生成的 ID 文件默认不提交，以免不同 BFCL 数据版本混用。

```bash
python scripts/select_bfcl_subset.py --per-category 12 --seed 2026
```

如只想冒烟测试：

```bash
python scripts/select_bfcl_subset.py --per-category 1 --seed 2026
```

## 查看实验矩阵

```bash
agent-serving-study list
agent-serving-study command single_gpu_medium
```

默认矩阵如下：

| 实验 | GPU | TP | `mem-fraction-static` | HiCache |
|---|---:|---:|---:|---|
| `single_gpu_small` | 1 | 1 | 0.50 | No |
| `single_gpu_medium` | 1 | 1 | 0.70 | No |
| `single_gpu_large` | 1 | 1 | 0.85 | No |
| `tp2_medium` | 2 | 2 | 0.70 | No |
| `single_gpu_medium_hicache` | 1 | 1 | 0.70 | 16 GiB host cache |

`mem-fraction-static` 同时包含模型权重和 KV Cache 池，报告中应从每次 `server.log` 与 metrics 记录真实 `max_total_num_tokens`，不能把 0.50/0.70/0.85 直接称为 KV Cache 容量。

## 运行

先做一组小规模基线：

```bash
agent-serving-study run single_gpu_medium
```

确认 BFCL 正确率、日志和显存都合理后，再依次运行：

```bash
agent-serving-study run single_gpu_small
agent-serving-study run single_gpu_large
agent-serving-study run tp2_medium
agent-serving-study run single_gpu_medium_hicache
```

每次运行都会新建：

```text
artifacts/<timestamp>-<experiment>/
  run_metadata.json
  server.log
  bfcl_generate.log
  bfcl_evaluate.log
  metrics_before.prom
  metrics_after.prom
  gpu_samples.csv
  bfcl/result/...
  bfcl/score/...
  summary.json
```

汇总全部实验：

```bash
agent-serving-study summarize-all --output artifacts/summary.csv
```

正式数据建议每个配置独立重复三次。每次都由编排器重新启动服务，因此 Prefix Cache 从空状态开始；任务集合、顺序、并发数、模型和生成参数保持不变。

## 指标定义

- 端到端任务延时：BFCL `handler.inference()` 的墙钟时间，覆盖一个任务的多轮模型调用和工具执行。
- Token throughput：负载期间 SGLang `prompt_tokens_total` 与 `generation_tokens_total` counter 增量除以 BFCL 总墙钟时间。
- Prefix Cache 命中率：`cached_tokens_total / prompt_tokens_total`；同时按 device/host/storage 保存命中 token，便于区分 GPU Radix Cache 和 HiCache。
- 成功率：生成阶段未发生 inference error 的任务比例；最终 benchmark score 以 BFCL `evaluate --partial-eval` 输出为准。

注意：`summary.json` 中的成功率是系统运行成功率，不等同于 BFCL 正确率。

## 常见调整

- OOM：优先降低 `mem_fraction_static` 或 benchmark 并发数。
- TP=2 通信异常：在对应配置的 `extra_server_args` 中加入 `--enable-p2p-check`。
- HiCache 主机内存不足：降低 `hicache_size_gb`，并在报告中记录实际值。
- 服务参数与课程镜像版本不一致：运行 `python -m sglang.launch_server --help`，只修改 `configs/experiments.json` 中的集中配置。
- 端口冲突：修改 `study.port`；编排器会同步 BFCL endpoint。

## 本地轻量检查

不需要 GPU：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
agent-serving-study list
```

## 方法学约束

1. 不在不同配置之间更换模型、任务 ID、并发数或生成参数。
2. 每组实验使用相同的冷缓存起点；不要复用上一个配置的服务进程。
3. 先用小子集排除 parser、OOM 和 endpoint 问题，再固定正式子集。
4. 保留原始日志。出现 timeout、OOM 或 crash 时也作为实验结果记录，不手工删除异常样本。
5. 共享服务器上确认 GPU 空闲，并通过 `CUDA_VISIBLE_DEVICES` 避免与其他同学冲突。
