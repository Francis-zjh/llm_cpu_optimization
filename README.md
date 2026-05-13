# CPU 推理全组合优化实验 — 个人作业提交

## 文件清单

| 文件                               | 说明                     |
| -------------------------------- | ---------------------- |
| `实验报告.md`                        | 完整实验报告                 |
| `cpu_all_optimized.py`           | 全组合优化实验主脚本（22 配置，断点续跑） |
| `cpu_core_sweep.py`              | 最优 CPU 核数扫描预实验脚本       |
| `cpu_all_opt_results_性能.json`    | 全组合实验结果（最佳性能模式，主实验）    |
| `cpu_all_opt_results_能效.json`    | 全组合实验结果（最佳能效模式，对比）     |
| `cpu_core_sweep_results_性能.json` | 核数扫描结果（最佳性能模式）         |
| `cpu_core_sweep_results_能效.json` | 核数扫描结果（最佳能效模式）         |
| `README.md`                      | 本文件 — 使用说明             |

**注：** 这里的“性能”与“能效”的意思是电脑电源的“最佳性能”与“最佳能效”模式。为了对比电源模式对性能的影响，我分别把电源设成不同的模式进行了两次实验。

## 环境要求

```bash
# 一键安装所有依赖
pip install -r requirements.txt
```

依赖清单详见同目录下的 [`requirements.txt`](requirements.txt)，核心依赖如下：

- Python 3.10+
- PyTorch 2.x（CPU 版）、Transformers、Datasets
- kvpress（SnapKV 实现）
- psutil、tqdm
- 可选：IPEX（`intel_extension_for_pytorch`，本机无 AVX-512 支持时以降级模式运行）

## 快速运行

```bash
# 最优 CPU 核数扫描
python cpu_core_sweep.py

# 全组合优化实验（22 配置 × 2 数据集 × 3 重复，可以通过最优 CPU 核数扫描实验确定的最优 CPU 数量来配置为各种优化算法分配的 CPU 核数）
python cpu_all_optimized.py

# 查看实验环境
python -c "import torch; print(torch.__version__); import os; print(os.cpu_count())"
```

## 实验设计

- **模型**: EleutherAI/pythia-70m (70M)
- **数据集**: wikitext-2-raw-v1, pg19-test
- **优化方法**: INT8 量化, FP16 半精度, SnapKV, CrossLayer KV 共享, torch.compile, IPEX
- **配置数**: 22 种（含基线和全组合）
- **重复**: 每配置 3 次，取均值±标准差
- **断点续跑**: 支持中断后恢复，不丢失已有结果

## 关键超参数说明

以下参数位于 `cpu_all_optimized.py` 文件顶部的 **User configuration** 区域（约第 48-54 行），修改后直接运行即可：

```python
MODEL_NAME = "EleutherAI/pythia-70m"    # 模型名称（支持 HuggingFace 任意模型）
RESULT_PATH = Path("cpu_all_opt_results.json")  # 结果输出路径
GENERATION_TOKENS = 1024                # 固定生成长度（tokens）
REPEATS = 3                             # 每配置重复次数
RUN_SEQLEN_SWEEP = True                 # 是否运行序列长度扫描
SEQ_LENGTHS = [128, 256, 512, 1024]    # 序列长度扫描点
```

各参数含义：

| 参数 | 配置位置（代码变量） | 说明 |
|------|--------------------|------|
| 生成长度 | `GENERATION_TOKENS` = 1024 | 核心消融实验的固定生成长度，覆盖中长文本生成场景。增大可观察长文本下的内存压力，减小可加速实验迭代 |
| 重复次数 | `REPEATS` = 3 | 每配置重复 3 次取均值±标准差。可设为 1（快速验证）或 5+（追求统计精度） |
| 随机种子 | `torch.manual_seed(7 + rep)` | 不同重复使用不同种子（7, 8, 9），避免采样偶然性。在 `main()` 中设置 |
| SnapKV 压缩比 | `snapkv_compression_ratio` = 0.2 | 保留 20% 的 KV 位置，窗口大小 16。在 `AblationConfig` 中针对每个配置单独设定 |
| CrossLayer 分组 | `cross_layer_groups` = [[4, 5]] | 末两层共享 KV Cache。可修改分组策略，如增大到 [[2,3,4,5]]。在 `AblationConfig` 中设定 |
| PPL 采样长度 | `max_length` = 512 tokens | 从语料随机截取 512-token 段计算困惑度。在 `compute_perplexity()` 函数中 |
| 预热长度 | 4 tokens（写死在 `measure_generation` 调用中） | Compile/IPEX 配置冷启动预热，避开 JIT 编译延迟。仅在 `use_compile` 或 `use_ipex` 时生效 |
| FLOPs 测量步数 | 4 tokens（写死在 `measure_flops` 函数中） | 短序列 profiler 测量，降低开销。INT8 配置的 FLOPs 不可靠（profiler 无法统计 INT8 算子） |
| 线程分配 | `n_threads` = 1/4/8 | 在 `AblationConfig` 中每个配置独立设定。Baseline 用 1 核，Baseline_opt/FP16 用 8 核，其余统一用 4 核 |

### 各优化方法的参数配置

每种优化方法对应 `AblationConfig` 中的一个或多个开关：

| 优化方法 | 配置开关 | 说明 |
|---------|---------|------|
| INT8 动态量化 | `use_quantization=True` | 在 `main()` 的 ablations 列表中为每个配置单独开启 |
| FP16 半精度 | `use_fp16=True` | 与 `use_quantization` 互斥（量化优先） |
| SnapKV | `use_snapkv=True` | 配合 `snapkv_compression_ratio` 和 `snapkv_window_size` 调节压缩强度 |
| CrossLayer | `use_cross_layer=True` | 配合 `cross_layer_groups` 指定哪些层共享 |
| torch.compile | `use_compile=True` | 编译模式可选 `"reduce-overhead"`，在 `maybe_optimize_runtime()` 中 |
| IPEX | `use_ipex=True` | 自动检测是否安装，未安装则跳过 |

如需增减或修改实验配置，直接在 `main()` 函数的 `ablations` 列表中添加/删除 `AblationConfig` 实例即可。
