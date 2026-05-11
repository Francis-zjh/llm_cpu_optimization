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

- Python 3.10+
- PyTorch 2.x (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- Transformers, Datasets (`pip install transformers datasets`)
- kvpress (`pip install kvpress`)
- psutil, tqdm (`pip install psutil tqdm`)
- 可选: IPEX (`pip install intel_extension_for_pytorch`)

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
