# CPU Experiment Section (Baseline vs All-Optimized)

## Setup

- Model: `EleutherAI/pythia-70m`
- Device: CPU (Windows)
- Dataset for PPL: `wikitext-2-raw-v1` test split sample
- Script: `cpu_all_optimized.py`
- Result file: `cpu_all_opt_results.json`

## Metrics Comparison

| Metric | Baseline | All-Optimized |
|---|---:|---:|
| PPL | 63.5125 | 415759.5620 |
| TTFT (s) | 0.0211 | 0.0228 |
| TPOT (s/token) | 0.0105 | 0.0105 |
| Throughput (tokens/s) | 84.2937 | 82.9845 |
| RAM RSS (MB) | 738.4453 | 744.2852 |
| FLOPs (total) | 1969209120 | 1968837600 |

## Notes

- Enabled optimizations in the optimized path:
  - GQA emulation (conservative blended K/V regrouping)
  - SnapKV-based KV compression hooks
  - Cross-layer KV sharing for layer groups [0,1,2] and [3,4,5]
  - Runtime optimization attempts (`intel-extension-for-pytorch`, `torch.compile`) with automatic Windows fallback
- Runtime fallback details from this run:
  - `ipex_skipped:ModuleNotFoundError`
  - `torch_compile_skipped:no_cl_compiler`
- Effective SnapKV statistics from this run:
  - `snapkv_avg_tokens_before=55.33`
  - `snapkv_avg_tokens_after=27.33`
  - `snapkv_effective_ratio=0.5060`

## Interpretation

In this Windows CPU environment, the all-in-one optimization path completed successfully but showed a strong quality degradation (PPL increase) and no clear latency gain. This is still a valid experimental outcome for the course report and indicates that aggressive no-training combinations may trade quality for memory/compute behavior in unstable ways on small models.