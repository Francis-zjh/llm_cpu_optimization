# 实验尝试历史（trial history）

本文档仅按真实实验尝试次数记录。  
当前只进行过一次 trial，因此只保留 Trial 1。

---

## Trial 1：CPU 全流程组合优化（唯一一次尝试）

### 目标

在不训练模型参数的前提下，针对 pythia-70m 在 CPU 上执行组合优化，并测量：
PPL、TTFT、TPOT、Throughput、FLOPs、RAM。

### 环境与基础设置

1. OS：Windows
2. conda 环境：llm-cpu-opt（Python 3.10）
3. 模型：EleutherAI/pythia-70m
4. 数据集：wikitext-2-raw-v1（PPL 与推理测试）
5. 镜像设置：
   - HF_ENDPOINT=https://hf-mirror.com
   - HF_HUB_DISABLE_SYMLINKS_WARNING=1

### 本次 trial 使用的加速方式与参数

1. GQA 仿真
   - target_kv_heads=4
   - mix_alpha=0.15（保守混合改写）
2. KV 压缩（SnapKV）
   - compression_ratio=0.5
   - window_size=16
3. 跨层 KV 共享
   - 层组 [0,1,2] 与 [3,4,5]
4. 运行时优化尝试
   - IPEX：尝试启用
   - torch.compile：尝试启用（依赖 cl 编译器）

### 实验结果

Baseline：
1. PPL: 63.5125
2. TTFT: 0.0471 s
3. TPOT: 0.0184 s/token
4. Throughput: 45.5229 tok/s
5. RAM: 737.76 MB
6. FLOPs: 1969209120

Optimized：
1. PPL: 415759.5620
2. TTFT: 0.0405 s
3. TPOT: 0.0190 s/token
4. Throughput: 46.0371 tok/s
5. RAM: 745.02 MB
6. FLOPs: 1968837600

运行备注：
1. ipex_skipped:ModuleNotFoundError
2. torch_compile_skipped:no_cl_compiler
3. cross_layer_shared_layers=4
4. snapkv_avg_tokens_before=55.33
5. snapkv_avg_tokens_after=27.33
6. snapkv_effective_ratio=0.5060

### 问题分析

1. 质量显著退化（PPL 爆炸）是本次 trial 最核心问题。
2. 可能原因：
   - 小模型上同时叠加 GQA 仿真 + 大比例 KV 压缩 + 跨层共享，语义失真叠加。
   - SnapKV 压缩比例 0.5 偏激进。
   - 运行时优化未生效（IPEX 与 compile 均跳过），导致质量代价没有换来明显速度收益。
3. 速度收益不明显且不稳定，不足以支持“组合优化有效”的结论。

### 本次 trial 的结论

1. 该方案可跑通，但当前参数组合不适合直接作为最终优化方案。
2. 这是可用于报告的有效负结果，应在论文中诚实呈现。
3. 下一次新实验应作为 Trial 2 另起记录，不在本文件中提前虚构。

---

## Trial 2：CPU 严格消融分析与温和设置探索

### 目标

基于 Trial 1 的退化问题分析，重构测试代码，强制加入模块独立的消融（Ablation）实验：分别考察 GQA 仿真、低压缩率 SnapKV、局部相邻跨层共享组合。并且增加 `pg-19` 作为长上下文比较来更严谨论证。

### 环境与基础设置

- 与 Trial 1 一致。针对网络原因触发长连接验证超时，启用了脱机的环境阻断 `HF_HUB_OFFLINE` 以及手动拦截，用以规避平台验证受阻。

### 本次 trial 使用的加速方式与参数

1. 拆解分析模式：
   - 保留了独立基线（Baseline）。
   - 只开 GQA (`Only GQA`)。
   - 只开 SnapKV (`Only SnapKV`)，并将 `compression_ratio` 由 0.5 温和下调至 0.2。
   - 只开 CrossLayer (`Only CrossLayer`)，且仅安全共享层组 [4, 5]。
   - All Optimized (组合上述温和参数)。
2. 指标记录分离：同时跟踪 Wiektex 测试长文本段以及 Pg-19 等长推断记录。

### 实验结果与诊断

1. **PPL剧毒定位**：通过独立的 `Only GQA` 可以清楚看到 PPL 从 63.51 爆炸式提升到 415759.56，证实了是小范围内缺乏优调的模拟 GQA 层平均在小模型上引发全盘崩溃（这是此 Trial 的最大发现）。
2. **温和配置平稳性**：由于把 SnapKV 参数降到了 0.2 并将共享层限制在部分浅表后段，在 `Only SnapKV` 和 `Only CrossLayer` 单独列上，PPL 是极为正常的 63.51（短文段）乃至 101.26（推演文本）。且其内存截断对输入损失计算并不会触发污染，具备了真正上机的潜力。
3. **环境约束被透明化记录**：执行自动探捕后优雅回调，没有由于强算子不可用抛出异常，如时效反馈了 Windows 缺失 `cl` 编译器导致的 `ipex` 弃用代价（故吞吐速率不佳）。

### 本次 trial 的结论

该模型完成了从一揽子测试转变为高解释度分析报告。明确指出了单纯数值组合在轻量级模型内的破坏性机制所在（权重模拟）。同时通过安全的动态 KV 生成缓存介入留有极佳余地，可用于高层报告撰写支撑。
