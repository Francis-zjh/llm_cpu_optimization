from __future__ import annotations

import json
import math
import os
import sys
import time
import gc
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.utils import extract_keys_and_values

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_NAME = "EleutherAI/pythia-70m"
GENERATION_TOKENS = 1024
DATA_DIR = Path(__file__).parent / "data"

_MAX_LOGICAL_CPU = os.cpu_count() or 16
PHASE1_CORES = [c for c in [1, 2, 4, 8, 16, 32] if c <= _MAX_LOGICAL_CPU]
PHASE1_REPEATS = 1
PHASE2_REPEATS = 3

SINGLE_CONFIGS = [
    "Baseline",
    "Quant_only",
    "FP16_only",
    "SnapKV_only",
    "Crosslayer_only",
    "Compile_only",
    "IPEX_only",
]

def set_thread_count(n: int) -> None:

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model, tokenizer

def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

def apply_fp16(model: torch.nn.Module) -> torch.nn.Module:
    try:
        return model.half()
    except Exception as exc:
        print(f"  [WARN] FP16 conversion failed: {exc}")
        return model

def _get_attention_modules(model: torch.nn.Module):
    layers = getattr(model.gpt_neox, "layers", [])
    modules = []
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    for idx, layer in enumerate(layers):
        attn = layer.attention
        if not hasattr(attn, "layer_idx"):
            attn.layer_idx = idx
        if not hasattr(attn, "head_dim"):
            attn.head_dim = head_dim
        if hasattr(attn, "config") and hasattr(attn.config, "num_attention_heads"):
            attn.config.num_key_value_heads = attn.config.num_attention_heads
        modules.append(attn)
    return modules

def attach_snapkv_hooks(
    model: torch.nn.Module,
    compression_ratio: float = 0.2,
    window_size: int = 16,
):
    press = SnapKVPress(
        compression_ratio=compression_ratio, window_size=window_size, kernel_size=5
    )
    stats = {"calls": 0.0, "tokens_before": 0.0, "tokens_after": 0.0}
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(model)
        except Exception:
            pass

    handles = []
    for attn in _get_attention_modules(model):
        layer_idx = attn.layer_idx

        def _hook(module, args, kwargs, output, _layer_idx=layer_idx):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            cache = kwargs.get("layer_past")
            if cache is None and len(args) > 3:
                cache = args[3]
            if hidden_states is None or cache is None:
                return output
            seq_len = hidden_states.shape[1]
            if seq_len <= press.window_size:
                return output
            try:
                keys, values = extract_keys_and_values(cache, _layer_idx)
                before = float(keys.shape[-2])
                new_keys, new_values = press.compress(
                    module, hidden_states, keys, values, output[1], kwargs
                )
                cache.layers[_layer_idx].keys = new_keys.contiguous()
                cache.layers[_layer_idx].values = new_values.contiguous()
                stats["calls"] += 1.0
                stats["tokens_before"] += before
                stats["tokens_after"] += float(new_keys.shape[-2])
            except Exception:
                pass
            return output

        handles.append(attn.register_forward_hook(_hook, with_kwargs=True))
    return handles, stats, press

def share_kv_cache_across_layer_groups(cache, groups):
    shared = 0
    for group in groups:
        if len(group) <= 1:
            continue
        leader = group[0]
        try:
            leader_keys = cache.layers[leader].keys
            leader_values = cache.layers[leader].values
            for idx in group[1:]:
                cache.layers[idx].keys = leader_keys
                cache.layers[idx].values = leader_values
                shared += 1
        except Exception:
            continue
    return shared

def maybe_optimize_runtime(model: torch.nn.Module, use_ipex: bool, use_compile: bool):
    notes = []
    if use_ipex:
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.float32)
            notes.append("ipex_applied")
        except Exception as exc:
            notes.append(f"ipex_skipped:{type(exc).__name__}")
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            notes.append("torch_compile_applied")
        except Exception as exc:
            notes.append(f"torch_compile_skipped:{type(exc).__name__}")
    return model, notes

def compute_perplexity(model, tokenizer, text, max_length=512):
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] > max_length:
            offset = torch.randint(0, input_ids.shape[1] - max_length, (1,)).item()
            input_ids = input_ids[:, offset : offset + max_length]
        was_fp16 = next(model.parameters()).dtype == torch.float16
        if was_fp16:
            model = model.float()
        with torch.inference_mode(), torch.amp.autocast("cpu", enabled=False):
            outputs = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=True)
            loss = float(outputs.loss)
        if was_fp16:
            model = model.half()
        if math.isnan(loss) or math.isinf(loss):
            return None
        return math.exp(loss)
    except Exception:
        return None

def measure_model_size_mb(model):
    try:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buf_bytes) / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def measure_generation(
    model, tokenizer, prompt,
    max_new_tokens=GENERATION_TOKENS,
    use_cross_layer=False, cross_layer_groups=None,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"]
    step_times = []

    with torch.inference_mode(), torch.amp.autocast("cpu", enabled=False):
        start = time.perf_counter()
        outputs = model(input_ids=generated, use_cache=True)
        if use_cross_layer and cross_layer_groups:
            share_kv_cache_across_layer_groups(outputs.past_key_values, cross_layer_groups)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_key_values = outputs.past_key_values
        ttft = time.perf_counter() - start

        for _ in range(max_new_tokens - 1):
            step_start = time.perf_counter()
            outputs = model(
                input_ids=next_token, use_cache=True, past_key_values=past_key_values
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            step_times.append(time.perf_counter() - step_start)

    total = time.perf_counter() - start
    tpot = sum(step_times) / max(1, len(step_times)) if step_times else None
    throughput = max_new_tokens / total if total > 0 else None
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "throughput_tok_s": throughput,
        "generated_tokens": max_new_tokens,
    }

def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

def get_config_opts(name: str):

    m = {
        "Baseline":        (False, False, False, False, False, False, []),
        "Quant_only":      (False, False, True,  False, False, False, []),
        "FP16_only":       (False, False, False, True,  False, False, []),
        "SnapKV_only":     (False, False, False, False, True,  False, []),
        "Crosslayer_only": (False, False, False, False, False, True,  [[4, 5]]),
        "Compile_only":    (False, True,  False, False, False, False, []),
        "IPEX_only":       (True,  False, False, False, False, False, []),
    }
    return m[name]

def run_single(config_name: str, n_threads: int, prompt: str, corpus_text: str) -> dict:
    set_thread_count(n_threads)
    use_ipex, use_compile, use_quant, use_fp16, use_snapkv, use_crosslayer, cross_groups = \
        get_config_opts(config_name)

    model, tokenizer = load_model_and_tokenizer()
    notes = [f"config:{config_name}", f"threads:{n_threads}"]
    size_fp32 = measure_model_size_mb(model)
    size_int8 = None

    if use_quant:
        model = apply_dynamic_quantization(model)
        size_int8 = measure_model_size_mb(model)
        notes.append("quantization_applied")

    if use_fp16 and not use_quant:
        model = apply_fp16(model)
        notes.append("fp16_applied")
    elif use_fp16 and use_quant:
        notes.append("fp16_skipped:quantization_overrides")

    handles = []
    press_stats = None
    if use_snapkv:
        handles, press_stats, _ = attach_snapkv_hooks(model)
        notes.append(f"snapkv_hooks={len(handles)}")

    model, runtime_notes = maybe_optimize_runtime(model, use_ipex, use_compile)
    notes.extend(runtime_notes)

    if use_compile or use_ipex:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _ = measure_generation(model, tokenizer, prompt, max_new_tokens=4)
            except Exception:
                pass

    ppl = compute_perplexity(model, tokenizer, corpus_text, max_length=512)
    gen = measure_generation(
        model, tokenizer, prompt, max_new_tokens=GENERATION_TOKENS,
        use_cross_layer=use_crosslayer, cross_layer_groups=cross_groups,
    )
    ram = rss_mb()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    snapkv_ratio = None
    if use_snapkv and press_stats and press_stats["calls"] > 0:
        avg_before = press_stats["tokens_before"] / press_stats["calls"]
        avg_after = press_stats["tokens_after"] / press_stats["calls"]
        if avg_before > 0:
            snapkv_ratio = round(1.0 - avg_after / avg_before, 4)

    del model, tokenizer
    gc.collect()

    return {
        "config": config_name,
        "n_threads": n_threads,
        "ppl": ppl,
        "ttft_s": gen["ttft_s"],
        "tpot_s": gen["tpot_s"],
        "throughput_tok_s": gen["throughput_tok_s"],
        "ram_rss_mb": ram,
        "model_size_mb": size_fp32,
        "quantized_size_mb": size_int8,
        "snapkv_effective_ratio": snapkv_ratio,
        "notes": notes,
    }

def _mean(vals):
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None

def main():
    prompt = "Pythia-70M is a small language model that can still be profiled on CPU."
    dataset = "wikitext"

    local_path = DATA_DIR / f"{dataset}_corpus.txt"
    if not local_path.exists():
        print(f"[ERROR] Local corpus not found at {local_path}")
        print("Run cpu_all_optimized.py first to download and cache the data.")
        sys.exit(1)

    corpus_text = local_path.read_text(encoding="utf-8").strip()
    print(f"[OK] Loaded local corpus ({len(corpus_text)} chars)")

    results: dict = {
        "model_name": MODEL_NAME,
        "generation_tokens": GENERATION_TOKENS,
        "dataset": dataset,
        "environment": {
            "torch": torch.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "cpu_count": os.cpu_count(),
            "has_ipex": HAS_IPEX,
        },
        "sweep_config": {
            "phase1_cores": PHASE1_CORES,
            "phase1_repeats": PHASE1_REPEATS,
            "phase2_repeats": PHASE2_REPEATS,
        },
        "phase1": {},
        "phase2": {},
        "final_summary": {},
    }

    print("\n" + "=" * 70)
    print("  PHASE 1: COARSE CORE COUNT SWEEP")
    print(f"  Configs: {SINGLE_CONFIGS}")
    print(f"  Cores:   {PHASE1_CORES}")
    print(f"  Repeats: {PHASE1_REPEATS}")
    print("=" * 70)

    phase1_data: dict = {}
    optimal_p1: dict = {}

    torch.manual_seed(7)

    for cfg in SINGLE_CONFIGS:
        print(f"\n>>> {cfg}")
        cfg_runs: dict = {}
        for nc in PHASE1_CORES:
            print(f"  {nc:2d} threads ...", end=" ", flush=True)
            t0 = time.perf_counter()
            rd = run_single(cfg, nc, prompt, corpus_text)
            elapsed = time.perf_counter() - t0
            thpt = rd["throughput_tok_s"] or 0
            print(f"Thpt={thpt:7.2f} tok/s  RAM={rd['ram_rss_mb']:7.1f} MB  ({elapsed:.0f}s)")
            cfg_runs[str(nc)] = rd

        phase1_data[cfg] = cfg_runs

        best = max(PHASE1_CORES, key=lambda c: cfg_runs[str(c)]["throughput_tok_s"] or 0)
        best_thpt = cfg_runs[str(best)]["throughput_tok_s"] or 0
        optimal_p1[cfg] = {"optimal_cores": best, "peak_thpt_tok_s": best_thpt}
        print(f"  >>> {cfg}: optimal = {best:2d} cores @ {best_thpt:.2f} tok/s")

    results["phase1"] = {
        "core_counts": PHASE1_CORES,
        "configs": SINGLE_CONFIGS,
        "results": phase1_data,
        "optimal_per_config": optimal_p1,
    }

    print("\n" + "=" * 70)
    print("  PHASE 2: FINE CORE COUNT SWEEP (3 repeats)")
    print("=" * 70)

    phase2_data: dict = {}
    final_optimal: dict = {}

    for cfg in SINGLE_CONFIGS:
        peak = optimal_p1[cfg]["optimal_cores"]

        fine = sorted({
            max(1, min(max(PHASE1_CORES), peak + d))
            for d in range(-2, 3)
        })
        print(f"\n>>> {cfg}  (Phase 1 peak = {peak}, fine = {fine})")

        cfg_fine: dict = {}
        for nc in fine:
            runs_list = []
            for rep in range(PHASE2_REPEATS):
                print(f"  {nc:2d} threads  rep {rep+1}/{PHASE2_REPEATS} ...", end=" ", flush=True)
                torch.manual_seed(7 + rep)
                t0 = time.perf_counter()
                rd = run_single(cfg, nc, prompt, corpus_text)
                print(f"Thpt={rd['throughput_tok_s']:7.2f} tok/s  ({time.perf_counter()-t0:.0f}s)")
                runs_list.append(rd)

            agg = {
                "runs": runs_list,
                "mean_throughput_tok_s": _mean([r["throughput_tok_s"] for r in runs_list]),
                "mean_ttft_s":          _mean([r["ttft_s"] for r in runs_list]),
                "mean_tpot_s":          _mean([r["tpot_s"] for r in runs_list]),
                "mean_ram_rss_mb":      _mean([r["ram_rss_mb"] for r in runs_list]),
                "mean_ppl":             _mean([r["ppl"] for r in runs_list]),
            }
            cfg_fine[str(nc)] = agg

        phase2_data[cfg] = cfg_fine

        best_n = max(cfg_fine.keys(), key=lambda k: cfg_fine[k]["mean_throughput_tok_s"])
        best_thpt = cfg_fine[best_n]["mean_throughput_tok_s"]
        final_optimal[cfg] = {
            "optimal_cores": int(best_n),
            "peak_thpt_tok_s": best_thpt,
        }
        print(f"  >>> {cfg}: FINAL optimal = {best_n:2s} cores @ {best_thpt:.2f} tok/s")

    results["phase2"] = {
        "repeats": PHASE2_REPEATS,
        "results": phase2_data,
        "final_optimal": final_optimal,
    }

    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: Optimal Core Count per Configuration")
    print("=" * 70)
    print(f"  {'Config':<18} {'Opt Cores':>10} {'Peak Thpt':>10} {'1-core Thpt':>11} {'Speedup':>8} {'Scaling Eff':>11}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*11} {'-'*8} {'-'*11}")

    summary_rows = []
    for cfg in SINGLE_CONFIGS:

        p2 = final_optimal.get(cfg, {})
        opt_cores = p2.get("optimal_cores", optimal_p1[cfg]["optimal_cores"])
        peak = p2.get("peak_thpt_tok_s", optimal_p1[cfg]["peak_thpt_tok_s"])

        thpt_1 = phase1_data[cfg].get("1", {}).get("throughput_tok_s") or 0
        speedup = peak / thpt_1 if thpt_1 > 0 else 0
        efficiency = speedup / opt_cores if opt_cores > 0 else 0

        summary_rows.append({
            "config": cfg,
            "optimal_cores": opt_cores,
            "peak_thpt_tok_s": round(peak, 2),
            "thpt_1core_tok_s": round(thpt_1, 2),
            "speedup_vs_1core": round(speedup, 2),
            "scaling_efficiency": round(efficiency, 4),
        })
        print(f"  {cfg:<18} {opt_cores:>10} {peak:>10.2f} {thpt_1:>11.2f} {speedup:>7.2f}x {efficiency:>10.4f}")

    results["final_summary"] = {
        "rows": summary_rows,
        "note": (
            "Scaling efficiency = speedup / optimal_cores. "
            "1.0 = perfect linear scaling."
        ),
    }

    output_path = Path("cpu_core_sweep_results.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Results saved to {output_path}")
    print(f"  Estimated runtime: Phase 1 ~ 30-60 min, Phase 2 ~ 90-180 min")
    print("=" * 70)

if __name__ == "__main__":
    main()
