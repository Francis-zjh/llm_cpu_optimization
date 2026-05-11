from __future__ import annotations

import json
import math
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"

import shutil
import time
import warnings
import gc
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Tuple

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.utils import extract_keys_and_values


MODEL_NAME = "EleutherAI/pythia-70m"
RESULT_PATH = Path("cpu_all_opt_results.json")

@dataclass
class AblationConfig:
    name: str
    use_gqa: bool = False
    gqa_target_kv_heads: int = 4
    gqa_mix_alpha: float = 0.15
    use_snapkv: bool = False
    snapkv_compression_ratio: float = 0.2
    snapkv_window_size: int = 16
    use_cross_layer: bool = False
    cross_layer_groups: list[list[int]] = field(default_factory=list)
    use_runtime_opt: bool = True

@dataclass
class RunMetrics:
    ppl: float | None
    ttft_s: float | None
    tpot_s: float | None
    throughput_tok_s: float | None
    ram_rss_mb: float | None
    flops: float | None
    generated_tokens: int
    notes: list[str]


def load_model_and_tokenizer() -> Tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model, tokenizer


def get_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    layers = getattr(model.gpt_neox, "layers", [])
    modules = []
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    for idx, layer in enumerate(layers):
        attention = layer.attention
        if not hasattr(attention, "layer_idx"):
            attention.layer_idx = idx
        if not hasattr(attention, "head_dim"):
            attention.head_dim = head_dim
        if hasattr(attention, "config") and hasattr(attention.config, "num_attention_heads"):
            attention.config.num_key_value_heads = attention.config.num_attention_heads
        modules.append(attention)
    return modules


def apply_gqa_emulation(model: torch.nn.Module, target_kv_heads: int = 4, mix_alpha: float = 0.15) -> None:
    config = model.config
    if not hasattr(config, "num_attention_heads"):
        return

    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    if num_heads <= 0 or hidden_size % num_heads != 0:
        return

    head_dim = hidden_size // num_heads
    group_size = max(1, num_heads // max(1, target_kv_heads))

    for attention in get_attention_modules(model):
        proj = attention.query_key_value
        weight = proj.weight.data.clone()
        bias = proj.bias.data.clone() if proj.bias is not None else None

        q_weight, k_weight, v_weight = weight.split(hidden_size, dim=0)
        q_weight = q_weight.contiguous()

        def regroup(projected: torch.Tensor) -> torch.Tensor:
            reshaped = projected.view(num_heads, head_dim, hidden_size)
            grouped = reshaped.view(target_kv_heads, group_size, head_dim, hidden_size).mean(dim=1)
            return grouped.repeat_interleave(group_size, dim=0).reshape(hidden_size, hidden_size)

        grouped_k = regroup(k_weight)
        grouped_v = regroup(v_weight)

        k_weight = (1.0 - mix_alpha) * k_weight + mix_alpha * grouped_k
        v_weight = (1.0 - mix_alpha) * v_weight + mix_alpha * grouped_v

        proj.weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)
        if bias is not None:
            q_bias, k_bias, v_bias = bias.split(hidden_size, dim=0)
            grouped_k_bias = k_bias.view(target_kv_heads, group_size, head_dim).mean(dim=1).repeat_interleave(group_size, dim=0).reshape(hidden_size)
            grouped_v_bias = v_bias.view(target_kv_heads, group_size, head_dim).mean(dim=1).repeat_interleave(group_size, dim=0).reshape(hidden_size)
            k_bias = (1.0 - mix_alpha) * k_bias + mix_alpha * grouped_k_bias
            v_bias = (1.0 - mix_alpha) * v_bias + mix_alpha * grouped_v_bias
            proj.bias.data = torch.cat([q_bias, k_bias, v_bias], dim=0)


def maybe_optimize_runtime(model: torch.nn.Module) -> Tuple[torch.nn.Module, list[str]]:
    notes: list[str] = []
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore

        model = ipex.optimize(model, dtype=torch.bfloat16)
        notes.append("ipex_applied")
    except Exception as exc:
        notes.append(f"ipex_skipped:{type(exc).__name__}")

    if os.name == "nt" and shutil.which("cl") is None:
        notes.append("torch_compile_skipped:no_cl_compiler_on_windows")
    else:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            notes.append("torch_compile_applied")
        except Exception as exc:
            notes.append(f"torch_compile_skipped:{type(exc).__name__}")

    return model, notes


def attach_snapkv_hooks(
    model: torch.nn.Module,
    compression_ratio: float = 0.5,
    window_size: int = 16,
) -> Tuple[list[Any], dict[str, float], SnapKVPress]:
    press = SnapKVPress(compression_ratio=compression_ratio, window_size=window_size, kernel_size=5)
    stats = {"calls": 0.0, "tokens_before": 0.0, "tokens_after": 0.0}
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(model)
        except Exception:
            pass

    handles: list[Any] = []
    for attention in get_attention_modules(model):
        layer_idx = attention.layer_idx

        def post_hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            layer_idx: int = layer_idx,
        ):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            cache = kwargs.get("layer_past")
            if cache is None and len(args) > 3:
                cache = args[3]
            if hidden_states is None or cache is None:
                return output

            cache_position = kwargs.get("cache_position")
            if cache_position is None and len(args) > 5:
                cache_position = args[5]
            if cache_position is not None and int(cache_position[-1]) > hidden_states.shape[1]:
                return output

            if hidden_states.shape[1] <= press.window_size:
                return output

            try:
                keys, values = extract_keys_and_values(cache, layer_idx)
                before = float(keys.shape[2])
                new_keys, new_values = press.compress(module, hidden_states, keys, values, output[1], kwargs)
                cache.layers[layer_idx].keys = new_keys.contiguous()
                cache.layers[layer_idx].values = new_values.contiguous()
                stats["calls"] += 1.0
                stats["tokens_before"] += before
                stats["tokens_after"] += float(new_keys.shape[2])
            except Exception as exc:
                warnings.warn(f"snapkv hook skipped on layer {layer_idx}: {exc}")
            return output

        handles.append(attention.register_forward_hook(post_hook, with_kwargs=True))

    return handles, stats, press


def share_kv_cache_across_layer_groups(cache: Any, groups: list[list[int]]) -> int:
    shared_layers = 0
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
                shared_layers += 1
        except Exception:
            continue
    return shared_layers


def load_corpus_text(dataset_type: str) -> str:
    try:
        if dataset_type == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5%]")
            texts = [row["text"] for row in ds if row.get("text")]
        else: # pg19
            # Attempt to load a small subset of pg19 test set
            ds = load_dataset("emozilla/pg19-test", split="test[:1]")
            texts = [row["text"] for row in ds if row.get("text")]
            
        corpus = "\n\n".join(texts).strip()
        if corpus:
            # truncate to max 2000 chars for benchmarking speed if pg19 is huge, or 4000.
            return corpus[:4000]
    except Exception as e:
        print(f"Fallback used for {dataset_type}: {e}")

    return (
        f"This is a local fallback corpus for benchmarking {dataset_type}. "
        "It is only used when the dataset download fails. "
        "The goal is to keep the pipeline runnable on Windows while preserving the benchmark flow.\n"
        "We still report the fallback in the results JSON."
    )


def compute_perplexity(model: torch.nn.Module, tokenizer: Any, text: str, max_length: int = 128) -> float | None:
    # Explicitly note this is raw CrossEntropy-based (CE) perplexity
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"]
        labels = input_ids.clone()
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, labels=labels, use_cache=True)
        loss = float(outputs.loss)
        return math.exp(loss)
    except Exception:
        return None


def measure_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 8,
    ablation: AblationConfig = None,
) -> Tuple[dict[str, float | int | None], int]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated = input_ids
    step_times: list[float] = []
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(input_ids=generated, attention_mask=attention_mask, use_cache=True)
        shared_layers = 0
        if ablation and ablation.use_cross_layer:
            shared_layers = share_kv_cache_across_layer_groups(outputs.past_key_values, ablation.cross_layer_groups)
            
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_key_values = outputs.past_key_values
        ttft = time.perf_counter() - start

        for _ in range(max_new_tokens - 1):
            step_start = time.perf_counter()
            outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
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
        "total_s": total,
        "decoded_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }, shared_layers


def measure_flops(model: torch.nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 4) -> float | None:
    try:
        from torch.profiler import ProfilerActivity, profile

        inputs = tokenizer(prompt, return_tensors="pt")
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=True)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
                for _ in range(max_new_tokens - 1):
                    outputs = model(input_ids=next_token, use_cache=True, past_key_values=past_key_values)
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    past_key_values = outputs.past_key_values
        total = 0.0
        for event in prof.key_averages():
            flops = getattr(event, "flops", None)
            if flops is not None:
                total += float(flops)
        return total if total > 0 else None
    except Exception:
        return None

def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)

def run_ablation(ablation: AblationConfig, model: torch.nn.Module, tokenizer: Any, prompt: str, corpus_text: str) -> RunMetrics:
    notes: list[str] = [f"ablation:{ablation.name}"]
    handles: list[Any] = []
    press_stats: dict[str, float] | None = None
    
    if ablation.use_gqa:
        apply_gqa_emulation(model, target_kv_heads=ablation.gqa_target_kv_heads, mix_alpha=ablation.gqa_mix_alpha)
        notes.append("gqa_applied")
        
    if ablation.use_snapkv:
        handles, press_stats, _ = attach_snapkv_hooks(model, compression_ratio=ablation.snapkv_compression_ratio, window_size=ablation.snapkv_window_size)
        notes.append(f"kvpress_hooks={len(handles)}")
        
    if ablation.use_runtime_opt:
        model, runtime_notes = maybe_optimize_runtime(model)
        notes.extend(runtime_notes)

    ppl = compute_perplexity(model, tokenizer, corpus_text, max_length=128)
    gen, shared_layers = measure_generation(model, tokenizer, prompt, max_new_tokens=8, ablation=ablation)
    flops = measure_flops(model, tokenizer, prompt, max_new_tokens=4)
    ram = rss_mb()

    # Cleanup hooks
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass

    if ablation.use_cross_layer:
        notes.append(f"cross_layer_shared_layers={shared_layers}")
        
    if ablation.use_snapkv:
        if press_stats and press_stats["calls"] > 0:
            avg_before = press_stats["tokens_before"] / press_stats["calls"]
            avg_after = press_stats["tokens_after"] / press_stats["calls"]
            ratio = 1.0 - (avg_after / avg_before if avg_before > 0 else 1.0)
            notes.append(f"snapkv_effective_ratio={ratio:.4f}")
        else:
            notes.append("snapkv_effective_ratio=0.0000")

    return RunMetrics(
        ppl=ppl,
        ttft_s=float(gen["ttft_s"]) if gen["ttft_s"] is not None else None,
        tpot_s=float(gen["tpot_s"]) if gen["tpot_s"] is not None else None,
        throughput_tok_s=float(gen["throughput_tok_s"]) if gen["throughput_tok_s"] is not None else None,
        ram_rss_mb=ram,
        flops=flops,
        generated_tokens=int(gen["generated_tokens"]),
        notes=notes,
    )


def main() -> None:
    torch.manual_seed(7)
    prompt = "Pythia-70M is a small language model that can still be profiled on CPU."

    datasets = ["wikitext", "pg19"]
    ablations = [
        AblationConfig(name="baseline", use_gqa=False, use_snapkv=False, use_cross_layer=False, use_runtime_opt=True),
        AblationConfig(name="gqa_only", use_gqa=True, gqa_target_kv_heads=4, gqa_mix_alpha=0.15, use_runtime_opt=True),
        AblationConfig(name="snapkv_only", use_snapkv=True, snapkv_compression_ratio=0.2, snapkv_window_size=16, use_runtime_opt=True),
        AblationConfig(name="crosslayer_only", use_cross_layer=True, cross_layer_groups=[[4, 5]], use_runtime_opt=True), # Mild sharing
        AblationConfig(name="all_optimized", use_gqa=True, use_snapkv=True, snapkv_compression_ratio=0.2, use_cross_layer=True, cross_layer_groups=[[4, 5]], use_runtime_opt=True),
    ]

    all_results = {
        "model_name": MODEL_NAME,
        "results": {},
        "environment": {
            "torch": torch.__version__,
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "cpu_count": os.cpu_count(),
        },
        "ppl_metric_note": "PPL is manually calculated via CrossEntropy loss over sequence (not hf evaluate library)."
    }

    for dt in datasets:
        corpus_text = load_corpus_text(dt)
        dt_results = {}
        for abl in ablations:
            print(f"Running Ablation: {abl.name} on Dataset: {dt}...")
            
            # Load fresh model for each ablation to avoid state pollution
            model, tokenizer = load_model_and_tokenizer()
            metrics = run_ablation(abl, model, tokenizer, prompt, corpus_text)
            
            dt_results[abl.name] = asdict(metrics)
            
            del model
            gc.collect()
            
        all_results["results"][dt] = {
            "fallback_used": corpus_text.startswith("This is a local fallback corpus"),
            "ablations": dt_results
        }

    RESULT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics written to {RESULT_PATH}")


if __name__ == "__main__":
    main()