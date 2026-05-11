from __future__ import annotations

import json
import math
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import gc
import shutil
import ssl
import sys
import time
import urllib3
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.utils import extract_keys_and_values

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "EleutherAI/pythia-70m"
RESULT_PATH = Path("cpu_all_opt_results.json")
GENERATION_TOKENS = 64  # number of tokens to generate for TTFT / TPOT / Throughput


# ---------------------------------------------------------------------------
# Config & result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    name: str
    use_quantization: bool = False
    use_snapkv: bool = False
    snapkv_compression_ratio: float = 0.2
    snapkv_window_size: int = 16
    use_cross_layer: bool = False
    cross_layer_groups: list[list[int]] = field(default_factory=list)


@dataclass
class RunMetrics:
    ppl: float | None
    ttft_s: float | None
    tpot_s: float | None
    throughput_tok_s: float | None
    ram_rss_mb: float | None
    flops: float | None
    model_size_mb: float | None
    quantized_size_mb: float | None
    generated_tokens: int
    notes: list[str]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_tokenizer() -> tuple[torch.nn.Module, Any]:
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


# ---------------------------------------------------------------------------
# Optimisation 1 — INT8 Dynamic Quantization  (primary positive result)
# ---------------------------------------------------------------------------
def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Post-training INT8 dynamic quantisation on all Linear layers.
    Training-free, works on any CPU, reduces model footprint ~4× on linear layers.
    """
    try:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return model
    except Exception as exc:
        print(f"  [WARN] Quantization failed: {exc}")
        return model


# ---------------------------------------------------------------------------
# Optimisation 2 — SnapKV (kvpress)
# ---------------------------------------------------------------------------
def _get_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the attention submodule of every GPT-NeoX layer."""
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
) -> tuple[list[Any], dict[str, float], SnapKVPress]:
    """Register forward hooks that compress KV caches after prefill via SnapKV."""
    press = SnapKVPress(
        compression_ratio=compression_ratio, window_size=window_size, kernel_size=5
    )
    stats: dict[str, float] = {"calls": 0.0, "tokens_before": 0.0, "tokens_after": 0.0}
    if hasattr(press, "post_init_from_model"):
        try:
            press.post_init_from_model(model)
        except Exception:
            pass

    handles: list[Any] = []
    for attn in _get_attention_modules(model):
        layer_idx = attn.layer_idx

        def _hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            _layer_idx: int = layer_idx,
        ):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            cache = kwargs.get("layer_past")
            if cache is None and len(args) > 3:
                cache = args[3]
            if hidden_states is None or cache is None:
                return output

            # Only compress once after prefill, not during token-by-token generation
            seq_len = hidden_states.shape[1]

            # Skip prefill if sequence is shorter than window size
            if seq_len <= press.window_size:
                return output

            try:
                keys, values = extract_keys_and_values(cache, _layer_idx)
                before = float(keys.shape[-2])
                new_keys, new_values = press.compress(
                    module, hidden_states, keys, values, output[1], kwargs
                )
                # Replace cache in-place
                cache.layers[_layer_idx].keys = new_keys.contiguous()
                cache.layers[_layer_idx].values = new_values.contiguous()
                stats["calls"] += 1.0
                stats["tokens_before"] += before
                stats["tokens_after"] += float(new_keys.shape[-2])
            except Exception as exc:
                warnings.warn(f"snapkv hook skipped on layer {_layer_idx}: {exc}")
            return output

        handles.append(attn.register_forward_hook(_hook, with_kwargs=True))
    return handles, stats, press


# ---------------------------------------------------------------------------
# Optimisation 3 — Cross-layer KV sharing  (exploratory)
# ---------------------------------------------------------------------------
def share_kv_cache_across_layer_groups(cache: Any, groups: list[list[int]]) -> int:
    """Share (alias) KV cache entries across layers within each group.

    After this call, layers in the same group point to the *same* KV tensors,
    saving memory at the cost of reduced expressiveness.
    Returns the number of layers that were overwritten.
    """
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


# ---------------------------------------------------------------------------
# Runtime optimisation attempt  (IPEX / torch.compile – usually skipped on Win)
# ---------------------------------------------------------------------------
def maybe_optimize_runtime(model: torch.nn.Module) -> tuple[torch.nn.Module, list[str]]:
    notes: list[str] = []
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore[import-untyped]

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus_text(dataset_type: str) -> str:
    try:
        if dataset_type == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5%]")
        else:
            ds = load_dataset("emozilla/pg19-test", split="test[:1]")
        texts = [row["text"] for row in ds if row.get("text")]
        corpus = "\n\n".join(texts).strip()
        if corpus:
            return corpus[:4000]
    except Exception as exc:
        print(f"  [WARN] Could not load {dataset_type}: {exc}")

    # Fallback
    return (
        "[FALLBACK] Local fallback corpus — dataset download failed. "
        "The benchmark pipeline is preserved; fallback is flagged in results."
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_perplexity(
    model: torch.nn.Module, tokenizer: Any, text: str, max_length: int = 128
) -> float | None:
    """Perplexity via CrossEntropy loss on the *input* sequence (not eval library)."""
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"]
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=True)
        loss = float(outputs.loss)
        if math.isnan(loss) or math.isinf(loss):
            return None
        return math.exp(loss)
    except Exception as exc:
        print(f"  [WARN] PPL failed: {exc}")
        return None


def measure_model_size_mb(model: torch.nn.Module) -> float:
    try:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buf_bytes) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def measure_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = GENERATION_TOKENS,
    ablation: AblationConfig | None = None,
) -> tuple[dict[str, Any], int]:
    """Return generation metrics and number of layers cross-shared."""
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"]
    step_times: list[float] = []

    with torch.inference_mode():
        # ---- prefill + first token ----
        start = time.perf_counter()
        outputs = model(input_ids=generated, use_cache=True)

        shared_layers = 0
        if ablation and ablation.use_cross_layer:
            shared_layers = share_kv_cache_across_layer_groups(
                outputs.past_key_values, ablation.cross_layer_groups
            )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_key_values = outputs.past_key_values
        ttft = time.perf_counter() - start

        # ---- remaining tokens ----
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
        "total_s": total,
        "decoded_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }, shared_layers


def measure_flops(
    model: torch.nn.Module, tokenizer: Any, prompt: str, max_new_tokens: int = 4
) -> float | None:
    """Aggregate CPU FLOPs from torch.profiler (best-effort on CPU)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        inputs = tokenizer(prompt, return_tensors="pt")
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=True)
                tok = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                pkv = outputs.past_key_values
                for _ in range(max_new_tokens - 1):
                    out = model(input_ids=tok, use_cache=True, past_key_values=pkv)
                    tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    pkv = out.past_key_values

        total = 0.0
        for evt in prof.key_averages():
            flops = getattr(evt, "flops", None)
            if flops is not None:
                total += float(flops)
        return total if total > 0 else None
    except Exception as exc:
        print(f"  [WARN] FLOPs measurement skipped: {exc}")
        return None


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation(
    ablation: AblationConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    corpus_text: str,
) -> RunMetrics:
    notes: list[str] = [f"ablation:{ablation.name}"]
    handles: list[Any] = []
    press_stats: dict[str, float] | None = None

    # 1. model size before any optimisation
    size_fp32 = measure_model_size_mb(model)
    notes.append(f"fp32_model_size_mb={size_fp32:.1f}")

    # 2. INT8 dynamic quantisation
    if ablation.use_quantization:
        model = apply_dynamic_quantization(model)
        size_int8 = measure_model_size_mb(model)
        notes.append(f"int8_model_size_mb={size_int8:.1f}")
        notes.append("quantization_applied")
    else:
        size_int8 = None

    # 3. SnapKV hooks
    if ablation.use_snapkv:
        handles, press_stats, _ = attach_snapkv_hooks(
            model,
            compression_ratio=ablation.snapkv_compression_ratio,
            window_size=ablation.snapkv_window_size,
        )
        notes.append(f"kvpress_hooks={len(handles)}")

    # 4. Runtime optimisation (attempt, usually skipped on Windows)
    model, runtime_notes = maybe_optimize_runtime(model)
    notes.extend(runtime_notes)

    # 5. Metrics
    print("    PPL …")
    ppl = compute_perplexity(model, tokenizer, corpus_text, max_length=128)

    print(f"    Generation ({GENERATION_TOKENS} tokens) …")
    gen, shared_layers = measure_generation(model, tokenizer, prompt, ablation=ablation)

    print("    FLOPs …")
    flops = measure_flops(model, tokenizer, prompt, max_new_tokens=4)

    ram = rss_mb()

    # 6. Cleanup hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # 7. Additional notes
    if ablation.use_cross_layer:
        notes.append(f"cross_layer_shared_layers={shared_layers}")

    if ablation.use_snapkv and press_stats and press_stats["calls"] > 0:
        avg_before = press_stats["tokens_before"] / press_stats["calls"]
        avg_after = press_stats["tokens_after"] / press_stats["calls"]
        ratio = 1.0 - (avg_after / avg_before if avg_before > 0 else 1.0)
        notes.append(f"snapkv_effective_ratio={ratio:.4f}")
    elif ablation.use_snapkv:
        notes.append("snapkv_effective_ratio=0.0000")

    return RunMetrics(
        ppl=ppl,
        ttft_s=float(gen["ttft_s"]) if gen["ttft_s"] is not None else None,
        tpot_s=float(gen["tpot_s"]) if gen["tpot_s"] is not None else None,
        throughput_tok_s=float(gen["throughput_tok_s"]) if gen["throughput_tok_s"] is not None else None,
        ram_rss_mb=ram,
        flops=flops,
        model_size_mb=size_fp32,
        quantized_size_mb=size_int8,
        generated_tokens=int(gen["generated_tokens"]),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(7)
    prompt = "Pythia-70M is a small language model that can still be profiled on CPU."

    datasets = ["wikitext", "pg19"]

    # Ablation matrix (6 experiments):
    #   baseline          – no optimisations (reference point)
    #   quant_only        – INT8 dynamic quantisation only
    #   snapkv_only       – SnapKV KV cache compression only
    #   crosslayer_only   – cross-layer KV sharing only
    #   quant+snapkv      – combined: compute (quant) + memory (SnapKV)
    #   all_optimized     – quant + snapkv + cross-layer
    ablations = [
        AblationConfig(name="baseline"),
        AblationConfig(name="quant_only", use_quantization=True),
        AblationConfig(
            name="snapkv_only",
            use_snapkv=True,
            snapkv_compression_ratio=0.2,
        ),
        AblationConfig(
            name="crosslayer_only",
            use_cross_layer=True,
            cross_layer_groups=[[4, 5]],
        ),
        AblationConfig(
            name="quant+snapkv",
            use_quantization=True,
            use_snapkv=True,
            snapkv_compression_ratio=0.2,
        ),
        AblationConfig(
            name="all_optimized",
            use_quantization=True,
            use_snapkv=True,
            snapkv_compression_ratio=0.2,
            use_cross_layer=True,
            cross_layer_groups=[[4, 5]],
        ),
    ]

    all_results = {
        "model_name": MODEL_NAME,
        "generation_tokens": GENERATION_TOKENS,
        "results": {},
        "environment": {
            "torch": torch.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "cpu_count": os.cpu_count(),
        },
        "ppl_metric_note": (
            "PPL computed via CrossEntropy loss on input sequence (math.exp(loss)), "
            "NOT the evaluate library."
        ),
        "quantization_note": (
            "INT8 dynamic quantisation via torch.ao.quantization.quantize_dynamic "
            "on all nn.Linear layers. Post-training, training-free."
        ),
        "snapkv_note": (
            "SnapKV from kvpress library, compression_ratio=0.2, window_size=16."
        ),
        "crosslayer_note": (
            "Cross-layer KV sharing on layers [4,5] (last 2 of 6, mild setting)."
        ),
    }

    for dt in tqdm.tqdm(datasets, desc="Dataset", unit="ds"):
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {dt}")
        corpus_text = load_corpus_text(dt)
        dt_results: dict[str, Any] = {}

        for abl in tqdm.tqdm(ablations, desc=f"  [{dt}]", unit="exp", leave=False):
            print(f"  === {abl.name} ===")
            model, tokenizer = load_model_and_tokenizer()
            metrics = run_ablation(abl, model, tokenizer, prompt, corpus_text)
            dt_results[abl.name] = asdict(metrics)

            del model, tokenizer
            gc.collect()

        all_results["results"][dt] = {
            "fallback_used": corpus_text.startswith("[FALLBACK]"),
            "ablations": dt_results,
        }
        print(f"{'=' * 60}\n")

    RESULT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"All experiments complete! Results → {RESULT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
