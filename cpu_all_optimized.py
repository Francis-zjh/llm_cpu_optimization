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

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.utils import extract_keys_and_values

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch\\.ao")
warnings.filterwarnings("ignore", message="Profiler clears events")


MODEL_NAME = "EleutherAI/pythia-70m"
RESULT_PATH = Path("cpu_all_opt_results.json")
GENERATION_TOKENS = 1024
REPEATS = 3
RUN_SEQLEN_SWEEP = True
SEQ_LENGTHS = [128, 256, 512, 1024]

@dataclass
class AblationConfig:
    name: str
    use_quantization: bool = False
    use_fp16: bool = False
    use_snapkv: bool = False
    snapkv_compression_ratio: float = 0.2
    snapkv_window_size: int = 16
    use_cross_layer: bool = False
    cross_layer_groups: list[list[int]] = field(default_factory=list)
    use_ipex: bool = False
    use_compile: bool = False
    n_threads: int = 8


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


def set_thread_count(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)


def load_model_and_tokenizer() -> tuple[torch.nn.Module, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True, dtype=torch.float32)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=False, dtype=torch.float32)
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model, tokenizer


def apply_dynamic_quantization(model: torch.nn.Module) -> torch.nn.Module:
    try:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return model
    except Exception as exc:
        print(f"  [WARN] Quantization failed: {exc}")
        return model


def apply_fp16(model: torch.nn.Module) -> torch.nn.Module:
    try:
        model = model.half()
        return model
    except Exception as exc:
        print(f"  [WARN] FP16 conversion failed: {exc}")
        return model


def _get_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
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
            except Exception as exc:
                warnings.warn(f"snapkv hook skipped on layer {_layer_idx}: {exc}")
            return output

        handles.append(attn.register_forward_hook(_hook, with_kwargs=True))
    return handles, stats, press


def share_kv_cache_across_layer_groups(cache: Any, groups: list[list[int]]) -> int:
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


def maybe_optimize_runtime(model: torch.nn.Module, ablation: AblationConfig) -> tuple[torch.nn.Module, list[str]]:
    notes: list[str] = []

    if ablation.use_ipex:
        try:
            import intel_extension_for_pytorch as ipex
            model = ipex.optimize(model, dtype=torch.float32)
            notes.append("ipex_applied")
        except Exception as exc:
            notes.append(f"ipex_skipped:{type(exc).__name__}")

    if ablation.use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            notes.append("torch_compile_applied")
        except Exception as exc:
            notes.append(f"torch_compile_skipped:{type(exc).__name__}")

    return model, notes


DATA_DIR = Path(__file__).parent / "data"


def load_corpus_text(dataset_type: str) -> str:
    local_path = DATA_DIR / f"{dataset_type}_corpus.txt"
    if local_path.exists():
        text = local_path.read_text(encoding="utf-8").strip()
        if text:
            print(f"  [OK] Loaded local {dataset_type}_corpus.txt ({len(text)} chars)")
            return text

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

    return (
        "[FALLBACK] Local fallback corpus — dataset download failed. "
        "The benchmark pipeline is preserved; fallback is flagged in results."
    )


def compute_perplexity(
    model: torch.nn.Module, tokenizer: Any, text: str, max_length: int = 512
) -> float | None:
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
            print(f"  [DEBUG] PPL skipped: loss={loss}")
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
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"]
    step_times: list[float] = []

    with torch.inference_mode(), torch.amp.autocast("cpu", enabled=False):
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
    try:
        from torch.profiler import ProfilerActivity, profile

        inputs = tokenizer(prompt, return_tensors="pt")
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.inference_mode(), torch.amp.autocast("cpu", enabled=False):
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


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _stdev(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 0.0
    m = sum(clean) / len(clean)
    var = sum((v - m) ** 2 for v in clean) / len(clean)
    return math.sqrt(var)


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "ppl", "ttft_s", "tpot_s", "throughput_tok_s",
        "ram_rss_mb", "flops", "model_size_mb", "quantized_size_mb", "generated_tokens",
    ]
    aggregated: dict[str, Any] = {}
    for k in numeric_keys:
        vals = [r[k] for r in runs]
        aggregated[k] = _mean(vals)
        aggregated[f"{k}_std"] = _stdev(vals)

    all_notes: list[str] = []
    for r in runs:
        all_notes.extend(r.get("notes", []))
    aggregated["notes"] = all_notes

    aggregated["runs"] = runs
    aggregated["num_repeats"] = len(runs)
    return aggregated


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

    size_fp32 = measure_model_size_mb(model)
    notes.append(f"fp32_model_size_mb={size_fp32:.1f}")

    if ablation.use_quantization:
        model = apply_dynamic_quantization(model)
        size_int8 = measure_model_size_mb(model)
        notes.append(f"int8_model_size_mb={size_int8:.1f}")
        notes.append("quantization_applied")
    else:
        size_int8 = None

    if ablation.use_fp16 and not ablation.use_quantization:
        model = apply_fp16(model)
        size_fp16 = measure_model_size_mb(model)
        notes.append(f"fp16_model_size_mb={size_fp16:.1f}")
        notes.append("fp16_applied")
    elif ablation.use_fp16 and ablation.use_quantization:
        notes.append("fp16_skipped:quantization_overrides")

    if ablation.use_snapkv:
        handles, press_stats, _ = attach_snapkv_hooks(
            model,
            compression_ratio=ablation.snapkv_compression_ratio,
            window_size=ablation.snapkv_window_size,
        )
        notes.append(f"kvpress_hooks={len(handles)}")

    model, runtime_notes = maybe_optimize_runtime(model, ablation)
    notes.extend(runtime_notes)

    if ablation.use_compile or ablation.use_ipex:
        print("    Warm-up to avoid cold start penalty …")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = measure_generation(model, tokenizer, prompt, max_new_tokens=4, ablation=ablation)
        except Exception as exc:
            print(f"    [WARN] Warmup generation failed: {exc}")

    print("    PPL …")
    ppl = compute_perplexity(model, tokenizer, corpus_text, max_length=512)

    print(f"    Generation ({GENERATION_TOKENS} tokens) …")
    gen, shared_layers = measure_generation(model, tokenizer, prompt, ablation=ablation)

    print("    FLOPs …")
    flops = measure_flops(model, tokenizer, prompt, max_new_tokens=4)

    ram = rss_mb()

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

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


def run_seqlen_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    lengths: list[int] = SEQ_LENGTHS,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    for L in lengths:
        gen_base, _ = measure_generation(model, tokenizer, prompt, max_new_tokens=L)

        handles, _, _ = attach_snapkv_hooks(model, compression_ratio=0.2, window_size=16)
        gen_skv, _ = measure_generation(model, tokenizer, prompt, max_new_tokens=L)
        for h in handles:
            h.remove()

        results[str(L)] = {
            "baseline": {
                "ttft_s": gen_base["ttft_s"],
                "tpot_s": gen_base["tpot_s"],
                "throughput_tok_s": gen_base["throughput_tok_s"],
            },
            "snapkv": {
                "ttft_s": gen_skv["ttft_s"],
                "tpot_s": gen_skv["tpot_s"],
                "throughput_tok_s": gen_skv["throughput_tok_s"],
            },
        }
    return results


def main() -> None:
    prompt = "Pythia-70M is a small language model that can still be profiled on CPU."

    datasets = ["wikitext", "pg19"]

    ablations = [
        AblationConfig(name="Baseline", n_threads=1),
        AblationConfig(name="Baseline_opt", n_threads=8),
        AblationConfig(name="Quant_INT8_only", use_quantization=True, n_threads=4),
        AblationConfig(name="SnapKV_only", use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="Crosslayer_only", use_cross_layer=True, cross_layer_groups=[[4, 5]], n_threads=4),
        AblationConfig(name="Quant_FP16_only", use_fp16=True, n_threads=8),
        AblationConfig(name="Compile_only", use_compile=True, n_threads=4),
        AblationConfig(name="Compile_SnapKV", use_compile=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="Compile_Quant_INT8", use_compile=True, use_quantization=True, n_threads=4),
        AblationConfig(name="Compile_Quant_FP16", use_compile=True, use_fp16=True, n_threads=4),
        AblationConfig(name="SnapKV_CrossLayer", use_snapkv=True, use_cross_layer=True, cross_layer_groups=[[4, 5]], snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="Quant_FP16_SnapKV", use_fp16=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="IPEX_only", use_ipex=True, n_threads=4),
        AblationConfig(name="IPEX_Compile", use_ipex=True, use_compile=True, n_threads=4),
        AblationConfig(name="IPEX_SnapKV", use_ipex=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="IPEX_CrossLayer", use_ipex=True, use_cross_layer=True, cross_layer_groups=[[4, 5]], n_threads=4),
        AblationConfig(name="IPEX_Quant_INT8", use_ipex=True, use_quantization=True, n_threads=4),
        AblationConfig(name="Compile_SnapKV_CrossLayer", use_compile=True, use_snapkv=True, use_cross_layer=True, cross_layer_groups=[[4, 5]], snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="Compile_Quant_FP16_SnapKV", use_compile=True, use_fp16=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="IPEX_Compile_SnapKV", use_ipex=True, use_compile=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="IPEX_Quant_INT8_SnapKV", use_ipex=True, use_quantization=True, use_snapkv=True, snapkv_compression_ratio=0.2, n_threads=4),
        AblationConfig(name="All_In_One", use_ipex=True, use_compile=True, use_quantization=True, use_snapkv=True, use_cross_layer=True, snapkv_compression_ratio=0.2, cross_layer_groups=[[4, 5]], n_threads=4),
    ]
    ablation_names = {a.name for a in ablations}

    all_results: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "generation_tokens": GENERATION_TOKENS,
        "repeats": REPEATS,
        "seqlen_sweep_lengths": SEQ_LENGTHS if RUN_SEQLEN_SWEEP else [],
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
        "quant_int8_note": (
            f"INT8 dynamic quantisation via torch.ao.quantization.quantize_dynamic "
            f"on all nn.Linear layers. REPEATS={REPEATS} per ablation."
        ),
        "snapkv_note": "SnapKV from kvpress library, compression_ratio=0.2, window_size=16.",
        "crosslayer_note": "Cross-layer KV sharing on layers [4,5] (last 2 of 6, mild setting).",
        "runtime_note": "IPEX and torch.compile are typically unavailable on Windows; auto-detected and skipped.",
        "flops_note": (
            "FLOPs via torch.profiler (CPU). Values for quantized models are unreliable "
            "because the profiler cannot correctly count INT8 operator FLOPs."
        ),
        "quant_fp16_note": "FP16 half precision via model.half(). Model size halved but CPU lacks native FP16 compute, so speed may not improve.",
    }

    if RESULT_PATH.exists():
        try:
            saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            if saved.get("model_name") == MODEL_NAME and "results" in saved:
                all_results = saved
                n_done = sum(
                    1 for ds in saved["results"].values()
                    for k in ds if k in ablation_names
                )
                print(f"[Resume] Loaded existing results ({n_done} configs done)")
        except Exception as exc:
            print(f"[Resume] Ignored corrupt results file: {exc}")

    def save_checkpoint() -> None:
        RESULT_PATH.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for dt in datasets:
        ds_done: set[str] = set()
        if "results" in all_results and dt in all_results["results"]:
            ds_done = {k for k in all_results["results"][dt] if k in ablation_names}

        if ds_done == ablation_names:
            print(f"\n[Skip] {dt} — all {len(ablations)} configs completed")
            continue

        if ds_done:
            missing = ablation_names - ds_done
            print(f"\n{'=' * 60}\n[Resume] {dt}: {len(ds_done)}/{len(ablations)} done, missing: {sorted(missing)}")
        else:
            print(f"\n{'=' * 60}\nRunning dataset: {dt}")

        corpus_text = load_corpus_text(dt)

        dt_results: dict[str, Any] = all_results.setdefault("results", {}).setdefault(dt, {})
        dt_results.setdefault("experiments", [])

        if "fallback_used" not in dt_results:
            dt_results["fallback_used"] = corpus_text.startswith("[FALLBACK]")
        if "core_ablation" not in dt_results["experiments"]:
            dt_results["experiments"].append("core_ablation")

        core_start = time.perf_counter()

        for abl in tqdm.tqdm(ablations, desc=f"  [{dt}]", unit="exp", leave=False):
            if abl.name in dt_results:
                print(f"  [Skip] {abl.name} already done")
                continue

            print(f"\n  === {abl.name} (REPEATS={REPEATS}, threads={abl.n_threads}) ===")
            run_list: list[dict[str, Any]] = []

            for rep in range(REPEATS):
                print(f"    ── repeat {rep + 1}/{REPEATS} ──")
                set_thread_count(abl.n_threads)
                model, tokenizer = load_model_and_tokenizer()
                torch.manual_seed(7 + rep)
                metrics = run_ablation(abl, model, tokenizer, prompt, corpus_text)
                run_list.append(asdict(metrics))
                del model, tokenizer
                gc.collect()

            dt_results[abl.name] = aggregate_runs(run_list)
            save_checkpoint()
            print(f"    ✓ checkpoint saved")

        core_elapsed = time.perf_counter() - core_start
        print(f"\n  [Core ablation for {dt} done in {core_elapsed:.0f}s]")

        if RUN_SEQLEN_SWEEP:
            if "seqlen_sweep" in dt_results:
                print(f"  [Skip] SeqLen sweep already done for {dt}")
            else:
                dt_results["experiments"].append("seqlen_sweep")
                print(f"\n  === Sequence-length sweep {SEQ_LENGTHS} ===")
                set_thread_count(8)
                model, tokenizer = load_model_and_tokenizer()
                sweep = run_seqlen_sweep(model, tokenizer, prompt, lengths=SEQ_LENGTHS)
                dt_results["seqlen_sweep"] = sweep
                del model, tokenizer
                gc.collect()
                print(f"  [SeqLen sweep done]")
                save_checkpoint()

        print(f"{'=' * 60}\n")

    print(f"\n{'=' * 60}")
    print(f"All experiments complete! Results → {RESULT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
