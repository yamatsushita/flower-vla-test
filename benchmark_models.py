#!/usr/bin/env python3
"""
benchmark_models.py — Measures inference speed and memory footprint for
FLOWER VLA (Florence-2-large) and RandomProjVLA side by side.

Usage:
    # Benchmark only on synthetic data (no dataset needed):
    conda run -n flower_cal python benchmark_models.py --synthetic --n-iters 50

    # Benchmark with real D_D validation data (needs checkpoint):
    conda run -n flower_cal python benchmark_models.py \
        --florence-ckpt path/to/florence.ckpt \
        --randproj-ckpt path/to/randproj.ckpt \
        --data-dir dataset/task_D_D \
        --n-iters 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "False")


# ── helpers ─────────────────────────────────────────────────────────────────

def make_fake_batch(batch_size: int = 1, device: str = "cuda") -> Dict:
    """Create a minimal synthetic batch matching the CALVIN data format."""
    return {
        "rgb_obs": {
            "rgb_static":  torch.randn(batch_size, 1, 3, 112, 112, device=device),
            "rgb_gripper": torch.randn(batch_size, 1, 3, 112, 112, device=device),
        },
        "lang_text": ["push the red block to the left"] * batch_size,
        "actions":   torch.randn(batch_size, 10, 7, device=device),
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


@torch.no_grad()
def measure_vram_mb(model: nn.Module, batch: Dict, device: str) -> float:
    """Peak VRAM during a single inference forward pass (MB)."""
    if device == "cpu" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _ = model.encode_observations(batch)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return round(peak, 1)


@torch.no_grad()
def measure_inference_ms(
    model: nn.Module,
    batch: Dict,
    device: str,
    n_warmup: int = 5,
    n_iters: int = 50,
) -> Dict[str, float]:
    """
    Measure end-to-end step() latency in milliseconds.
    Returns mean, std, min, max.
    """
    goal = {"lang_text": batch["lang_text"][0]}
    obs  = batch["rgb_obs"]

    # Build the obs dict expected by step()
    step_obs = {
        "rgb_obs": {
            "rgb_static":  obs["rgb_static"],
            "rgb_gripper": obs["rgb_gripper"],
        }
    }
    model.eval()
    model.reset()

    # Warm-up
    for _ in range(n_warmup):
        model.reset()
        _ = model.step(step_obs, goal)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        model.reset()
        t0 = time.perf_counter()
        _ = model.step(step_obs, goal)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": round(float(times.mean()), 2),
        "std_ms":  round(float(times.std()),  2),
        "min_ms":  round(float(times.min()),  2),
        "max_ms":  round(float(times.max()),  2),
        "p50_ms":  round(float(np.percentile(times, 50)), 2),
        "p95_ms":  round(float(np.percentile(times, 95)), 2),
    }


def model_size_mb(model: nn.Module) -> float:
    """Estimate model weight size in MB (float32 equivalent)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 ** 2), 1)


# ── load models ─────────────────────────────────────────────────────────────

def load_florence_model(ckpt_path: Optional[str], device: str):
    """Load FLOWER VLA (Florence-2) from checkpoint or create fresh."""
    from omegaconf import OmegaConf
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading Florence-2 from checkpoint: {ckpt_path}")
        from flower.models.flower import FLOWERVLA
        model = FLOWERVLA.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        print("No Florence-2 checkpoint given; loading pretrained weights from HF…")
        opt_cfg = OmegaConf.create({
            "_target_": "torch.optim.AdamW",
            "transformer_weight_decay": 0.05,
            "learning_rate": 2e-5,
            "betas": [0.9, 0.95],
        })
        lr_cfg = OmegaConf.create({"lr_scheduler": {
            "init_lr": 2e-5, "init_lr_scale": 0.1, "final_lr_scale": 0.5,
            "total_steps": 50000, "phase_ratio": "(0.05,0.1,0.85)", "lr": 2e-5,
        }})
        from flower.models.flower import FLOWERVLA
        model = FLOWERVLA(
            vlm_path="microsoft/Florence-2-large",
            use_second_view=True,
            sampling_type="uniform",
            dit_dim=1024, n_heads=16, n_layers=18,
            use_rope=True, query_seq_len=100,
            num_sampling_steps=4,
            optimizer=opt_cfg, lr_scheduler=lr_cfg,
            load_pretrained=False,
        )
    model = model.to(device).eval()
    return model


def load_randproj_model(ckpt_path: Optional[str], device: str):
    """Load RandomProjVLA from checkpoint or create fresh."""
    from omegaconf import OmegaConf
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading RandomProjVLA from checkpoint: {ckpt_path}")
        from flower.models.random_proj_vla import RandomProjVLA
        model = RandomProjVLA.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        print("No RandomProj checkpoint; creating fresh model…")
        opt_cfg = OmegaConf.create({
            "_target_": "torch.optim.AdamW",
            "transformer_weight_decay": 0.05,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.95],
        })
        lr_cfg = OmegaConf.create({"lr_scheduler": {
            "init_lr": 1e-4, "init_lr_scale": 0.1, "final_lr_scale": 0.1,
            "total_steps": 35000, "phase_ratio": "(0.05,0.15,0.80)", "lr": 1e-4,
        }})
        from flower.models.random_proj_vla import RandomProjVLA
        model = RandomProjVLA(
            use_second_view=True,
            sampling_type="uniform",
            dit_dim=1024, n_heads=16, n_layers=18,
            use_rope=True, query_seq_len=100,
            num_sampling_steps=4,
            optimizer=opt_cfg, lr_scheduler=lr_cfg,
            load_pretrained=False,
        )
    model = model.to(device).eval()
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def benchmark_one(
    name: str,
    model: nn.Module,
    device: str,
    n_iters: int,
    batch_size: int,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    batch = make_fake_batch(batch_size=batch_size, device=device)

    params     = count_parameters(model)
    size_mb    = model_size_mb(model)
    vram_mb    = measure_vram_mb(model, batch, device)
    latency    = measure_inference_ms(model, batch, device, n_iters=n_iters)

    result = {
        "name":           name,
        "device":         device,
        "batch_size":     batch_size,
        "params_total":   params["total"],
        "params_trainable": params["trainable"],
        "params_frozen":  params["frozen"],
        "weight_size_mb": size_mb,
        "peak_vram_encode_mb": vram_mb,
        **{f"latency_{k}": v for k, v in latency.items()},
    }

    print(f"  Parameters (total/trainable/frozen): "
          f"{params['total']/1e6:.1f}M / "
          f"{params['trainable']/1e6:.1f}M / "
          f"{params['frozen']/1e6:.1f}M")
    print(f"  Model weight size: {size_mb:.1f} MB")
    print(f"  Peak VRAM (encode): {vram_mb:.1f} MB")
    print(f"  Inference latency (step, n={n_iters}):")
    print(f"    mean={latency['mean_ms']} ms  std={latency['std_ms']} ms  "
          f"p50={latency['p50_ms']} ms  p95={latency['p95_ms']} ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark VLA models")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-iters",   type=int, default=50)
    parser.add_argument("--n-warmup",  type=int, default=5)
    parser.add_argument("--florence-ckpt", default=None)
    parser.add_argument("--randproj-ckpt", default=None)
    parser.add_argument("--output",    default="logs/benchmark_results.json")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory/(1024**3):.1f} GB")

    results = []

    # ── Florence-2 ──
    try:
        model_f = load_florence_model(args.florence_ckpt, device)
        r = benchmark_one("FLOWER VLA (Florence-2-large)", model_f, device,
                          n_iters=args.n_iters, batch_size=args.batch_size)
        results.append(r)
        del model_f
        if device != "cpu":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"WARNING: Could not benchmark Florence-2: {e}")

    # ── RandomProjVLA ──
    try:
        model_r = load_randproj_model(args.randproj_ckpt, device)
        r = benchmark_one("RandomProjVLA", model_r, device,
                          n_iters=args.n_iters, batch_size=args.batch_size)
        results.append(r)
        del model_r
        if device != "cpu":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"WARNING: Could not benchmark RandomProjVLA: {e}")

    # ── Save results ──
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out}")

    # ── Quick comparison table ──
    if len(results) == 2:
        r_f, r_r = results[0], results[1]
        speed_ratio = r_r["latency_mean_ms"] / r_f["latency_mean_ms"]
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<35} {'Florence-2':>15} {'RandProj':>15}")
        print("-" * 65)
        print(f"{'Params (M, trainable)':<35} {r_f['params_trainable']/1e6:>14.0f}M "
              f"{r_r['params_trainable']/1e6:>14.0f}M")
        print(f"{'Params (M, frozen)':<35} {r_f['params_frozen']/1e6:>14.0f}M "
              f"{r_r['params_frozen']/1e6:>14.0f}M")
        print(f"{'Weight size (MB)':<35} {r_f['weight_size_mb']:>15.0f} "
              f"{r_r['weight_size_mb']:>15.0f}")
        print(f"{'Peak VRAM (encode, MB)':<35} {r_f['peak_vram_encode_mb']:>15.0f} "
              f"{r_r['peak_vram_encode_mb']:>15.0f}")
        print(f"{'Inference mean (ms)':<35} {r_f['latency_mean_ms']:>15.1f} "
              f"{r_r['latency_mean_ms']:>15.1f}")
        print(f"{'Inference p95 (ms)':<35} {r_f['latency_p95_ms']:>15.1f} "
              f"{r_r['latency_p95_ms']:>15.1f}")
        print(f"{'Speed-up (RandProj vs F2)':<35} {'–':>15} {1/speed_ratio:>14.2f}x")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
