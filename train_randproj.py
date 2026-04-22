#!/usr/bin/env python3
"""
Training launcher for RandomProjVLA on CALVIN D→D.

Usage:
    python train_randproj.py
    python train_randproj.py max_epochs=50

The script is identical to train_local.py / flower/training_calvin.py but:
  - uses config_calvin_randproj (random projection model, D_D dataset)
  - imports RandomProjVLA so checkpoint-resume works correctly
"""
import os
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch

# PyTorch 2.6 changed weights_only default to True, but PL checkpoints contain
# omegaconf objects.  Patch torch.load so resume works with trusted local ckpts.
_orig_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import wandb

sys.path.insert(0, str(Path(__file__).parent))

import flower.models.flower as _flower_m
import flower.models.random_proj_vla as _rp_m  # noqa: F401 — ensures RandomProjVLA is importable
from flower.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()


@rank_zero_only
def _log(*args, **kwargs):
    logger.info(*args, **kwargs)


def _setup_callbacks(callbacks_cfg: DictConfig):
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]


def _setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
        cfg.logger.id = cfg.logger.name.replace("/", "_")
    return hydra.utils.instantiate(cfg.logger)


def _get_model_class(target: str):
    """Resolve the model class from a dotted string, checking both modules."""
    class_name = target.split(".")[-1]
    # Check RandomProjVLA first, then fall back to flower.py
    if hasattr(_rp_m, class_name):
        return getattr(_rp_m, class_name)
    return getattr(_flower_m, class_name)


@hydra.main(config_path="conf", config_name="config_calvin_randproj")
def train(cfg: DictConfig) -> None:
    try:
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _clear_cuda_cache()

        _log(f"Initialising RandomProjVLA training — seed {cfg.seed}")

        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # Support explicit ckpt_path via env var RANDPROJ_CKPT, hydra config, or auto-detect
        explicit_ckpt = (
            os.environ.get("RANDPROJ_CKPT")
            or cfg.get("ckpt_path", None)
        )
        last_ckpt = Path(explicit_ckpt) if explicit_ckpt else get_last_checkpoint(Path.cwd())
        model = hydra.utils.instantiate(cfg.model)
        if last_ckpt is not None:
            _log(f"Will resume training state from checkpoint: {last_ckpt}")

        if "pretrain_chk" in cfg:
            initialize_pretrained_weights(model, cfg)

        train_logger = _setup_logger(cfg, model)
        callbacks = _setup_callbacks(cfg.callbacks) + [
            LearningRateMonitor(logging_interval="step")
        ]

        work_dir = Path.cwd() / f"seed_{cfg.seed}"
        work_dir.mkdir(exist_ok=True)
        os.chdir(work_dir)

        num_devices = (
            cfg.trainer.devices
            if isinstance(cfg.trainer.devices, int)
            else len(cfg.trainer.devices)
        )
        strategy = "ddp_find_unused_parameters_true" if num_devices > 1 else "auto"

        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "strategy": strategy,
            "accelerator": "gpu",
            "devices": cfg.trainer.devices,
            "use_distributed_sampler": num_devices > 1,
            "default_root_dir": work_dir,
            "sync_batchnorm": num_devices > 1,
        }

        _log(f"Config:\n{cfg}")
        _log(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
        _log(print_system_env_info())

        _clear_cuda_cache()
        trainer = Trainer(**trainer_args)

        try:
            trainer.fit(model, datamodule=datamodule,
                        ckpt_path=str(last_ckpt) if last_ckpt else None)
        except Exception as e:
            import traceback
            _log(f"Training error: {type(e).__name__}: {e}")
            _log(traceback.format_exc())
            raise

    except Exception as e:
        import traceback
        logger.error(f"Training failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        _clear_cuda_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, str(Path(__file__).parent))
    train()
