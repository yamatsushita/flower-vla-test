#!/usr/bin/env python3
"""
Local training script for FLOWER VLA on CALVIN.
Run with: python train_local.py [overrides]

Examples:
  # Debug run (uses calvin_debug_dataset)
  python train_local.py

  # Full D->D training (after downloading task_D_D)
  python train_local.py root_data_dir=./dataset/task_D_D benchmark_name=calvin_d rollout_lh_skip_epochs=19

  # ABCD->D training (after downloading task_ABCD_D)
  python train_local.py root_data_dir=./dataset/task_ABCD_D benchmark_name=calvin_abcd rollout_lh_skip_epochs=19 use_extracted_rel_actions=true
"""
import os
import sys
from pathlib import Path

# Ensure the repo root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Patch sys.argv to use local config
import hydra
from omegaconf import DictConfig
import runpy

# Override config to use local version
for i, arg in enumerate(sys.argv):
    if arg.startswith("--config-name"):
        break
else:
    # Insert local config if not specified
    sys.argv.insert(1, "--config-name=config_calvin_local")

# Run the training script
runpy.run_path(str(Path(__file__).parent / "flower" / "training_calvin.py"), run_name="__main__")
