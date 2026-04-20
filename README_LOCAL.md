# Flower VLA — Local Training Setup

Reproducing training from [intuitive-robots/flower_vla_calvin](https://github.com/intuitive-robots/flower_vla_calvin) on a single RTX 4090 (Windows 11 + WSL2).

## Environment

| | |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4090 (24 GB) |
| conda env | `flower_cal` (Python 3.13.2, torch 2.6.0+cu124) |

## Setup

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/yamatsushita/flower-vla-test.git
cd flower-vla-test
```

### 2. Install dependencies
```bash
# Install calvin_env (pybullet already in conda; skip deps)
pip install -e calvin_env --no-deps
pip install -e "calvin_env/calvin_env/tacto" --no-deps

# Install this package
pip install -e . --no-deps

# Install remaining deps
pip install pytorch-lightning==2.0.8 wandb sentence-transformers einops_exts \
            moviepy tqdm termcolor torchsde torchdiffeq opencv-python \
            pyrender urdfpy hydra-colorlog
```

### 3. pyhash stub (Python 3.12+ compatibility)
pyhash 0.9.3 uses `use_2to3` removed in Python 3.12. Create a pure-Python stub:
```bash
conda run -n flower_cal python -c "
import site, pathlib
stub = '''
import struct
def fnv1_32(data, seed=0):
    h = 2166136261 ^ seed
    for b in (data.encode() if isinstance(data,str) else data):
        h = ((h * 16777619) ^ (b if isinstance(b,int) else ord(b))) & 0xFFFFFFFF
    return h
def fnv1a_32(data, seed=0):
    h = 2166136261 ^ seed
    for b in (data.encode() if isinstance(data,str) else data):
        h = (((h ^ (b if isinstance(b,int) else ord(b))) * 16777619)) & 0xFFFFFFFF
    return h
'''
sp = pathlib.Path(site.getsitepackages()[0]) / 'pyhash.py'
sp.write_text(stub)
print('Written to', sp)
"
```

### 4. Download pretrained FLOWER weights
```bash
# Via WSL (Windows HuggingFace download blocked by corporate proxy):
wsl -- wget https://huggingface.co/mbreuss/flower_vla_pret/resolve/main/360000_model_weights.pt \
      -O checkpoints/flower_pret/360000_model_weights.pt
wsl -- wget https://huggingface.co/mbreuss/flower_vla_pret/resolve/main/config.yaml \
      -O checkpoints/flower_pret/config.yaml
```

### 5. Download CALVIN dataset
```bash
# task_D_D (~165 GB) using BITS transfer (Windows):
Start-BitsTransfer -Source "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip" \
                   -Destination "dataset/task_D_D.zip" -Asynchronous

# Or unzip the debug dataset for quick iteration:
# cp -r <existing_debug_dataset> dataset/calvin_debug_dataset
```

## Training

### Quick smoke test (2 batches, 1 epoch)
```bash
conda run -n flower_cal python flower/training_calvin.py \
  --config-name=config_calvin_local \
  trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=0
```

### Full training on debug dataset
```bash
conda run -n flower_cal python flower/training_calvin.py --config-name=config_calvin_local
```

### Full training on task_D_D (after download + unzip)
Edit `conf/config_calvin_local.yaml`:
- Set `root_data_dir` to the unzipped `task_D_D` folder
- Set `use_extracted_rel_actions: true` (after running `preprocess/extract_by_key.py`)

```bash
conda run -n flower_cal python flower/training_calvin.py --config-name=config_calvin_local
```

## Windows Compatibility Patches Applied

| File | Issue | Fix |
|------|-------|-----|
| `flower/wrappers/hulc_wrapper.py` | EGL detection runs `./EGL_options.o` (Linux binary) | Catch `FileNotFoundError`/`OSError` in addition to `EglDeviceNotFoundError` |
| `flower/rollout/rollout_long_horizon.py` | `on_validation_start` initializes CALVIN env even when rollouts are skipped | Added `skip_epochs` check at start of `on_validation_start` |
| `flower/training_calvin.py` | Hardcoded `ddp_find_unused_parameters_true` strategy + `use_libuv` issue | Single-GPU detection: use `"auto"` strategy when `devices=1` |

## Config Files

- `conf/config_calvin_local.yaml` — single-GPU local training config
- `conf/model/flower_local.yaml` — model config with local pretrained weights path
- `train_local.py` — convenience launcher with `PYTHONPATH` set

## Known Issues

- CALVIN rollout evaluation requires Linux (EGL/pybullet rendering). Set `rollout_lh_skip_epochs: 999` to skip rollouts during training-only runs on Windows.
- HuggingFace downloads may be blocked by corporate proxy on Windows — use WSL wget instead.
- pyhash 0.9.3 is incompatible with Python 3.12+; use the pure-Python stub above.
