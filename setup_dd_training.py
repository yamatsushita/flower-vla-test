"""
Setup and launch training on CALVIN task_D_D dataset.

Run this after task_D_D.zip finishes downloading:
    python setup_dd_training.py

Steps performed:
1. Unzips task_D_D.zip to dataset/task_D_D/
2. Runs preprocessing (extract_by_key.py) to extract relative actions
3. Generates the paraphrase language embeddings (if not present)
4. Launches training with config_calvin_dd.yaml
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
ZIP_PATH = DATASET_DIR / "task_D_D.zip"
TASK_DIR = DATASET_DIR / "task_D_D"
CONDA_ENV = "flower_cal"


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)
    return result


def conda_run(python_cmd):
    return run(["conda", "run", "-n", CONDA_ENV, "python", "-c", python_cmd])


def main():
    # 1. Unzip
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found. Is the download complete?")
        sys.exit(1)

    if not TASK_DIR.exists():
        print(f"Unzipping {ZIP_PATH} to {DATASET_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATASET_DIR)
        print("Unzip complete.")
    else:
        print(f"{TASK_DIR} already exists, skipping unzip.")

    # 2. Preprocess: extract relative actions
    # Output: dataset/task_D_D/{split}/extracted/ep_rel_actions.npy
    for split in ("training", "validation"):
        out = TASK_DIR / split / "extracted" / "ep_rel_actions.npy"
        if not out.exists():
            print(f"Preprocessing {split} split...")
            run([
                "conda", "run", "-n", CONDA_ENV,
                "python", str(BASE / "preprocess" / "extract_by_key.py"),
                "-i", str(DATASET_DIR),
                "--in_task", "task_D_D",
                "--in_split", split,
            ])
        else:
            print(f"Preprocessing already done for {split}, skipping.")

    # 3. Verify lang embeddings exist
    lang_dir = TASK_DIR / "training" / "lang_paraphrase-MiniLM-L3-v2"
    if not lang_dir.exists():
        print("WARNING: lang_paraphrase-MiniLM-L3-v2 not found in task_D_D.")
        print("Please generate language embeddings with: python preprocess/generate_lang_embeddings.py")
    else:
        print(f"Language embeddings found: {lang_dir}")

    # 4. Launch training
    print("\nStarting Flower VLA training on task_D_D...")
    run([
        "conda", "run", "-n", CONDA_ENV,
        "python", str(BASE / "flower" / "training_calvin.py"),
        "--config-name=config_calvin_dd",
    ])


if __name__ == "__main__":
    main()
