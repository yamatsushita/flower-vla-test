"""
Automation script: waits for task_D_D download, stops debug training,
unzips, preprocesses, and launches real D->D training.

Run with:
    conda run -n flower_cal python automate_dd_training.py
"""
import subprocess
import sys
import time
import os
import pathlib
import zipfile

BASE = pathlib.Path(r"C:\Users\yamatsushita\Home\Projects.2025\remote_cli\w1_20260420\flower-vla-test")
ZIP = BASE / "dataset" / "task_D_D.zip"
TASK_DIR = BASE / "dataset" / "task_D_D"
LOG = BASE / "automation.log"
CONDA_ENV = "flower_cal"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, **kw):
    log("RUN: " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log("ERROR: exit code " + str(r.returncode))
        sys.exit(1)
    return r


def ps(cmd):
    return subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True)


log("=== Automation started ===")

# ── 1. Poll BITS download ────────────────────────────────────────────────
log("Polling BITS download for task_D_D.zip...")
while True:
    r = ps("Get-BitsTransfer | Where-Object {$_.DisplayName -like '*Calvin*'} | Select-Object -ExpandProperty JobState")
    state = r.stdout.strip()
    r2 = ps(
        "Get-BitsTransfer | Where-Object {$_.DisplayName -like '*Calvin*'} | "
        "ForEach-Object { [math]::Round($_.BytesTransferred/1GB,1).ToString() + 'GB/' + "
        "[math]::Round($_.BytesTotal/1GB,1).ToString() + 'GB' }"
    )
    progress = r2.stdout.strip()
    log(f"BITS state: '{state}'  {progress}")

    if state == "Transferred":
        log("Download reported complete by BITS.")
        break
    if not state:
        # No BITS job — check if file is already fully downloaded
        if ZIP.exists() and ZIP.stat().st_size > 100_000_000_000:
            log(f"No BITS job found, but ZIP is {ZIP.stat().st_size / 1e9:.1f} GB — treating as complete.")
            break
        log("No BITS job found yet, waiting...")
    elif state == "Error":
        log("BITS transfer in error state. Exiting.")
        sys.exit(1)

    time.sleep(120)

# ── 2. Finalize BITS ─────────────────────────────────────────────────────
ps("Get-BitsTransfer | Where-Object {$_.DisplayName -like '*Calvin*'} | Complete-BitsTransfer")
log("BITS transfer finalized.")
time.sleep(5)

# ── 3. Kill debug training processes ─────────────────────────────────────
log("Killing Python training processes...")
r = ps(
    "Get-Process | Where-Object {$_.Name -eq 'python'} | "
    "ForEach-Object { $id = $_.Id; "
    "try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host ('Stopped '+$id) } "
    "catch { Write-Host ('Skip '+$id) } }"
)
log("Kill output: " + r.stdout.strip()[:300])
time.sleep(10)

# ── 4. Unzip ─────────────────────────────────────────────────────────────
if TASK_DIR.exists():
    log(f"{TASK_DIR} already exists, skipping unzip.")
else:
    log(f"Unzipping {ZIP} ({ZIP.stat().st_size / 1e9:.1f} GB)... this will take 20-40 minutes")
    with zipfile.ZipFile(str(ZIP), "r") as z:
        total = len(z.namelist())
        for i, member in enumerate(z.infolist()):
            z.extract(member, str(BASE / "dataset"))
            if i % 10000 == 0:
                log(f"  Unzip progress: {i}/{total} files")
    log("Unzip complete.")

# ── 5. Preprocess relative actions ───────────────────────────────────────
for split in ("training", "validation"):
    out = TASK_DIR / split / "extracted" / "ep_rel_actions.npy"
    if out.exists():
        log(f"Preprocess already done for {split}, skipping.")
        continue
    log(f"Preprocessing {split} split (extract_by_key.py)...")
    run([
        "conda", "run", "-n", CONDA_ENV,
        "python", str(BASE / "preprocess" / "extract_by_key.py"),
        "-i", str(BASE / "dataset"),
        "--in_task", "task_D_D",
        "--in_split", split,
    ])
    log(f"Preprocessing {split} done.")

# ── 6. Launch D->D training ───────────────────────────────────────────────
log("Starting Flower VLA D->D training (35 epochs, ~14 hours expected)...")
os.chdir(str(BASE))
run([
    "conda", "run", "-n", CONDA_ENV,
    "python", str(BASE / "flower" / "training_calvin.py"),
    "--config-name=config_calvin_dd",
])

log("=== Training complete! ===")

# ── 7. Generate report and push ───────────────────────────────────────────
log("Generating report and pushing to GitHub...")
run([
    "conda", "run", "-n", CONDA_ENV,
    "python", str(BASE / "generate_report.py"),
])
log("=== Report generated and pushed! ===")
