"""
Polls metrics.csv every 5 minutes. When training finishes (or a set number of epochs
accumulate), calls generate_report.py and sends a notification to the log.

Launch as a detached process:
    pythonw monitor_training.py   # or:  python monitor_training.py &
"""
import pathlib, time, csv, subprocess, sys, logging, os

BASE = pathlib.Path(__file__).parent.resolve()
LOGS_DIR = BASE / "logs" / "runs"
MONITOR_LOG = BASE / "monitor.log"
MAX_EPOCHS = 35          # final epoch index (0-based 34 = 35th epoch)
POLL_SEC = 300           # check every 5 minutes
CONDA_ENV = "flower_cal"

logging.basicConfig(
    filename=str(MONITOR_LOG),
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.info


def find_latest_metrics():
    """Return the most-recently-modified metrics.csv under logs/runs/."""
    csvs = sorted(
        LOGS_DIR.rglob("metrics.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return csvs[-1] if csvs else None


def max_epoch_in_csv(path):
    """Return the highest 'epoch' value that has a train/action_loss logged."""
    best = -1
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                ep = row.get("epoch", "").strip()
                loss = row.get("train/action_loss", "").strip()
                if ep and loss:
                    try:
                        best = max(best, int(float(ep)))
                    except ValueError:
                        pass
    except Exception as e:
        log(f"Error reading {path}: {e}")
    return best


def run_report():
    log("Training finished — generating report...")
    result = subprocess.run(
        ["conda", "run", "-n", CONDA_ENV, "python",
         str(BASE / "generate_report.py")],
        cwd=str(BASE),
        capture_output=True, text=True,
    )
    log(f"generate_report stdout: {result.stdout[-1000:]}")
    if result.returncode != 0:
        log(f"generate_report FAILED: {result.stderr[-1000:]}")
    else:
        log("Report generated and pushed successfully.")


def main():
    log("monitor_training.py started")
    reported = False
    while True:
        try:
            csv_path = find_latest_metrics()
            if csv_path is None:
                log("No metrics.csv yet — waiting...")
            else:
                ep = max_epoch_in_csv(csv_path)
                log(f"Latest epoch with loss: {ep} / {MAX_EPOCHS - 1}  ({csv_path.name})")
                if ep >= MAX_EPOCHS - 1 and not reported:
                    run_report()
                    reported = True
                    log("All done — monitor exiting.")
                    sys.exit(0)
        except Exception as e:
            log(f"Unexpected error: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
