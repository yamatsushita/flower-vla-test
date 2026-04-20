"""
Post-training: reads metrics.csv, generates plots, fills in LaTeX, compiles PDF, and pushes.

Run after training completes:
    conda run -n flower_cal python generate_report.py
"""
import subprocess
import sys
import glob
import pathlib
import re

import numpy as np

BASE = pathlib.Path(__file__).parent
REPORT_DIR = BASE / "report"
FIGURES_DIR = REPORT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Find metrics CSV ──────────────────────────────────────────────────
def find_metrics_csv():
    csvs = sorted(glob.glob(str(BASE / "logs" / "runs" / "**" / "metrics.csv"), recursive=True),
                  key=lambda p: pathlib.Path(p).stat().st_mtime)
    if not csvs:
        print("ERROR: no metrics.csv found under logs/")
        sys.exit(1)
    latest = csvs[-1]
    print(f"Using metrics: {latest}")
    return latest


def load_metrics(csv_path):
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def col(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, "").strip()
        if v:
            try:
                vals.append((float(r.get("epoch", "0") or 0), float(v)))
            except ValueError:
                pass
    if not vals:
        return np.array([]), np.array([])
    ep, v = zip(*sorted(vals))
    return np.array(ep), np.array(v)


# ── 2. Generate plots ────────────────────────────────────────────────────
def generate_plots(csv_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return False

    rows = load_metrics(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training loss
    ep, loss = col(rows, "train/action_loss")
    if len(ep):
        axes[0].plot(ep, loss, "b-o", markersize=3, linewidth=1.5, label="train/action_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Action Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # Validation loss
    ep_v, loss_v = col(rows, "val_act/action_loss")
    if len(ep_v):
        axes[1].plot(ep_v, loss_v, "r-o", markersize=3, linewidth=1.5, label="val/action_loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Action Loss")
        axes[1].set_title("Validation Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    plt.tight_layout()
    out = FIGURES_DIR / "training_loss.pdf"
    plt.savefig(str(out), bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved training loss plot: {out}")

    # Separate val plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if len(ep_v):
        ax2.plot(ep_v, loss_v, "r-o", markersize=3, linewidth=1.5, label="val/action_loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Action Loss")
        ax2.set_title("Validation Loss")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    plt.tight_layout()
    out_v = FIGURES_DIR / "val_loss.pdf"
    plt.savefig(str(out_v), bbox_inches="tight")
    plt.savefig(str(out_v).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved val loss plot: {out_v}")

    return True


# ── 3. Fill in LaTeX with actual metrics ─────────────────────────────────
def update_latex(csv_path):
    rows = load_metrics(csv_path)
    ep_t, loss_t = col(rows, "train/action_loss")
    ep_v, loss_v = col(rows, "val_act/action_loss")

    tex_path = REPORT_DIR / "report.tex"
    tex = tex_path.read_text(encoding="utf-8")

    # Final epoch and loss
    final_train_loss = f"{loss_t[-1]:.4f}" if len(loss_t) else "N/A"
    final_val_loss = f"{loss_v[-1]:.4f}" if len(loss_v) else "N/A"
    n_epochs = int(ep_t[-1]) + 1 if len(ep_t) else 0

    # Insert summary after \section{Training Results}
    summary = (
        f"Training completed {n_epochs} epochs. "
        f"Final train action loss: {final_train_loss}. "
        f"Final validation action loss: {final_val_loss}."
    )
    tex = tex.replace(
        r"\subsection{Training Loss}",
        r"\subsection{Training Loss}" + f"\n\n{summary}\n"
    )

    tex_path.write_text(tex, encoding="utf-8")
    print(f"Updated LaTeX with: {summary}")


# ── 4. Compile PDF ────────────────────────────────────────────────────────
def compile_pdf():
    # Check for pdflatex
    r = subprocess.run(["pdflatex", "--version"], capture_output=True)
    if r.returncode != 0:
        # Try miktex
        r2 = subprocess.run(["miktex-pdflatex", "--version"], capture_output=True)
        if r2.returncode != 0:
            print("pdflatex not found. Install MiKTeX or TeX Live.")
            print("On Windows: winget install MiKTeX.MiKTeX")
            return False
        cmd = "miktex-pdflatex"
    else:
        cmd = "pdflatex"

    print(f"Compiling PDF with {cmd}...")
    for _ in range(2):  # run twice for references
        result = subprocess.run(
            [cmd, "-interaction=nonstopmode", "report.tex"],
            cwd=str(REPORT_DIR),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("pdflatex error:")
            print(result.stdout[-2000:])
            return False

    # BibTeX pass
    subprocess.run(["bibtex", "report"], cwd=str(REPORT_DIR), capture_output=True)
    # Final two passes
    for _ in range(2):
        subprocess.run(
            [cmd, "-interaction=nonstopmode", "report.tex"],
            cwd=str(REPORT_DIR), capture_output=True
        )

    pdf = REPORT_DIR / "report.pdf"
    if pdf.exists():
        print(f"PDF compiled: {pdf} ({pdf.stat().st_size // 1024} KB)")
        return True
    return False


# ── 5. Commit and push ────────────────────────────────────────────────────
def git_push():
    def g(args):
        return subprocess.run(["git", "--no-pager"] + args, cwd=str(BASE), capture_output=True, text=True)

    g(["add", "report/", "automate_dd_training.py"])
    r = g(["commit", "-m",
           "Add training report, plots, and D_D training configs\n\n"
           "- report/report.tex: LaTeX report with training settings and results\n"
           "- report/references.bib: bibliography\n"
           "- report/figures/: training/validation loss plots\n"
           "- report/report.pdf: compiled PDF\n"
           "- automate_dd_training.py: automated D_D setup and training launcher\n\n"
           "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"])
    print("Git commit:", r.stdout.strip() or r.stderr.strip())

    r2 = g(["push", "origin", "main"])
    print("Git push:", r2.stdout.strip() or r2.stderr.strip())


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = find_metrics_csv()
    generate_plots(csv_path)
    update_latex(csv_path)
    ok = compile_pdf()
    if not ok:
        print("\nPDF compilation failed — LaTeX and figures are ready but need manual compilation.")
        print("Run: cd report && pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex")
    git_push()
    print("\nDone!")
