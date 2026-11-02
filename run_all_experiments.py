import sys
import time
import subprocess
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_command(script_path):
    # execute script in isolated python process to prevent openmp conflicts
    cmd = [sys.executable, str(script_path)]
    print(f"\n>>> Running: {script_path.name}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=root_dir)
    if res.returncode != 0:
        raise RuntimeError(f"Step {script_path.name} failed with exit code {res.returncode}")
    print(f">>> Finished: {script_path.name} in {time.time() - t0:.1f}s")


def main():
    total_start = time.time()
    print("=" * 80)
    print("TCGA PAN-CANCER RNA-SEQ ML CLASSIFICATION PIPELINE")
    print("Reproducible 10-Split Multi-Model Benchmark & Synthesis")
    print("=" * 80)

    # list of pipeline steps to run sequentially
    steps = [
        root_dir / "experiments" / "run_step1_linear.py",
        root_dir / "experiments" / "run_step2_random_forest.py",
        root_dir / "experiments" / "run_step3_xgboost.py",
        root_dir / "experiments" / "run_step4_neural_nets.py",
        root_dir / "experiments" / "run_step5_aggregate_and_compare.py",
        root_dir / "experiments" / "run_step6_n_efficiency.py",
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}]")
        run_command(step)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"FULL BENCHMARK PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s!")
    print("Results saved to: results/")
    print("Figures and tables saved to: reports/figures/ and reports/tables/")
    print("=" * 80)


if __name__ == "__main__":
    main()
