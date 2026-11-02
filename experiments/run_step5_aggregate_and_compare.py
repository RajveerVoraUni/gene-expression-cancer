import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import RESULTS_DIR
from experiments.run_feature_stability import run_feature_stability_analysis
from reports.generate_plots_and_tables import generate_visualizations


def main():
    print("Step 5: Aggregating all results and producing summary reports")

    # combine metric tables from each individual model step
    dfs = []
    for fname in ["linear_results.csv", "rf_results.csv", "xgb_results.csv", "nn_results.csv"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            dfs.append(pd.read_csv(fpath))
            print(f"Loaded {fname} ({len(dfs[-1])} rows)")

    assert len(dfs) > 0
    main_df = pd.concat(dfs, ignore_index=True)
    out_main_csv = RESULTS_DIR / "main_model_comparison.csv"
    main_df.to_csv(out_main_csv, index=False)

    # combine active feature sets for stability analysis
    all_active = {}
    for fname in ["linear_active_features.json", "rf_active_features.json", "xgb_active_features.json", "nn_active_features.json"]:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath, "r") as f:
                d = json.load(f)
                all_active.update(d)

    out_active_json = RESULTS_DIR / "active_feature_sets.json"
    with open(out_active_json, "w") as f:
        json.dump(all_active, f, indent=2)

    # run pairwise stability analysis
    print("\nRunning feature selection stability...")
    run_feature_stability_analysis()

    # generate publication plots and latex tables
    print("\nGenerating report figures and tables...")
    generate_visualizations()

    # print final comparative summary
    print("\nCross-validated performance summary across 10 splits:")
    summary = main_df.groupby(["category", "model"]).agg({
        "accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "log_loss": ["mean", "std"],
        "active_features": ["mean"],
        "fit_time_s": ["mean", "std"],
    })
    print(summary.to_string())


if __name__ == "__main__":
    main()
