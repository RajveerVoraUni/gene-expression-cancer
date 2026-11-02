import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent if (Path(__file__).resolve().parent / 'src').exists() else Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import numpy as np
import pandas as pd
from src.config import RESULTS_DIR
from src.stability import compute_stability_metrics, compute_selection_frequencies

def run_feature_stability_analysis():
    print("=== Starting Feature Selection Stability Analysis ===")
    feat_json_path = RESULTS_DIR / "active_feature_sets.json"
    assert feat_json_path.exists(), f"Could not find {feat_json_path}! Run 01_run_main_comparison.py first."

    with open(feat_json_path, "r") as f:
        data = json.load(f)

    stability_records = []
    top_genes_records = []

    models_to_analyze = ["Lasso", "ElasticNet", "RandomForest", "XGBoost", "Sparse_NN"]

    for model in models_to_analyze:
        raw_sets = data.get(model, [])
        feature_sets = [set(s) for s in raw_sets if len(s) > 0]

        if len(feature_sets) >= 2:
            metrics = compute_stability_metrics(feature_sets)
            mean_size = np.mean([len(s) for s in feature_sets])
            std_size = np.std([len(s) for s in feature_sets])

            stability_records.append({
                "model": model,
                "mean_feature_count": mean_size,
                "std_feature_count": std_size,
                "mean_jaccard": metrics["mean_jaccard"],
                "std_jaccard": metrics["std_jaccard"],
                "mean_dice": metrics["mean_dice"],
                "std_dice": metrics["std_dice"],
                "n_splits": len(feature_sets),
            })

            # Selection frequency
            freqs = compute_selection_frequencies(feature_sets)
            for gene, freq in freqs[:25]:
                top_genes_records.append({
                    "model": model,
                    "gene": gene,
                    "selection_frequency": freq,
                    "frequency_pct": (freq / len(feature_sets)) * 100.0,
                })

    stab_df = pd.DataFrame(stability_records)
    stab_csv = RESULTS_DIR / "feature_stability.csv"
    stab_df.to_csv(stab_csv, index=False)
    print(f"Saved stability metrics to: {stab_csv}")
    print(stab_df[["model", "mean_feature_count", "mean_jaccard", "mean_dice"]])

    freq_df = pd.DataFrame(top_genes_records)
    freq_csv = RESULTS_DIR / "recurrent_biomarker_genes.csv"
    freq_df.to_csv(freq_csv, index=False)
    print(f"Saved recurrent genes to: {freq_csv}")

if __name__ == "__main__":
    run_feature_stability_analysis()
