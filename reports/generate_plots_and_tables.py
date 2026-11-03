import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)


def generate_visualizations():
    # 1. main comparison boxplots (2 in line 1, 1 in line 2 for superior legibility)
    main_csv = RESULTS_DIR / "main_model_comparison.csv"
    if main_csv.exists():
        df_main = pd.read_csv(main_csv)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # test macro-f1 score distribution across splits
        sns.boxplot(data=df_main, x="model", y="macro_f1", hue="model", legend=False, ax=ax1, palette="Blues_r")
        ax1.set_title("Test Macro-F1 Score (10 Splits)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Macro-F1 Score", fontsize=11)
        ax1.set_xlabel("Model", fontsize=11)
        ax1.tick_params(axis="x", rotation=35)
        ax1.set_ylim(0.980, 1.003)

        # test cross-entropy log-loss (probability calibration)
        sns.boxplot(data=df_main, x="model", y="log_loss", hue="model", legend=False, ax=ax2, palette="Reds_r")
        ax2.set_title("Test Multi-class Log-Loss (Lower is Better)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Log-Loss (Cross-Entropy)", fontsize=11)
        ax2.set_xlabel("Model", fontsize=11)
        ax2.tick_params(axis="x", rotation=35)
        ax2.set_ylim(-0.005, 0.115)

        plt.tight_layout()
        plot1_path = FIGURES_DIR / "01_model_comparison_boxplots.png"
        fig.savefig(plot1_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # format latex summary table
        summary_df = df_main.groupby(["category", "model"]).agg({
            "accuracy": ["mean", "std"],
            "macro_f1": ["mean", "std"],
            "log_loss": ["mean", "std"],
            "active_features": ["mean"],
            "fit_time_s": ["mean", "std"],
        }).reset_index()

        summary_df.columns = [
            "Category", "Model",
            "Acc_Mean", "Acc_Std",
            "F1_Mean", "F1_Std",
            "Loss_Mean", "Loss_Std",
            "Active_Features",
            "Time_Mean", "Time_Std"
        ]

        latex_rows = []
        for _, row in summary_df.iterrows():
            latex_rows.append(
                f"{row['Model']} & {row['Category']} & "
                f"${row['Acc_Mean']:.3f} \\pm {row['Acc_Std']:.3f}$ & "
                f"${row['F1_Mean']:.3f} \\pm {row['F1_Std']:.3f}$ & "
                f"${row['Loss_Mean']:.3f} \\pm {row['Loss_Std']:.3f}$ & "
                f"{int(round(row['Active_Features']))} & "
                f"${row['Time_Mean']:.2f} \\pm {row['Time_Std']:.2f}$ \\\\"
            )

        table_content = r"""\begin{table}[ht]
\centering
\caption{Cross-Validated Performance, Sparsity, and Computational Cost Across 10 Repeated Stratified Splits ($p=1000$).}
\label{tab:main_model_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Family} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Log-Loss} & \textbf{Active Genes} & \textbf{Fit Time (s)} \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
        tex_path = TABLES_DIR / "main_comparison_table.tex"
        with open(tex_path, "w") as f:
            f.write(table_content)


    # 2. sample efficiency curves
    n_csv = RESULTS_DIR / "n_efficiency_comparison.csv"
    if n_csv.exists():
        df_n = pd.read_csv(n_csv)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(
            data=df_n,
            x="n_train",
            y="macro_f1",
            hue="model",
            style="model",
            markers=True,
            dashes=False,
            markersize=8,
            linewidth=2,
            ax=ax
        )
        ax.set_title("Sample Efficiency Stress Test: Macro-F1 vs. Training Sample Size", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Training Samples ($n_{train}$)", fontsize=11)
        ax.set_ylabel("Test Macro-F1 Score", fontsize=11)
        ax.set_ylim(0.960, 1.003)
        ax.set_yticks(np.arange(0.96, 1.005, 0.01))
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        plot2_path = FIGURES_DIR / "02_n_efficiency_curves.png"
        fig.savefig(plot2_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    # 3. feature selection stability barplot
    stab_csv = RESULTS_DIR / "feature_stability.csv"
    if stab_csv.exists():
        df_stab = pd.read_csv(stab_csv)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bar_df = df_stab.melt(
            id_vars=["model"],
            value_vars=["mean_jaccard", "mean_dice"],
            var_name="Metric",
            value_name="Score"
        )
        bar_df["Metric"] = bar_df["Metric"].map({"mean_jaccard": "Jaccard Index", "mean_dice": "Dice Coefficient"})

        sns.barplot(data=bar_df, x="model", y="Score", hue="Metric", ax=ax, palette="muted")
        ax.set_title("Feature Selection Stability Across Resampling Splits", fontsize=13, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Overlap Similarity Score", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.6, label="Moderate stability threshold (0.5)")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plot4_path = FIGURES_DIR / "04_feature_stability_barplot.png"
        fig.savefig(plot4_path, dpi=300)
        plt.close(fig)

        # stability summary in latex
        latex_stab = []
        for _, row in df_stab.iterrows():
            latex_stab.append(
                f"{row['model']} & {row['mean_feature_count']:.1f} \\pm {row['std_feature_count']:.1f} & "
                f"${row['mean_jaccard']:.3f} \\pm {row['std_jaccard']:.3f}$ & "
                f"${row['mean_dice']:.3f} \\pm {row['std_dice']:.3f}$ \\\\"
            )

        stab_table_content = r"""\begin{table}[ht]
\centering
\caption{Feature Selection Stability and Signature Size Across 10 Repeated Resampling Splits.}
\label{tab:feature_stability}
\resizebox{0.85\linewidth}{!}{%
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Mean Signature Size ($k$)} & \textbf{Pairwise Jaccard} & \textbf{Pairwise Dice} \\
\midrule
""" + "\n".join(latex_stab) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
        stab_tex_path = TABLES_DIR / "feature_stability_table.tex"
        with open(stab_tex_path, "w") as f:
            f.write(stab_table_content)


    # 4. pareto trade-off frontier
    if main_csv.exists():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        df_agg = df_main.groupby(["model", "category"]).agg({
            "log_loss": "mean",
            "active_features": "mean",
            "fit_time_s": "mean"
        }).reset_index()

        scatter = sns.scatterplot(
            data=df_agg,
            x="active_features",
            y="log_loss",
            hue="category",
            size="fit_time_s",
            sizes=(100, 600),
            palette="deep",
            ax=ax
        )
        for _, r in df_agg.iterrows():
            ax.annotate(
                r["model"],
                (r["active_features"] * 1.05, r["log_loss"]),
                fontsize=11
            )

        ax.set_title("Pareto Trade-Off: Active Features (Sparsity) vs. Test Log-Loss", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Active Features (Log Scale)", fontsize=11)
        ax.set_ylabel("Test Log-Loss (Calibration / Quality)", fontsize=11)
        ax.set_xscale("log")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        plot5_path = FIGURES_DIR / "05_pareto_frontier.png"
        fig.savefig(plot5_path, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    generate_visualizations()
