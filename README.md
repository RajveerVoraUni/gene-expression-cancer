# Flexibility versus Interpretability in High-Dimensional Multiclass Cancer Classification from RNA-Seq Data

Authors: Rajveer Vora · Amartya Amritanshu · P. Sanjith Reddy

Comparative study of sparse linear, tree-ensemble, and neural models for tumor-type classification from extremely high-dimensional gene-expression data (\(p >> n\)).

## Objective

We study how predictive performance, model flexibility, feature-selection stability, interpretability, and computational cost trade off when classifying tumor type from RNA-Seq data.

The models form a progression from simple & interpretable to flexible & opaque:
Logistic → Lasso → Elastic Net → XGBoost → Sparse NN

The goal is not to chase the highest accuracy.  
It is to measure how much predictive gain (if any) we obtain for each step up in complexity and loss of interpretability.

## Research Questions

| # | Question |
|---|----------|
| Q1 | How does classification performance vary across linear, regularized, tree-ensemble, and neural models in the \(p >> n\) regime? |
| Q2 | Does added flexibility improve classification, and how does this depend on the number of genes \(p\) and training size \(n\)? |
| Q3 | How do the number and nature of selected genes change from sparse linear models to nonlinear and neural models? |
| Q4 | Are selected genes stable under resampling (especially given correlated expression features)? Does Elastic Net select more stably than Lasso? |
| Q5 | What predictive improvement, if any, is obtained per unit increase in computational cost? |

## Dataset

- Source: [UCI Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene) (Fiorini, 2016; DOI [10.24432/C5R88H](https://doi.org/10.24432/C5R88H))
- Size: 801 samples × 20,531 continuous RNA-Seq features (Illumina HiSeq)
- Classes (imbalance ratio ≈ 3.8 : 1):

| Class | Tumor Type                        | Count |
|-------|-----------------------------------|-------|
| BRCA  | Breast invasive carcinoma         | 300   |
| KIRC  | Kidney renal clear cell carcinoma | 146   |
| LUAD  | Lung adenocarcinoma               | 141   |
| PRAD  | Prostate adenocarcinoma           | 136   |
| COAD  | Colon adenocarcinoma              | 78    |

- Native task: Multiclass classification / clustering  
- Genes are anonymized (gene_0 … gene_20530) → no biological annotation of selected features is possible

## Models

| Model                    | Flexibility   | Interpretability | Feature Selection                      |
|--------------------------|---------------|------------------|----------------------------------------|
| Multinomial Logistic     | Low           | Very high        | None                                   |
| Lasso Logistic           | Low           | Very high        | Exact zeros (nonzero coefficients)     |
| Elastic Net Logistic     | Low–moderate  | High             | Exact zeros + grouping of correlated genes |
| XGBoost                  | High          | Moderate–low     | Importance / SHAP / permutation        |
| Sparse-input Neural Net  | Very high     | Low              | Group Lasso on input-weight columns    |

All penalized and neural models use standardized inputs (standardization is fit on training folds only).

## Experimental Design

### Data splits
- Fixed stratified test set (20% ≈ 160 samples) — never used for tuning or selection
- Remaining pool (≈ 641 samples) supplies all training data
- Results averaged over ≥ 20 repeated stratified splits (same splits used across models so comparisons are paired)

### Preprocessing pipeline (leakage-controlled)

expression → (optional transform) → variance filtering → standardization → model

Every data-dependent step (variance filter, standardization, PCA, importance ranking, hyperparameter tuning) is fit inside the training folds only.

### Varying dimension \(p\)
Genes ranked by training-set variance; top-\(p\) used:
\[
p ϵ \{50,\; 100,\; 500,\; 1000,\; 5000,\; 20531\}
\]
Relative to the training pool this gives \(p/n\) from ~0.08 to ~32.

### Varying training size \(n\)
At fixed \(p\) (and repeated across a small set of \(p\) values):
\[
n ϵ \{50,\; 100,\; 200,\; 400,\; 600\}
\]

### Hyperparameter tuning
Nested cross-validation inside the training data.  
For Lasso and Elastic Net both the minimum-error and 1-SE rules are reported (1-SE yields sparser signatures).

## Evaluation

Predictive metrics:
Accuracy, macro-F1, log-loss (overall); precision, recall, F1 per class; confusion matrices.

Interpretability & selection:
- Lasso / Elastic Net → nonzero coefficients  
- XGBoost → top-\(k\) genes by SHAP or permutation importance  
- Sparse NN → top-\(k\) genes by input-weight column norm  

\(k\) is matched to the median size of the Lasso signature so that selected-set comparisons are fair.

Stability:
Over \(B ≈ 50\) bootstrap resamples we compute Jaccard similarity between selected sets and each gene’s selection frequency.

Computational cost:
Wall-clock training time, number of parameters, number of selected genes, and epochs to convergence, all as functions of \(p\) and \(n\).

## Repository Structure

```text
flexibility-interpretability-rnaseq-classification/
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_logistic_lasso_en.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_sparse_nn.ipynb
│   ├── 05_performance_grids.ipynb    # full p × n results
│   ├── 06_selection_stability.ipynb
│   └── 07_cost_interpretability.ipynb
├── src/
│   ├── preprocessing.py
│   ├── splits.py
│   ├── models/
│   │   ├── logistic.py
│   │   ├── lasso_en.py
│   │   ├── xgboost_model.py
│   │   └── sparse_nn.py
│   ├── evaluation.py
│   ├── stability.py
│   └── utils.py
├── results/
├── report/
├── requirements.txt
└── README.md
```
