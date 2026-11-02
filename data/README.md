# Dataset and Split Documentation

## UCI TCGA Pan-Cancer RNA-Seq Dataset
This benchmark uses the standard **TCGA Pan-Cancer HiSeq (801 × 20,531)** gene expression dataset from the UCI Machine Learning Repository:
- **UCI Dataset Repository**: [Gene Expression Cancer RNA-Seq Dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
- **Samples (n)**: 801 patients with primary solid tumors.
- **Features (p)**: 20,531 continuous gene expression levels (log2(RSEM + 1) normalized).
- **Tumor Classes (5-way)**:
  1. `BRCA`: Breast Invasive Carcinoma (n = 300)
  2. `KIRC`: Kidney Renal Clear Cell Carcinoma (n = 146)
  3. `LUAD`: Lung Adenocarcinoma (n = 141)
  4. `PRAD`: Prostate Adenocarcinoma (n = 136)
  5. `COAD`: Colon Adenocarcinoma (n = 78)

---

## Directory Setup
To train models from raw data:
1. Download `TCGA-PANCAN-HiSeq-801x20531.tar.gz` from the UCI repository.
2. Unpack the folder directly into the repository root:
   ```bash
   tar -xvf TCGA-PANCAN-HiSeq-801x20531.tar.gz
   ```
   Ensuring:
   ```
   TCGA-PANCAN-HiSeq-801x20531/
   ├── data.csv     (197 MB: gene expression matrix)
   └── labels.csv   (12 KB: tumor class labels)
   ```

3. When executing any experiment, `src/data_loader.py` automatically builds a high-speed binary cache (`data/data_cache.npz`, ~50 MB) using `pyarrow`. Subsequent loads take under 0.8 seconds.

---

## Frozen Benchmark Splits
To eliminate sampling variance across different modeling families, all models are evaluated on frozen split files committed directly to this repository:

1. **`splits_stratified_10repeats.npz`** (21 KB):
   - 10 repeated stratified 80/20 train/test splits.
   - Training fold: n_train = 640 samples.
   - Test fold: n_test = 161 samples.
   - Identical class distributions preserved in every split.
   - Used for the primary 8-model benchmark, feature stability analysis, and hyperparameter tuning.

2. **`splits_n_efficiency.npz`** (35 KB):
   - 5 independent stratified splits evaluated across varying sample budgets:
     n_train in {30, 60, 120, 250, 640}
   - Fixed test fold (n_test = 161) for fair generalization evaluation under extreme sample scarcity (n ≪ p).
