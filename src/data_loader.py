import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import (
    RAW_DATA_PATH,
    RAW_LABELS_PATH,
    CACHE_NPZ_PATH,
    CLASS_NAMES,
    TOTAL_SAMPLES,
    TOTAL_GENES,
)


def build_data_cache(force=False):
    # skip if cache is already built
    if CACHE_NPZ_PATH.exists() and not force:
        return

    # load sample labels and classes
    labels_df = pd.read_csv(RAW_LABELS_PATH)
    sample_ids_labels = labels_df.iloc[:, 0].astype(str).values
    classes = labels_df["Class"].values

    # load gene expression matrix with pyarrow for speed
    data_df = pd.read_csv(RAW_DATA_PATH, engine="pyarrow")
    sample_ids_data = data_df.iloc[:, 0].astype(str).values
    gene_names = data_df.columns[1:].values.astype(str)

    # make sure sample count and gene count match dataset specifications
    assert len(sample_ids_labels) == TOTAL_SAMPLES
    assert len(gene_names) == TOTAL_GENES
    assert np.array_equal(sample_ids_labels, sample_ids_data)

    # convert expression values to float32
    X = data_df.iloc[:, 1:].values.astype(np.float32)
    assert not np.isnan(X).any()
    assert np.isfinite(X).all()

    # encode cancer class names to integer labels
    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(classes)

    # save parsed data to binary file so loading takes under a second
    np.savez_compressed(
        CACHE_NPZ_PATH,
        X=X,
        y=y,
        gene_names=gene_names,
        sample_ids=sample_ids_labels,
        classes=le.classes_,
    )


def load_data():
    # build cache on the first run if missing
    if not CACHE_NPZ_PATH.exists():
        build_data_cache()

    # load binary cache
    cache = np.load(CACHE_NPZ_PATH, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
    gene_names = cache["gene_names"]
    sample_ids = cache["sample_ids"]
    classes = cache["classes"]

    # restore label encoder
    le = LabelEncoder()
    le.classes_ = classes

    return X, y, gene_names, sample_ids, le


if __name__ == "__main__":
    X, y, gene_names, sample_ids, le = load_data()
    print(f"Loaded X: {X.shape}, y: {y.shape}")
