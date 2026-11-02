import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from src.config import (
    SPLITS_PATH,
    N_EFFICIENCY_SPLITS_PATH,
    N_SPLITS,
    TEST_SIZE,
    RANDOM_STATE,
    N_TRAIN_GRID,
)
from src.data_loader import load_data


def generate_frozen_splits():
    # load labels to generate stratified folds
    _, y, _, sample_ids, _ = load_data()
    indices = np.arange(len(y))

    # create outer 80/20 train/test splits
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    split_dict = {}
    n_efficiency_dict = {}

    for split_idx, (train_idx, test_idx) in enumerate(sss.split(indices, y)):
        split_dict[f"split_{split_idx}_train"] = train_idx
        split_dict[f"split_{split_idx}_test"] = test_idx

        # create smaller training subsamples for sample efficiency tests
        y_train = y[train_idx]
        for n_sub in N_TRAIN_GRID:
            if n_sub >= len(train_idx):
                sub_train_idx = train_idx
            else:
                sub_sss = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=n_sub,
                    random_state=RANDOM_STATE + split_idx * 100 + n_sub
                )
                sub_train_local, _ = next(sub_sss.split(train_idx, y_train))
                sub_train_idx = train_idx[sub_train_local]

            n_efficiency_dict[f"split_{split_idx}_n_{n_sub}_train"] = sub_train_idx

    # save frozen splits to disk
    np.savez_compressed(SPLITS_PATH, n_splits=N_SPLITS, **split_dict)
    np.savez_compressed(N_EFFICIENCY_SPLITS_PATH, n_splits=N_SPLITS, n_train_grid=N_TRAIN_GRID, **n_efficiency_dict)


def load_split(split_idx: int):
    # read train and test indices for a given split
    splits = np.load(SPLITS_PATH)
    train_idx = splits[f"split_{split_idx}_train"]
    test_idx = splits[f"split_{split_idx}_test"]
    return train_idx, test_idx


def load_n_efficiency_split(split_idx: int, n_train: int):
    # read subsampled train indices and corresponding test indices
    splits = np.load(SPLITS_PATH)
    n_splits = np.load(N_EFFICIENCY_SPLITS_PATH)
    train_idx = n_splits[f"split_{split_idx}_n_{n_train}_train"]
    test_idx = splits[f"split_{split_idx}_test"]
    return train_idx, test_idx


if __name__ == "__main__":
    generate_frozen_splits()
