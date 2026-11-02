import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class LeakageFreePreprocessor:
    def __init__(self, p: int = 1000, apply_pca: bool = False, pca_variance: float = 0.95, random_state: int = 42):
        self.p = p
        self.apply_pca = apply_pca
        self.pca_variance = pca_variance
        self.random_state = random_state

        self.selected_gene_indices = None
        self.scaler = None
        self.pca = None

    def fit_transform(self, X_train: np.ndarray, gene_names: np.ndarray = None):
        n_features = X_train.shape[1]

        # calculate variance strictly on training samples
        variances = np.var(X_train, axis=0)

        # select top-p most variable genes
        if self.p is None or self.p >= n_features:
            self.selected_gene_indices = np.where(variances > 1e-12)[0]
        else:
            self.selected_gene_indices = np.argsort(variances)[::-1][:self.p]

        X_train_filtered = X_train[:, self.selected_gene_indices]

        # scale training features to zero mean and unit variance
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_filtered)

        # apply pca on training set if requested
        if self.apply_pca:
            self.pca = PCA(n_components=self.pca_variance, random_state=self.random_state)
            X_train_out = self.pca.fit_transform(X_train_scaled)
        else:
            X_train_out = X_train_scaled

        selected_names = gene_names[self.selected_gene_indices] if gene_names is not None else None
        return X_train_out, self.selected_gene_indices, selected_names

    def transform(self, X_test: np.ndarray):
        assert self.scaler is not None

        # filter to the same genes selected during training
        X_test_filtered = X_test[:, self.selected_gene_indices]

        # transform test set using training mean and scale
        X_test_scaled = self.scaler.transform(X_test_filtered)

        # transform test set using training pca loadings if applicable
        if self.apply_pca:
            X_test_out = self.pca.transform(X_test_scaled)
        else:
            X_test_out = X_test_scaled

        return X_test_out


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.create_splits import load_split

    X, y, gene_names, _, _ = load_data()
    train_idx, test_idx = load_split(0)

    pre = LeakageFreePreprocessor(p=1000)
    Xtr, sel_idx, sel_names = pre.fit_transform(X[train_idx], gene_names)
    Xte = pre.transform(X[test_idx])
    print(f"Xtr: {Xtr.shape}, Xte: {Xte.shape}")
