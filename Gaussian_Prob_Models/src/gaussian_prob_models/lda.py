import numpy as np

from .qda import QDA


"""
    This is also the engine...

    Inheriting from QDA since most of the code is same.
    The only difference is in covariance matrix.

    Linear terms, and ∑_c = ∑ , ∀ c ∈ C, results in linear class boundaries...
"""


class LDA(QDA):
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.pi = np.zeros(n_classes)
        self.mu = np.zeros((n_classes, n_features))
        self.pooled_cov = np.zeros((n_features, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.pi[idx] = len(X_c) / n_samples
            self.mu[idx] = np.mean(X_c, axis=0)

        # Computing pooled covariance matrix which is the weighted average of class covariances
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            diff = X_c - self.mu[idx]
            self.pooled_cov += np.dot(diff.T, diff)

        self.pooled_cov /= n_samples - n_classes

        self.cov = np.array([self.pooled_cov for _ in range(n_classes)])
