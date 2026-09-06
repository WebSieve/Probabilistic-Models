import numpy as np
from scipy.special import logsumexp

"""
The main engine of the car...
cov is diff for each class and given the quadratic term,
the class boundaries are all conic sections. 😛
"""


class QDA:
    def __init__(self):
        self.pi = None
        self.mu = None
        self.cov = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize priors, mean, and cov matrix with zeros
        self.pi = np.zeros(n_classes)
        self.mu = np.zeros((n_classes, n_features))
        self.cov = np.zeros((n_classes, n_features, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.pi[idx] = len(X_c) / n_samples
            self.mu[idx] = np.mean(X_c, axis=0)
            self.cov[idx] = np.cov(X_c, rowvar=False)

    def _compute_log_likelihood(self, X: np.ndarray, c):
        _, n_features = X.shape

        diff = X - self.mu[c]
        inv_cov = np.linalg.inv(self.cov[c])
        _, log_det = np.linalg.slogdet(self.cov[c])

        mah_dist_sq = np.sum((diff @ inv_cov) * diff, axis=1)
        log_pdf = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mah_dist_sq)
        return log_pdf

    def predict_log_proba(self, X: np.ndarray):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_joint = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.pi[idx])
            log_likelihood = self._compute_log_likelihood(X, idx)
            log_joint[:, idx] = log_prior + log_likelihood

        return log_joint

    def predict_proba(self, X: np.ndarray):
        log_joint = self.predict_log_proba(X)
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        return np.exp(log_joint - log_norm)

    def predict(self, X: np.ndarray):
        log_joint = self.predict_log_proba(X)
        return self.classes[np.argmax(log_joint, axis=1)]
