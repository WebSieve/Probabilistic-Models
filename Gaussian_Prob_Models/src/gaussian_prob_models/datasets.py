"""
Synthetic classification datasets for LDA / QDA experiments.
"""

import numpy as np


def generate_synthetic_classification_data(
    n_samples=400, dataset_type="unequal_cov", n_features=2, random_state=42
):
    """
    Generates synthetic classification datasets tailored for testing
    Gaussian Discriminant Analysis (GDA), LDA, and QDA.

    Parameters:
    -----------
    n_samples : int( Total number of samples to generate.)
    dataset_type : str
        The geometry/profile of the dataset:
        - 'unequal_cov': Binary classes with distinct covariances (QDA showcase).
        - 'equal_cov': Binary classes sharing the same covariance (LDA showcase).
        - 'multiclass': 3 classes with unique covariances.
        - 'high_dim': High-dimensional feature space to test numerical stability.
    n_features : int (Dimensionality of the feature space (used mainly for 'high_dim').)
    random_state : int (Seed for reproducibility.)

    Returns:
    --------
    X : ndarray of shape (n_samples, n_features) (The feature matrix.)
    y : ndarray of shape (n_samples,) (The integer class labels.)
    """
    np.random.seed(random_state)
    half_samples = n_samples // 2

    if dataset_type == "unequal_cov":
        # Classic QDA win: Class 0 and Class 1 have vastly different spreads and orientations
        mean_0 = np.array([0.0, 0.0])
        cov_0 = np.array([[1.0, 0.3], [0.3, 1.0]])
        X_0 = np.random.multivariate_normal(mean_0, cov_0, half_samples)
        y_0 = np.zeros(half_samples, dtype=int)

        mean_1 = np.array([3.0, 3.0])
        cov_1 = np.array([[2.0, -1.5], [-1.5, 2.5]])
        X_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples - half_samples)
        y_1 = np.ones(n_samples - half_samples, dtype=int)

        X = np.vstack((X_0, X_1))
        y = np.concatenate((y_0, y_1))

    elif dataset_type == "equal_cov":
        # Classic LDA win: Both classes share an identical covariance matrix
        shared_cov = np.array([[1.5, -0.8], [-0.8, 1.5]])

        mean_0 = np.array([-1.5, 0.0])
        X_0 = np.random.multivariate_normal(mean_0, shared_cov, half_samples)
        y_0 = np.zeros(half_samples, dtype=int)

        mean_1 = np.array([2.5, 2.0])
        X_1 = np.random.multivariate_normal(
            mean_1, shared_cov, n_samples - half_samples
        )
        y_1 = np.ones(n_samples - half_samples, dtype=int)

        X = np.vstack((X_0, X_1))
        y = np.concatenate((y_0, y_1))

    elif dataset_type == "multiclass":
        # 3-Class problem to test Softmax / multinomial classification logic
        n_classes = 3
        per_class = n_samples // n_classes

        X_list, y_list = [], []
        means = [np.array([-3, 0]), np.array([3, 0]), np.array([0, 4])]
        covs = [
            np.array([[1.0, 0.2], [0.2, 1.0]]),
            np.array([[1.5, -1.0], [-1.0, 1.5]]),
            np.array([[0.8, 0.0], [0.0, 2.0]]),
        ]

        for c in range(n_classes):
            X_c = np.random.multivariate_normal(means[c], covs[c], per_class)
            y_c = np.full(per_class, c, dtype=int)
            X_list.append(X_c)
            y_list.append(y_c)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

    elif dataset_type == "high_dim":
        # High-dimensional dataset to test numerical stability of log_det and matrix inversion
        per_class = n_samples // 2
        mean_0 = np.zeros(n_features)
        mean_1 = np.ones(n_features) * 1.5

        # Generate random positive-definite covariance matrices via random orthogonal rotation
        A0 = np.random.randn(n_features, n_features)
        cov_0 = np.dot(A0, A0.T) / n_features + np.eye(n_features)

        A1 = np.random.randn(n_features, n_features)
        cov_1 = np.dot(A1, A1.T) / n_features + np.eye(n_features)

        X_0 = np.random.multivariate_normal(mean_0, cov_0, per_class)
        X_1 = np.random.multivariate_normal(mean_1, cov_1, n_samples - per_class)

        X = np.vstack((X_0, X_1))
        y = np.concatenate(
            (np.zeros(per_class, dtype=int), np.ones(n_samples - per_class, dtype=int))
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Shuffle dataset to prevent ordered class blocks
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]
