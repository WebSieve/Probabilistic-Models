"""
CLI entry point for synthetic data generation.

Single source of truth lives in the installable package
(``gaussian_prob_models.datasets``); this script is just a thin wrapper so
``python data/generate_data.py`` keeps working from the project root.

"""

import sys
from pathlib import Path

import numpy as np

try:
    from gaussian_prob_models.datasets import generate_synthetic_classification_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gaussian_prob_models.datasets import generate_synthetic_classification_data


if __name__ == "__main__":
    for dtype in ["unequal_cov", "equal_cov", "multiclass", "high_dim"]:
        X, y = generate_synthetic_classification_data(dataset_type=dtype)
        print(
            f"Profile '{dtype}': X shape = {X.shape}, Unique classes = {np.unique(y)}"
        )
