"""
Gaussian probabilistic models (LDA / QDA from scratch).
"""

from .datasets import generate_synthetic_classification_data
from .lda import LDA
from .qda import QDA

__all__ = ["LDA", "QDA", "generate_synthetic_classification_data", "run_pipeline"]


def run_pipeline(output_dir="reports", n_samples=600, random_state=42):
    """
    Full pipeline: unit tests are run by ``main``; this trains and plots.
    """
    from .pipeline import run_pipeline as _run

    return _run(output_dir=output_dir, n_samples=n_samples, random_state=random_state)


def demo() -> None:
    """
    Run the QDA decision-boundary demo (previously executed on import).
    """
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Shared generator (single source of truth in .datasets)
    X, y = generate_synthetic_classification_data(
        n_samples=400, dataset_type="unequal_cov", random_state=42
    )

    # Train QDA
    qda = QDA()
    qda.fit(X, y)

    # Plotting decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    mesh_preds = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, mesh_preds, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", edgecolor="k")
    plt.title("QDA Curved Decision Boundary from Scratch")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("qda_boundary_demo.png")
    plt.show()


def main() -> None:
    """
    One-command entry point: data -> train -> figures (saved + displayed).
    """
    from .pipeline import main as _main

    _main()
