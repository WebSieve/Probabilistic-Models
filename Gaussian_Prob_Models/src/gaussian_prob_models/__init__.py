"""
Gaussian probabilistic models (LDA / QDA from scratch).
"""

from .datasets import generate_synthetic_classification_data
from .lda import LDA
from .qda import QDA

__all__ = ["LDA", "QDA", "generate_synthetic_classification_data", "run_pipeline"]


def run_pipeline(output_dir="reports", n_samples=600, random_state=42, show=True):
    """
    Full pipeline: trains LDA/QDA on all profiles, scores, saves + displays figures.
    """
    from .pipeline import run_pipeline as _run

    return _run(
        output_dir=output_dir, n_samples=n_samples, random_state=random_state, show=show
    )


def demo(save_path="qda_boundary_demo.png", show=True) -> None:
    """
    Run the QDA decision-boundary demo (previously executed on import).
    """
    from .evaluate import plot_boundary

    # Shared generator (single source of truth in .datasets)
    X, y = generate_synthetic_classification_data(
        n_samples=400, dataset_type="unequal_cov", random_state=42
    )

    # Train QDA
    qda = QDA()
    qda.fit(X, y)

    plot_boundary(
        qda,
        X,
        y,
        "QDA Curved Decision Boundary from Scratch",
        save_path,
        show=show,
    )


def main(argv=None) -> None:
    """
    One-command entry point: data -> train -> figures (saved + displayed).
    """
    from .pipeline import main as _main

    _main(argv)
