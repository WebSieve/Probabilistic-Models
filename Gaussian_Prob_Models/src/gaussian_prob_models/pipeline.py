"""
One-command pipeline: data -> train -> figures + full statistics.

    PYTHONPATH=src python -m gaussian_prob_models

All figures and ``metrics.json`` land in ``reports/``.
Tests are NOT run here; run them manually with::
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

import argparse
import json
from pathlib import Path

from .datasets import generate_synthetic_classification_data
from .evaluate import (
    evaluate_model,
    gui_available,
    plot_accuracy_bars,
    plot_boundary,
    plot_confusion_matrix,
    plot_metrics_comparison,
    train_test_split,
)
from .lda import LDA
from .qda import QDA

PROFILES = ["unequal_cov", "equal_cov", "multiclass", "high_dim"]
MODELS = {"LDA": LDA, "QDA": QDA}


def run_pipeline(output_dir="reports", n_samples=600, random_state=42, show=True):
    """
    Generate data, train LDA/QDA per profile, save (+ optionally show) figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for profile in PROFILES:
        kwargs = {"n_features": 5} if profile == "high_dim" else {}
        X, y = generate_synthetic_classification_data(
            n_samples=n_samples,
            dataset_type=profile,
            random_state=random_state,
            **kwargs,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        summary[profile] = {}
        for name, cls in MODELS.items():
            ev = evaluate_model(cls, X_train, y_train, X_test, y_test)
            summary[profile][name] = {k: v for k, v in ev.items() if k != "model"}
            plot_confusion_matrix(
                ev["confusion_matrix"],
                ev["labels"],
                f"{name} confusion matrix ({profile}, test set)",
                out / f"confusion_{profile}_{name.lower()}.png",
                show=show,
            )
            if X.shape[1] == 2:  # boundary plots need 2-D data
                plot_boundary(
                    ev["model"],
                    X_test,
                    y_test,
                    f"{name} decision boundary ({profile})",
                    out / f"boundary_{profile}_{name.lower()}.png",
                    show=show,
                )

    figures = [
        plot_accuracy_bars(summary, out / "accuracy_comparison.png", show=show),
        plot_metrics_comparison(summary, out / "metrics_comparison.png", show=show),
    ]
    figures += sorted(out.glob("boundary_*.png"))
    figures += sorted(out.glob("confusion_*.png"))

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    return {
        "results": summary,
        "figures": [str(f) for f in figures],
        "metrics_file": str(metrics_path),
    }


def print_summary(report):
    print(
        "\n---------- Test-set scores (accuracy / macro-precision / macro-recall / macro-F1) ----------"
    )
    for profile, scores in report["results"].items():
        for name in ("LDA", "QDA"):
            s = scores[name]
            print(
                f"{profile:12s} {name}: "
                f"acc={s['test_acc']:.3f} (train {s['train_acc']:.3f})  "
                f"P={s['macro_precision']:.3f}  "
                f"R={s['macro_recall']:.3f}  "
                f"F1={s['macro_f1']:.3f}"
            )
    print("\n---------- Per-class scores ----------")
    for profile, scores in report["results"].items():
        for name in ("LDA", "QDA"):
            for label, c in scores[name]["per_class"].items():
                print(
                    f"{profile:12s} {name} class {label}: "
                    f"P={c['precision']:.3f} R={c['recall']:.3f} "
                    f"F1={c['f1']:.3f} (n={c['support']})"
                )
    print(f"\n---------- Scores saved to {report['metrics_file']} ----------")
    print("------------------ Figures --------------------------")
    for fig in report["figures"]:
        print(f"  {fig}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train LDA/QDA on synthetic data; save + display figures and scores."
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument(
        "--no-show", action="store_true", help="save figures without opening windows"
    )
    args = parser.parse_args(argv)

    report = run_pipeline(output_dir=args.output_dir, show=not args.no_show)
    print_summary(report)
    if not args.no_show and gui_available():
        import matplotlib.pyplot as plt

        if plt.get_fignums():  # keep windows alive until you close them
            plt.show()
    print("\nPIPELINE OK: figures and metrics written to reports/.")
