"""
Evaluation helpers: splits, metrics, and figures for LDA / QDA.
"""

from pathlib import Path

import numpy as np

_NON_INTERACTIVE_BACKENDS = frozenset(
    {"agg", "pdf", "svg", "ps", "eps", "cairo", "pgf", "template", "inline"}
)


def gui_available():
    """
    True when matplotlib uses a backend that can open figure windows.
    """
    import matplotlib

    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


def _maybe_show(plt, show):
    """Display the current figure non-blocking, else close it.

    Skipped silently on headless backends (e.g. Agg). Returns True when the
    figure was left open; the caller is then responsible for one final
    blocking ``plt.show()`` to keep windows alive.
    """
    if show and gui_available():
        plt.show(block=False)
        return True
    plt.close()
    return False


def train_test_split(X, y, test_frac=0.3, random_state=0):
    """
    Shuffled train/test split (no sklearn dependency).
    """
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    n_test = int(round(len(y) * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def evaluate_model(model_cls, X_train, y_train, X_test, y_test):
    """
    Fit ``model_cls`` and return model + full test/train statistics.
    """
    model = model_cls()
    model.fit(X_train, y_train)
    return {
        "model": model,
        "train_acc": float((model.predict(X_train) == y_train).mean()),
        **classification_stats(y_test, model.predict(X_test)),
    }


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Confusion matrix with rows = true, columns = predicted (numpy only).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    return cm, labels


def classification_stats(y_true, y_pred, labels=None):
    """Accuracy, per-class + macro precision/recall/F1, and confusion matrix."""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return {
        "test_acc": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(cm[i].sum()),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": [str(label) for label in labels],
    }


def plot_boundary(model, X, y, title, save_path, resolution=500, show=True):
    """
    Decision-boundary contour + data scatter for 2-D datasets.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    mesh_preds = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, mesh_preds, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", edgecolor="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    _maybe_show(plt, show)
    return save_path


def plot_confusion_matrix(cm, labels, title, save_path, show=True):
    """Annotated confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    _maybe_show(plt, show)
    return save_path


def plot_metrics_comparison(results, save_path, show=True):
    """4-panel grouped bars: test accuracy + macro precision/recall/F1.

    ``results``: {profile: {"LDA": {...stats...}, "QDA": {...stats...}}}.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("test_acc", "Test accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
    ]
    profiles = list(results.keys())
    x = np.arange(len(profiles))
    width = 0.35

    _, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        lda_vals = [results[p]["LDA"][key] for p in profiles]
        qda_vals = [results[p]["QDA"][key] for p in profiles]
        ax.bar(x - width / 2, lda_vals, width, label="LDA")
        ax.bar(x + width / 2, qda_vals, width, label="QDA")
        ax.set_xticks(x, profiles, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    _maybe_show(plt, show)
    return save_path


def plot_accuracy_bars(results, save_path, show=True):
    """Grouped bar chart: LDA vs QDA test accuracy per dataset profile.

    ``results``: {profile: {"LDA": {"test_acc": ...}, "QDA": {"test_acc": ...}}}.
    """
    import matplotlib.pyplot as plt

    profiles = list(results.keys())
    lda_acc = [results[p]["LDA"]["test_acc"] for p in profiles]
    qda_acc = [results[p]["QDA"]["test_acc"] for p in profiles]

    x = np.arange(len(profiles))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, lda_acc, width, label="LDA")
    plt.bar(x + width / 2, qda_acc, width, label="QDA")
    plt.xticks(x, profiles)
    plt.ylim(0, 1.05)
    plt.ylabel("Test accuracy")
    plt.title("LDA vs QDA test accuracy per dataset")
    plt.legend()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    _maybe_show(plt, show)
    return save_path
