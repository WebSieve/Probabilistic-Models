import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

"""

Note:-
    I keep having trouble with dimensions and indexes.
    otherwise everything else runs in a flow state.

"""


class BayesianNaiveBayes:
    def __init__(self, alpha=1.0, beta_0=1.0, beta_1=1.0):
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def fit(self, X: np.ndarray, y: np.ndarray, top_k_features=None):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.class_counts = np.zeros(self.n_classes)
        self.feature_counts = np.zeros((self.n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_counts[idx] = X_c.shape[0]
            self.feature_counts[idx] = X_c.sum(axis=0)

        # posterior means under Beta priors
        self.pie_bar = (self.class_counts + self.alpha) / (
            n_samples + self.alpha * self.n_classes
        )
        self.theta_bar = (self.feature_counts + self.beta_1) / (
            self.class_counts[:, np.newaxis] + self.beta_0 + self.beta_1
        )

        if top_k_features is not None and top_k_features < n_features:
            self.mi_scores = self._compute_mi_scores()
            self.selected_features = np.argsort(self.mi_scores)[::-1][:top_k_features]
            self.theta_bar = self.theta_bar[:, self.selected_features]
        else:
            self.selected_features = np.arange(n_features)

    def _compute_mi_scores(self):
        n_features = self.theta_bar.shape[1]
        # marginal success rate, same Beta smoothing so the estimates line up
        theta_j = (self.feature_counts.sum(axis=0) + self.beta_1) / (
            self.class_counts.sum() + self.beta_0 + self.beta_1
        )
        theta_jc = np.clip(self.theta_bar, 1e-12, 1 - 1e-12)

        mi_scores = np.zeros(n_features)
        for j in range(n_features):
            I = 0.0
            for idx, c in enumerate(self.classes):
                p_c = self.pie_bar[idx]
                th = theta_jc[idx, j]
                I += p_c * (
                    th * np.log(th / theta_j[j])
                    + (1 - th) * np.log((1 - th) / (1 - theta_j[j]))
                )
            mi_scores[j] = I
        return mi_scores

    def predict_log_proba(self, X):
        if self.selected_features is not None:
            X = X[:, self.selected_features]

        theta = np.clip(self.theta_bar, 1e-12, 1 - 1e-12)
        log_theta = np.log(theta)
        log_1m_theta = np.log1p(-theta)

        log_scores = np.zeros((X.shape[0], self.n_classes))
        for idx in range(self.n_classes):
            log_scores[:, idx] = (
                np.log(self.pie_bar[idx])
                + X @ log_theta[idx]
                + (1 - X) @ log_1m_theta[idx]
            )
        return log_scores

    def predict_proba(self, X):
        log_scores = self.predict_log_proba(X)
        return np.exp(log_scores - logsumexp(log_scores, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]


def _confusion(y_true, y_pred, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            cm[i, j] = int(np.sum((y_true == a) & (y_pred == b)))
    return cm


def _roc(y_true, proba):
    order = np.argsort(proba)[::-1]
    y = y_true[order]
    p = proba[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for i in range(len(p)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        if i == len(p) - 1 or p[i] != p[i + 1]:
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
    return np.array(fpr), np.array(tpr)


if __name__ == "__main__":
    # 1. Synthetic data: 50 features but only two of them are really predictive
    rng = np.random.default_rng(42)
    X = rng.binomial(1, 0.3, size=(500, 50))
    y = rng.choice([0, 1], size=500)
    X[y == 1, 5] = rng.binomial(1, 0.8, size=y.sum())
    X[y == 0, 12] = rng.binomial(1, 0.8, size=(y == 0).sum())

    # 2. Hold out 20% so the accuracy figure actually means something
    cut = int(0.8 * len(y))
    X_train, X_test, y_train, y_test = X[:cut], X[cut:], y[:cut], y[cut:]

    # 3. Fit with MI feature selection (keep the top 10)
    bnb = BayesianNaiveBayes()
    bnb.fit(X_train, y_train, top_k_features=10)

    preds = bnb.predict(X_test)
    proba = bnb.predict_proba(X_test)[:, 1]
    print(f"Accuracy on held-out data: {(preds == y_test).mean():.3f}")
    print("Selected features:", bnb.selected_features)

    # 4. Which selected features carry the most signal
    print("\nTop discriminative features (activation rate per class):")
    for f in bnb.selected_features[:5]:
        r0 = bnb.feature_counts[0, f] / bnb.class_counts[0]
        r1 = bnb.feature_counts[1, f] / bnb.class_counts[1]
        print(
            f"  feature {f:2d}   class 0: {r0:.2f}   class 1: {r1:.2f}   MI={bnb.mi_scores[f]:.3f}"
        )

    # 5. Diagnostics on the held-out set
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mi = bnb.mi_scores
    colors = [
        "teal" if j in bnb.selected_features else "#c0c0c0" for j in range(len(mi))
    ]
    axes[0, 0].bar(range(len(mi)), mi, color=colors)
    axes[0, 0].set_xlabel("Feature")
    axes[0, 0].set_ylabel("Mutual information")
    axes[0, 0].set_title("Feature MI ranking (selected in teal)")

    im = axes[0, 1].imshow(bnb.theta_bar, cmap="YlGnBu", aspect="auto")
    axes[0, 1].set_yticks(range(bnb.n_classes), bnb.classes)
    axes[0, 1].set_xticks(range(len(bnb.selected_features)), bnb.selected_features)
    axes[0, 1].set_xlabel("Selected feature")
    axes[0, 1].set_ylabel("Class")
    axes[0, 1].set_title("P(active | class) for selected features")
    fig.colorbar(im, ax=axes[0, 1])

    cm = _confusion(y_test, preds, bnb.classes)
    axes[1, 0].imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[1, 0].set_xticks(range(len(bnb.classes)), bnb.classes)
    axes[1, 0].set_yticks(range(len(bnb.classes)), bnb.classes)
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title("Confusion matrix")

    fpr, tpr = _roc(y_test, proba)
    auc = np.trapezoid(tpr, fpr)
    axes[1, 1].plot(fpr, tpr, color="teal", lw=2)
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1, 1].fill_between(fpr, tpr, alpha=0.1, color="teal")
    axes[1, 1].set_xlabel("False positive rate")
    axes[1, 1].set_ylabel("True positive rate")
    axes[1, 1].set_title(f"ROC (AUC = {auc:.3f})")

    fig.tight_layout()

    # 6. How much does feature selection actually buy us, plus calibration
    ks = range(1, 51)
    accs = []
    for k in ks:
        m = BayesianNaiveBayes()
        m.fit(X_train, y_train, top_k_features=k)
        accs.append((m.predict(X_test) == y_test).mean())

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    axes[0].plot(list(ks), accs, color="teal", lw=2)
    axes[0].axvline(10, color="gray", ls="--")
    axes[0].set_xlabel("Top-k features")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Held-out accuracy vs feature count")

    axes[1].hist(
        proba[y_test == 0], bins=10, alpha=0.6, color="#c0c0c0", label="class 0"
    )
    axes[1].hist(proba[y_test == 1], bins=10, alpha=0.6, color="teal", label="class 1")
    axes[1].set_xlabel("P(class 1)")
    axes[1].set_title("Predicted probability by true class")
    axes[1].legend()

    fig.tight_layout()
    plt.show()
