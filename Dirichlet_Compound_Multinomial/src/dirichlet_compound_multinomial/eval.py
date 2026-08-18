import numpy as np
import matplotlib.pyplot as plt


class Evaluator:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(np.asarray(y_true) == np.asarray(y_pred))

    @staticmethod
    def confusion_matrix(y_true, y_pred, classes=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if classes is None:
            classes = np.unique(np.concatenate([y_true, y_pred]))
        cls_index = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[cls_index[t], cls_index[p]] += 1
        return cm, classes

    @staticmethod
    def precision_recall_f1(y_true, y_pred, positive=1):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = np.sum((y_pred == positive) & (y_true == positive))
        fp = np.sum((y_pred == positive) & (y_true != positive))
        fn = np.sum((y_pred != positive) & (y_true == positive))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def report(y_true, y_pred, positive=1):
        cm, classes = Evaluator.confusion_matrix(y_true, y_pred)
        acc = Evaluator.accuracy(y_true, y_pred)
        prf = Evaluator.precision_recall_f1(y_true, y_pred, positive)
        print(f"accuracy : {acc:.4f}")
        print(f"precision : {prf['precision']:.4f}")
        print(f"recall : {prf['recall']:.4f}")
        print(f"f1 : {prf['f1']:.4f}")
        print("confusion matrix (rows=true, cols=pred):")
        print(cm)
        return {"accuracy": acc, "confusion_matrix": cm, **prf}

    @staticmethod
    def plot_confusion_matrix(
        y_true, y_pred, classes=None, title="Confusion Matrix", ax=None
    ):
        cm, classes = Evaluator.confusion_matrix(y_true, y_pred, classes)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), [str(c) for c in classes])
        ax.set_yticks(range(len(classes)), [str(c) for c in classes])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        return ax

    @staticmethod
    def learning_curve(
        model_factory,
        X_train,
        y_train,
        X_test,
        y_test,
        fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
        seed=42,
    ):
        rng = np.random.default_rng(seed)
        errors = []
        for frac in fractions:
            n = max(1, int(frac * len(X_train)))
            idx = rng.choice(len(X_train), size=n, replace=False)
            model = model_factory()
            model.fit(X_train[idx], y_train[idx])
            pred = model.predict(X_test)
            errors.append(1.0 - Evaluator.accuracy(y_test, pred))
        return np.array(fractions), np.array(errors)

    @staticmethod
    def plot_learning_curve(
        model_factory,
        X_train,
        y_train,
        X_test,
        y_test,
        fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
        seed=42,
        title="Error Rate vs Training Set Size",
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        fracs, errors = Evaluator.learning_curve(
            model_factory, X_train, y_train, X_test, y_test, fractions, seed
        )
        ax.plot(fracs * 100, errors * 100, marker="o")
        ax.set_xlabel("training set size (%)")
        ax.set_ylabel("test error rate (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    @staticmethod
    def class_word_probabilities(model, c):
        alphas = model.alpha_c[c]
        return alphas / alphas.sum()

    @staticmethod
    def top_discriminative_words(model, vocab, class_a, class_b, k=20):
        p_a = Evaluator.class_word_probabilities(model, class_a)
        p_b = Evaluator.class_word_probabilities(model, class_b)
        log_ratio = np.log(p_a) - np.log(p_b)
        top_idx = np.argsort(log_ratio)[-k:][::-1]
        return [(vocab[i], np.exp(log_ratio[i]), p_a[i], p_b[i]) for i in top_idx]

    @staticmethod
    def plot_top_words(words, ax=None, title="Top Class-Discriminating Words"):
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        labels = [w[0] for w in words]
        ratios = [w[1] for w in words]
        ax.barh(labels[::-1], ratios[::-1])
        ax.set_xscale("log")
        ax.set_xlabel("P(word | spam) / P(word | ham)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)
        return ax

