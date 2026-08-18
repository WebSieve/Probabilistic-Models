import numpy as np
from scipy.special import gammaln, betaln
from .data import dataClass


class DCM:
    def __init__(self, alpha_init=1.0) -> None:
        self.alpha_init = alpha_init
        self.classes = None
        self.class_prior = {}
        self.alpha_c = {}

    def _log_beta(self, alpha):
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.classes = np.unique(y_train)
        vocab_size = X_train.shape[1]
        for idx, c in enumerate(self.classes):
            mask = y_train == c
            Xc = X_train[mask]
            self.class_prior[c] = np.sum(mask) / len(y_train)
            self.alpha = np.ones(vocab_size) * self.alpha_init
            self.word_count_sum = np.sum(Xc, axis=0)
            self.alpha_c[c] = self.alpha + self.word_count_sum

    def compute_dcm_log_likelihood(self, doc_counts, c):
        alpha = self.alpha_c[c]
        N_i = np.sum(doc_counts)
        log_multi_coef = gammaln(N_i + 1) - np.sum(gammaln(doc_counts + 1))
        log_beta_ratio = self._log_beta(doc_counts + alpha) - self._log_beta(alpha)
        return log_multi_coef + log_beta_ratio

    def predict_log_proba(self, X: np.ndarray):
        log_proba = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_prior[c])
            for i in range(X.shape[0]):
                log_likelihood = self.compute_dcm_log_likelihood(X[i], c)
                log_proba[i, idx] = log_prior + log_likelihood
        return log_proba

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probas, axis=1)]


if __name__ == "__main__":
    data = dataClass()

    messages, raw_labels = data.getMsgLabel()

    vocab, doc_freq = data.buildVocab(messages)
    X = data.docs_to_matrix(messages, vocab)
    y = data.labels_to_vector(raw_labels)

    X_train, X_test, y_train, y_test = data.train_test_split(X, y, test_frac=0.2)

    dcm = DCM(alpha_init=0.1)
    dcm.fit(X_train, y_train)

    predictions = dcm.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"DCM Model Test Accuracy: {accuracy * 100:.2f}%")
