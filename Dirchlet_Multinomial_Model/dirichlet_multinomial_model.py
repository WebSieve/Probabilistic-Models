# ok so this is my attempt at implementing the dirichlet-multinomial model
# from scratch. i'm still figuring stuff out so the code might not be the
# most efficient but it works.
#
# basically the idea is:
# - we have documents which are just bags of words
# - each class has its own word probability vector theta
# - we put a dirichlet prior on theta (conjugate prior to multinomial)
# - then we can either:
#   1. compute the posterior over theta given data
#    - or integrate theta out for the compound likelihood
# - the compound likelihood is what we use for classification

from collections import Counter
from scipy.special import gammaln
import matplotlib.pyplot as plt
import numpy as np
import re


class dmm:
    """
    dirichlet multinomial model.
    its a full bayesian classifier not a point estimate one.
    """

    def __init__(self, alpha_prior=1.0):
        # alpha_prior is the dirichlet concentration parameter.
        # if u pass a scalar it gets broadcast to every word.
        # higher values = stronger prior belief in uniform word probs.
        self.alpha_prior = alpha_prior

        # these get filled when we call fit()
        self.classes = None
        self.class_word_counts = None
        self.class_counts = None
        self.vocab_size = None
        self.vocab = None

    # data preparation

    def build_vocab(self, documents):
        """
        takes a list of raw document strings and returns a sorted vocab list.
        pretty straightforward - collect all unique words, sort them.
        """
        vocab_set = set()
        for doc in documents:
            # get rid of punctuation and split into words
            words = re.sub(r'[,.!";:?()\-]', " ", doc).lower().split()
            for w in words:
                vocab_set.add(w)
        return sorted(vocab_set)

    def doc_to_counts(self, doc, vocab):
        """
        turn a single document into a count vector of length len(vocab).
        so position i in the output = how many times vocab[i] appears in doc.
        """
        # clean the doc same way as we did for vocab building
        words = re.sub(r'[,.!";:?()\-]', " ", doc).lower().split()
        # count how many times each word appears
        word_counts = Counter(words)
        # build the vector - 0 if word not in this doc
        result = np.array([word_counts[w] for w in vocab], dtype=int)
        return result

    def prepare_data(self, filepath, n_docs=None):
        """
        read a labelled dataset from a file.
        expected format is one doc per line: label\tdocument text

        returns X (document-term count matrix), y (labels), vocab (word list)
        """
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if n_docs is not None:
            lines = lines[:n_docs]

        labels = []
        documents = []
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) != 2:
                # skip bad lines just in case
                continue
            labels.append(parts[0])
            documents.append(parts[1])

        # convert string labels to ints
        unique_labels = sorted(set(labels))
        # make a dict to map label string -> int
        label_map = {}
        for i, lbl in enumerate(unique_labels):
            label_map[lbl] = i
        y = np.array([label_map[lbl] for lbl in labels], dtype=int)

        # build vocab from all documents
        vocab = self.build_vocab(documents)

        # convert all documents to count matrix
        X = np.array([self.doc_to_counts(d, vocab) for d in documents], dtype=int)

        return X, y, vocab

    # core probability functions

    def log_likelihood(self, x, theta):
        """
        multinomial log likelihood.
        formula: log(N! / prod(x_v!)) + sum(x_v * log(theta_v))

        x     = word counts for one doc (vector of length V)
        theta = word probabilities (vector of length V, must sum to 1)
        """
        N = x.sum()
        # log of the multinomial coefficient: N! / (x_1! * x_2! * ... * x_V!)
        # using gammaln for numerical stability (gammaln(n+1) = log(n!))
        log_coeff = gammaln(N + 1) - gammaln(x + 1).sum()
        # the actual probability part: theta_1^x_1 * theta_2^x_2 * ...
        log_prob = (x * np.log(theta)).sum()
        return log_coeff + log_prob

    def log_prior(self, theta, alpha):
        """
        dirichlet log prior density: log Dir(theta | alpha)

        formula: log(1/B(alpha)) + sum((alpha_v - 1) * log(theta_v))
        where log(1/B(alpha)) = log(gamma(alpha_0)) - sum(log(gamma(alpha_v)))
        and alpha_0 = sum(alpha_v)

        i spent a while deriving this one lol but it makes sense now.
        B(alpha) is the multivariate beta function.
        """
        alpha_0 = alpha.sum()
        # the log normalizer: 1/B(alpha)
        log_norm = gammaln(alpha_0) - gammaln(alpha).sum()
        # the kernel: product of theta_v^(alpha_v - 1)
        log_kernel = ((alpha - 1) * np.log(theta)).sum()
        return log_norm + log_kernel

    def log_posterior(self, theta, alpha_prior, x):
        """
        dirichlet log posterior density: log Dir(theta | alpha_prior + x)

        since dirichlet is conjugate to multinomial, the posterior is just
        dirichlet with updated parameters: alpha_posterior = alpha_prior + x
        """
        alpha_posterior = alpha_prior + x
        alpha_0 = alpha_posterior.sum()
        log_norm = gammaln(alpha_0) - gammaln(alpha_posterior).sum()
        log_kernel = ((alpha_posterior - 1) * np.log(theta)).sum()
        return log_norm + log_kernel

    def posterior_predictive(self, alpha_prior, counts):
        """
        predictive probability for the next word.

        P(next word = v | data) = (alpha_v + N_v) / sum(alpha + N)

        this is literally just the posterior mean of theta_v.
        """
        alpha_post = alpha_prior + counts
        return alpha_post / alpha_post.sum()

    # compound log likelihood
    # this is the key formula for classification - we integrate
    # theta out so we dont need to pick a single theta value.

    def compound_log_likelihood(self, x, alpha_posterior):
        """
        dirichlet-multinomial compound log likelihood.

        this is what u get when u integrate theta out:

            P(x | alpha') = integral of P(x | theta) * P(theta | alpha') d_theta

        the closed form is:

            log P(x | alpha') = log_gamma(alpha'_0) - log_gamma(N + alpha'_0)
                              + sum(log_gamma(x_v + alpha'_v) - log_gamma(alpha'_v))

        no theta parameter needed! its been integrated out analytically.
        this is the magic of conjugate priors.
        """
        # total concentration
        a0 = alpha_posterior.sum()
        # total words in this document
        N = x.sum()

        # first part: gamma(alpha'_0) / gamma(N + alpha'_0)
        term1 = gammaln(a0) - gammaln(N + a0)

        # second part: product_v gamma(x_v + alpha'_v) / gamma(alpha'_v)
        term2 = gammaln(x + alpha_posterior).sum() - gammaln(alpha_posterior).sum()

        return term1 + term2

    # training - fit the model

    def fit(self, X, y):
        """
        fit the model by aggregating word counts per class.

        for each class k:
            alpha'_k = alpha_prior + sum of word counts for docs in class k

        X = (n_docs, vocab_size) count matrix
        y = (n_docs,) labels
        """
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)

        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]
        n_classes = len(self.classes)

        # allocate space for per-class aggregates
        self.class_word_counts = np.zeros((n_classes, self.vocab_size), dtype=int)
        self.class_counts = np.zeros(n_classes, dtype=int)

        # aggregate word counts for each class
        for idx, k in enumerate(self.classes):
            mask = y == k
            self.class_counts[idx] = mask.sum()
            self.class_word_counts[idx] = X[mask].sum(axis=0)

    def _get_posterior_alphas(self, class_idx):
        """
        get the posterior dirichlet parameters for a given class.
        just alpha_prior + observed word counts for that class.
        """
        if np.isscalar(self.alpha_prior):
            prior = np.full(self.vocab_size, self.alpha_prior)
        else:
            prior = np.asarray(self.alpha_prior)
        return prior + self.class_word_counts[class_idx]

    # prediction

    def predict_log_proba(self, X):
        """
        compute log P(class | doc) for each doc.
        we assume uniform class prior so we just need log P(doc | class)
        which is the compound likelihood.
        """
        X = np.asarray(X, dtype=int)
        n_docs = X.shape[0]
        n_classes = len(self.classes)

        log_probs = np.zeros((n_docs, n_classes))

        for k in range(n_classes):
            alpha_post = self._get_posterior_alphas(k)
            for i in range(n_docs):
                log_probs[i, k] = self.compound_log_likelihood(X[i], alpha_post)

        return log_probs

    def predict_proba(self, X):
        """
        normalised class probabilities using softmax.
        """
        log_p = self.predict_log_proba(X)
        # subtract max for numerical stability (stops exp from overflowing)
        log_p = log_p - log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        # normalise so each row sums to 1
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        predict the most likely class for each doc.
        just picks the class with highest posterior probability.
        """
        log_p = self.predict_log_proba(X)
        best_idx = np.argmax(log_p, axis=1)
        return self.classes[best_idx]

    # posterior analysis
    # once we've fit the model we can look at the posterior
    # distribution over theta for each class.

    def posterior_mean(self, class_idx):
        """E[theta_v] = alpha'_v / alpha'_0. gives smoothed word probs."""
        alphas = self._get_posterior_alphas(class_idx)
        return alphas / alphas.sum()

    def posterior_mode(self, class_idx):
        """
        mode of the posterior dirichlet.

        formula: (alpha'_v - 1) / (alpha'_0 - V)
        but this is only valid if all alpha'_v > 1.
        if any alpha'_v <= 1 then the mode is on the boundary
        and the formula doesnt work so we return nan.
        """
        alphas = self._get_posterior_alphas(class_idx)
        if np.all(alphas > 1):
            return (alphas - 1) / (alphas.sum() - len(alphas))
        else:
            # mode is on the boundary, cant compute interior mode
            return np.full_like(alphas, np.nan)

    def posterior_variance(self, class_idx):
        """
        variance of each theta_v under the posterior dirichlet.

        var[theta_v] = alpha'_v * (alpha'_0 - alpha'_v)
                       / (alpha'_0^2 * (alpha'_0 + 1))
        """
        alphas = self._get_posterior_alphas(class_idx)
        a0 = alphas.sum()
        return (alphas * (a0 - alphas)) / (a0**2 * (a0 + 1))

    # plotting helpers

    def plot_posterior_marginal(self, class_idx, word_idx, word="", ax=None):
        """
        plot the marginal posterior density of theta for a specific word.

        the marginal of a dirichlet is a beta distribution:
            theta_v ~ beta(alpha'_v, alpha'_0 - alpha'_v)
        """
        from scipy.stats import beta

        if ax is None:
            fig, ax = plt.subplots()

        alphas = self._get_posterior_alphas(class_idx)
        a = alphas[word_idx]
        b = alphas.sum() - a

        if word:
            label = f"word '{word}' (class {self.classes[class_idx]})"
        else:
            label = f"word {word_idx} (class {self.classes[class_idx]})"

        theta_vals = np.linspace(0, 1, 300)
        ax.plot(theta_vals, beta.pdf(theta_vals, a, b), label=label)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # synthetic data for testing

    @staticmethod
    def generate_synthetic_data(
        n_docs=200,
        vocab_size=10,
        n_classes=2,
        alpha_gen=0.5,
        doc_length_range=(5, 50),
        seed=42,
    ):
        """
        generate synthetic documents from a true dirichlet-multinomial process.

        for each class:
            1. draw true theta ~ dirichlet(alpha_gen, alpha_gen, ...)
            2. for each doc, draw length uniformly
            3. draw word counts ~ multinomial(length, theta)

        returns X (counts), y (labels), and the true theta values
        """
        rng = np.random.default_rng(seed)

        # draw true word probs for each class from dirichlet
        true_theta = rng.dirichlet(np.full(vocab_size, alpha_gen), size=n_classes)

        docs = []
        labels = []
        n_per_class = n_docs // n_classes

        for c in range(n_classes):
            lengths = rng.integers(
                doc_length_range[0], doc_length_range[1] + 1, size=n_per_class
            )
            for L in lengths:
                counts = rng.multinomial(L, true_theta[c])
                docs.append(counts)
                labels.append(c)

        return np.array(docs), np.array(labels), true_theta


#  demo / test code

if __name__ == "__main__":
    # testing with synthetic data
    print("-" * 60)
    print("DEMO 1: synthetic data test")
    print("-" * 60)

    # generate data with 8 words, 2 classes
    X, y, true_theta = dmm.generate_synthetic_data(
        n_docs=200, vocab_size=8, n_classes=2, alpha_gen=0.5, doc_length_range=(5, 40)
    )

    # split into train and test (80/20)
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train = X[idx[:split]]
    y_train = y[idx[:split]]
    X_test = X[idx[split:]]
    y_test = y[idx[split:]]

    print(f"training set size: {len(X_train)}")
    print(f"test set size:     {len(X_test)}")
    print()

    # fit model
    model = dmm(alpha_prior=1.0)
    model.fit(X_train, y_train)

    # test accuracy
    y_pred = model.predict(X_test)
    error = np.mean(y_pred != y_test)
    print(f"test error rate: {error:.3f}  (random chance = 0.50)")
    print()

    # compare true theta vs posterior mean for class 0
    print("class 0: true theta vs posterior mean:")
    print(f"{'word':>5}  {'true':>8}  {'mean':>8}  {'var':>8}")
    m = model.posterior_mean(0)
    v = model.posterior_variance(0)
    for w in range(true_theta.shape[1]):
        print(f"{w:>5d}  {true_theta[0, w]:>8.4f}  {m[w]:>8.4f}  {v[w]:>8.6f}")

    print()

    # posterior marginal plot
    print("-" * 60)
    print("DEMO 2: posterior marginal plot")
    print("-" * 60)

    fig, ax = plt.subplots(figsize=(6, 4))
    model.plot_posterior_marginal(0, 2, word="word 2", ax=ax)
    ax.set_title("marginal posterior of theta for word 2 (class 0)")
    plt.tight_layout()
    plt.show()

    # posterior predictive
    print()
    print("=" * 60)
    print("DEMO 3: posterior predictive (next word)")
    print("=" * 60)

    pred = model.posterior_predictive(
        alpha_prior=np.ones(model.vocab_size),
        counts=model.class_word_counts[0],
    )

    # show top 5 most probable next words
    top_idx = np.argsort(pred)[-5:][::-1]
    print("top 5 next-word probs for class 0:")
    for i in top_idx:
        print(f"word {i}: {pred[i]:.4f}")

    print()
    print("done!")
