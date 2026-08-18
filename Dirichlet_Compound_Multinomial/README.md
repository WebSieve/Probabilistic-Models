# Dirichlet Compound Multinomial — Document Classification

A fully Bayesian text classifier built from scratch with NumPy and SciPy. It models documents as bags of words and classifies them using the **Dirichlet-Multinomial compound distribution** — the Dirichlet prior integrated out analytically, with no point estimates, no MCMC, and no machine-learning frameworks.

## The Model

Each class $c$ has a word-probability vector $\theta_c$ over a shared vocabulary of size $V$.

- **Prior:** $\theta_c \sim \text{Dirichlet}(\alpha)$
- **Likelihood:** $\mathbf{x} \sim \text{Multinomial}(N, \theta_c)$, where $\mathbf{x}$ is a document's word-count vector and $N = \sum_v x_v$
- **Posterior (conjugate):** $\theta_c \mid \text{data} \sim \text{Dirichlet}(\alpha + \mathbf{c}_c)$, where $\mathbf{c}_c$ is the summed word counts of class $c$'s training documents
- **Predictive (θ integrated out):** the compound likelihood

$$\log P(\mathbf{x} \mid \alpha') = \log \frac{\Gamma(\alpha'_0)}{\Gamma(N + \alpha'_0)} + \sum_{v=1}^{V} \Big(\log \Gamma(x_v + \alpha'_v) - \log \Gamma(\alpha'_v)\Big)$$

with $\alpha'_0 = \sum_v \alpha'_v$. For a new document, the class posterior is

$$P(c \mid \mathbf{x}) \propto P(c) \cdot P(\mathbf{x} \mid \alpha'_c),$$

computed in log space with `scipy.special.gammaln` and normalized via `logsumexp`.

### Why the compound distribution?

Because $\theta$ is integrated out, predictions reflect *full posterior uncertainty* rather than a single fitted parameter vector. This matters most on small training sets, where the posterior is wide and a point estimate is overconfident.

## Project Structure

```
Dirichlet_Compound_Multinomial/
├── pyproject.toml                 # Package config (NumPy, SciPy, Matplotlib only)
├── LICENSE                        # MIT
├── README.md
├── assets/
│   ├── SMSSpamCollection          # UCI SMS Spam Collection v.1 (5,574 messages)
│   └── readme                     # UCI dataset description / license
└── src/dirichlet_compound_multinomial/
    ├── __init__.py                # Public API
    ├── data.py                    # Tokenization, vocab, count matrix, stratified split
    ├── model.py                   # DCM: fit, compound log-likelihood, predict
    └── eval.py                    # Metrics, confusion matrix, learning curve, top words
```

## Getting Started

Requires Python ≥ 3.12.

```bash
cd Dirichlet_Compound_Multinomial
uv sync                # or: pip install -e .
python -m dirichlet_compound_multinomial.model
```

### As a library

```python
from dirichlet_compound_multinomial import dataClass, DCM, Evaluator

data = dataClass()
messages, labels = data.getMsgLabel()          # load + tokenize SMS corpus
y = data.labels_to_vector(labels)

train_idx, test_idx = data.split_indices(y)    # stratified split (ham:spam 87:13)
vocab, _ = data.buildVocab([messages[i] for i in train_idx])
X = data.docs_to_matrix(messages, vocab)       # (n_docs, V) integer counts

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = DCM(alpha_init=1.0)
model.fit(X_train, y_train)

pred = model.predict(X_test)
Evaluator.report(y_test, pred)                 # accuracy, precision/recall/F1, confusion matrix
```

### Visualizations

```python
import matplotlib.pyplot as plt
from dirichlet_compound_multinomial import Evaluator

Evaluator.plot_confusion_matrix(y_test, pred)
Evaluator.plot_learning_curve(lambda: DCM(alpha_init=1.0), X_train, y_train, X_test, y_test)

top = Evaluator.top_discriminative_words(model, vocab, class_a=1, class_b=0, k=10)
Evaluator.plot_top_words(top)
plt.show()
```

## Results — UCI SMS Spam Collection

Dataset: **5,574** SMS messages (4,827 ham / 747 spam). Stratified 80/20 split, vocabulary of **7,921** words built from training messages only.

| Metric | Value |
| --- | --- |
| Accuracy | 98.75% |
| Precision (spam) | 96.58% |
| Recall (spam) | 94.00% |
| F1 (spam) | 95.27% |

```
Confusion matrix (rows = true, cols = predicted)
       ham  spam
ham   961     5
spam    9   141
```

Learning curve: error falls from ~3.0% (10% of data) to ~1.25% (100% of data).

## Data Attribution

The SMS Spam Collection v.1 is provided by Tiago A. Almeida and José María Gómez Hidalgo (UCI Machine Learning Repository). The corpus is distributed free for research use, as-is. See `assets/readme` for the full description and license terms.

## Implementation Philosophy

- **From scratch** — no scikit-learn, PyTorch, or TensorFlow; pure NumPy/SciPy
- **Conjugate inference** — exact analytical posterior, no sampling
- **Numerical stability** — all computations in log space via `gammaln`/`logsumexp`
- **Stratified evaluation** — train/test split preserves class proportions; vocabulary built on train only (no data leakage)

## License

MIT — see `LICENSE`.
