# Probabilistic Models

A showcase of *probabilistic models* implemented from scratch using `NumPy` and `SciPy`. This repository demonstrates Bayesian inference techniques with clean, readable implementations.

## Current Models

### Beta-Binomial Model

**Location:** `Beta_Binomial_model/beta_bin_model.py`

Bayesian sequential updating for binary outcomes (success/failure trials).

- **Prior:** Beta(α, β) — conjugate prior for binomial likelihood
- **Likelihood:** Binomial(n, p) — probability of k successes in n trials
- **Posterior:** Beta(α + k, β + n − k) — updated belief after observing data
- **Posterior Predictive:** Beta-Binomial distribution — predicts future observations

**Key Capabilities:**

- Sequential Bayesian updating (observe batch → update → repeat)
- Full posterior density computation over probability p
- Posterior mean, variance, and predictive distributions
- Visualization of belief evolution across observations

**Use Cases:** A/B testing, clinical trials, quality control, any sequential binary decision problem.

---

### Dirichlet-Multinomial Model (Dirichlet-Multinomial Naive Bayes)

**Location:** `Dirichlet_Multinomial_Model/dirichlet_multinomial_model.py` (clean)  
**Mathematical details:** `Dirichlet_Multinomial_Model/doc.md`

Full Bayesian text classifier using the Dirichlet-Multinomial compound distribution.

- **Prior:** Dirichlet(α) over word probability vectors θ
- **Likelihood:** Multinomial(N, θ) — document word counts
- **Posterior:** Dirichlet(α + word_counts) — conjugate update per class
- **Predictive:** Dirichlet-Multinomial — θ integrated out analytically

**Key Capabilities:**

- Full Bayesian inference (no point estimates)
- Analytical posterior moments: mean, mode, variance per word per class
- Marginal posterior densities (Beta marginals for individual word probabilities)
- Posterior predictive distribution for next-word prediction
- Synthetic data generator for experimentation
- Built-in comparison with classic Multinomial Naive Bayes

**Visualization:**

- Marginal posterior densities (Beta distributions)
- Error rate vs. training set size (DM vs MNB comparison)
- True vs. learned word probability bar charts

**Use Cases:** Text classification, document categorization, spam detection, topic modeling — especially effective with small training sets where uncertainty quantification matters.

---

### Bayesian Naive Bayes

**Location:** `BayesianNaiveBayes/bayesian_nb.py`  
**Mathematical details:** `BayesianNaiveBayes/bayesian_nb_notes.md`

Binary-feature Naive Bayes classifier with a full Bayesian treatment and mutual-information feature selection.

- **Prior:** $\text{Dirichlet}(\alpha)$ over class proportions $\pi$; $\text{Beta}(\beta_1, \beta_0)$ over each per-class feature probability $\theta_{cj}$
- **Likelihood:** Product of $\text{Bernoulli}(x_j \mid \theta_{cj})$ across features — Naive Bayes conditional independence
- **Posterior:** $\text{Dirichlet}(\alpha + N_c)$ and $\text{Beta}(\beta_1 + n_{cj}, \beta_0 + N_c - n_{cj})$ — conjugate updates per class
- **Predictive:** Posterior-mean plug-in — $P(c \mid \mathbf{x}) \propto \bar{\pi}_c \prod_{j=1}^{D} \bar{\theta}_{cj}^{x_j} (1 - \bar{\theta}_{cj})^{1 - x_j}$, normalized via `logsumexp`

**Key Capabilities:**

- Posterior-mean parameter estimation (no hard point estimates)
- Mutual-information feature selection with configurable top-k
- Stable log-space scoring and softmax normalization using `scipy.special.logsumexp`
- Conjugate updates from Dirichlet and Beta priors (no MCMC)
- Handles sparse binary features well out of the box

**Visualization:**

- Feature MI ranking with selected features highlighted
- P(active | class) heatmap for selected features
- Confusion matrix and ROC curve with AUC
- Held-out accuracy vs. number of kept features
- Predicted probability histogram split by true class

**Use Cases:** Classification with sparse binary features — bag-of-words text, spam filtering, medical/genomic screening — especially when feature selection or reliable probabilities matter.

---

### Dirichlet Compound Multinomial (DCM) — Document Classification

**Location:** `Dirichlet_Compound_Multinomial/` (standalone package, own README)  
**Dataset:** UCI SMS Spam Collection v.1 (5,574 messages) in `Dirichlet_Compound_Multinomial/assets/`

A fully Bayesian text classifier using the Dirichlet-Multinomial compound distribution, with θ integrated out analytically.

- **Prior:** θ_c ~ Dirichlet(α) over per-class word-probability vectors
- **Likelihood:** Multinomial(N, θ_c) — document word counts
- **Posterior:** Dirichlet(α + c_c) — conjugate update from summed per-class counts
- **Predictive:** compound likelihood P(x | α′) — θ integrated out, no point estimates

**Key Capabilities:**

- Full Bayesian inference (no point estimates, no MCMC)
- Log-space compound likelihood via `scipy.special.gammaln` + `logsumexp`
- Stratified train/test split preserving class proportions; vocabulary built on train only (no data leakage)
- From-scratch tokenization, vocabulary, and sparse count-matrix construction (`np.add.at`)
- Built-in evaluation: accuracy, precision/recall/F1, confusion matrix, learning curve, top discriminative words

**Results (UCI SMS Spam):**

- Accuracy **98.75%**, F1 (spam) **95.27%** on a 7,921-word vocabulary
- Error rate falls from ~3.0% (10% of data) to ~1.25% (100%)

**CI:** GitHub Actions workflow runs a benchmark smoke test with an accuracy-regression gate on every push/PR.

**Use Cases:** Text classification, spam detection, document categorization — especially effective on small training sets where posterior uncertainty matters.

---

### Gaussian Discriminant Analysis (LDA / QDA)

**Location:** `Gaussian_Prob_Models/` (standalone package, own README)
**Dataset:** Synthetic Gaussian profiles in `src/gaussian_prob_models/datasets.py` — unequal/equal covariance (binary), 3-class multiclass, and a high-dim stability stress test

Linear and Quadratic Discriminant Analysis from scratch: each class is a multivariate Gaussian, posteriors via Bayes' rule computed in log space.

- **Prior:** π_c = N_c / N — class frequencies
- **Means:** μ_c — per-class sample means
- **QDA covariances:** one MLE covariance Σ_c per class → quadratic decision boundary
- **LDA covariances:** single pooled Σ shared across classes → linear boundary (`LDA` subclasses `QDA`, overriding only `fit`)
- **Predictive:** P(c | x) ∝ π_c · N(x | μ_c, Σ_c) — Mahalanobis + `slogdet` log-likelihoods normalized via `logsumexp`

**Key Capabilities:**

- Full generative probabilistic classifier (exact Bayes posteriors, no discriminative shortcuts)
- Numerically stable log-space scoring; no scikit-learn, PyTorch, or TensorFlow
- One-command pipeline: data → train → figures + `metrics.json` (figures saved to `reports/` and opened on screen)
- Single data generator feeding the pipeline, the CLI shim, and the tests; 17 stdlib-unittest tests run manually, never as a pipeline side effect

**Results (synthetic, 600 samples/profile, 70/30 split, seed 42):**

- QDA **99.44%** on unequal covariances (its showcase); LDA **99.44%** on equal covariances
- Both ≥96.6% on 3-class problems, ~90% in 5-D

**Visualization:**

- LDA/QDA decision-boundary contours per 2-D profile
- Confusion-matrix heatmaps for every profile, accuracy and precision/recall/F1 comparison bars

**Use Cases:** Tabular classification with roughly Gaussian classes — medical biometrics, quality control, sensor data — and as a teaching reference for generative vs. discriminative modeling.

---

## Implementation Philosophy

- **From scratch** — No scikit-learn, PyMC, PyTorch, or TensorFlow; pure NumPy/SciPy
- **Conjugate priors** — Analytical posterior updates, no MCMC or variational inference needed
- **Numerical stability** — *Log-space* computations using `scipy.special.gammaln`
- **Readable code** — Heavily commented, line-by-line explanations in `doc.md`
- **Educational focus** — Every mathematical step is explicit and traceable

## Coming Soon

New models currently in development:

- **Gaussian Process Regression** — Non-parametric Bayesian regression with kernel methods
- **Hidden Markov Models** — Sequential latent state inference with forward-backward algorithm
- **Bayesian Linear Regression** — Conjugate normal-inverse-gamma prior with posterior predictive
- **Variational Autoencoder (from scratch)** — ELBO optimization with reparameterization trick

---

## Repository Structure

```
PPModels/
├── pyproject.toml                  # Project configuration
├── BayesianNaiveBayes/
│   ├── bayesian_nb.py             # Bayesian Naive Bayes implementation + demo
│   └── bayesian_nb_notes.md       # Mathematical documentation
├── Beta_Binomial_model/
│   └── beta_bin_model.py          # Beta-Binomial implementation + demo
├── Dirichlet_Multinomial_Model/
│   ├── dirichlet_multinomial_model.py  # Clean implementation
│   ├── doc.md                      # Full mathematical documentation
│   └── demo.txt                    # Sample data format
├── Dirichlet_Compound_Multinomial/     # DCM text classifier on UCI SMS spam
│   ├── README.md                       # Math, usage, benchmark results
│   ├── pyproject.toml                  # NumPy, SciPy, Matplotlib only
│   ├── LICENSE                         # MIT
│   ├── .github/workflows/ci.yml        # CI with accuracy-regression gate
│   ├── assets/
│   │   ├── SMSSpamCollection           # UCI SMS Spam Collection v.1
│   │   └── readme                      # Dataset description / license
│   └── src/dirichlet_compound_multinomial/
│       ├── __init__.py                 # Public API (dataClass, DCM, Evaluator)
│       ├── data.py                     # Tokenization, vocab, count matrix, split
│       ├── model.py                    # DCM: compound log-likelihood, fit, predict
│       └── eval.py                     # Metrics, confusion matrix, learning curve
└── Gaussian_Prob_Models/               # LDA/QDA classifiers on synthetic Gaussian data
    ├── README.md                       # Math, usage, benchmark results
    ├── pyproject.toml                  # NumPy, SciPy, Matplotlib, Seaborn
    ├── requirements.txt                # Pinned dependency versions
    ├── data/generate_data.py           # CLI shim over the packaged generator
    ├── tests/test_models.py            # 17 stdlib-unittest tests (run manually)
    └── src/gaussian_prob_models/
        ├── __init__.py                 # Public API (LDA, QDA, data generator, run_pipeline)
        ├── qda.py                      # QDA: fit, log-likelihood, predict/proba
        ├── lda.py                      # LDA(QDA): pooled-covariance fit
        ├── datasets.py                 # Synthetic data generator (single source of truth)
        ├── evaluate.py                 # Metrics, confusion matrix, figures
        └── pipeline.py                 # One-command pipeline: data -> train -> figures
```

## License

MIT
