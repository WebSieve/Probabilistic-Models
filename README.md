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
└── Dirichlet_Multinomial_Model/
    ├── dirichlet_multinomial_model.py  # Clean implementation
    ├── doc.md                      # Full mathematical documentation
    └── demo.txt                    # Sample data format
```

## License

MIT
