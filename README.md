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

**Location:** `Dirchlet_Multinomial_Model/dirichlet_multinomial_model.py` (clean)  
**Mathematical details:** `Dirchlet_Multinomial_Model/doc.md`

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
├── Beta_Binomial_model/
│   └── beta_bin_model.py          # Beta-Binomial implementation + demo
└── Dirchlet_Multinomial_Model/
    ├── dirichlet_multinomial_model.py  # Clean implementation
    ├── d.py                        # Heavily commented + demos
    ├── doc.md                      # Full mathematical documentation
    └── demo.txt                    # Sample data format
```

## License

MIT
