# Bayesian Naive Bayes

A from-scratch Naive Bayes classifier with a full Bayesian treatment: Beta priors on the per-class feature probabilities, a Dirichlet prior on the class proportions, and optional mutual-information feature selection. Everything is estimated as a **posterior mean** rather than a plain maximum-likelihood estimate, which is what makes the model "Bayesian."

The file ships as a single class, `BayesianNaiveBayes`, plus a few plotting/metric helpers used by the `__main__` demo.

---

## 1. Model setup

We have $N$ samples, each a binary vector $x \in \{0,1\}^D$ (feature $j$ is either present or absent), and a class label $y \in \{1, \dots, K\}$. Naive Bayes assumes features are independent **given the class**:

$$
P(x \mid y=c) = \prod_{j=1}^{D} P(x_j \mid y=c)
$$

with each per-class, per-feature probability a Bernoulli parameter

$$
\theta_{cj} = P(x_j = 1 \mid y = c).
$$

The class proportions themselves are $\pi_c = P(y = c)$.

### 1.1 Priors

We place a Dirichlet prior on the class proportions,

$$
\pi \sim \mathrm{Dir}(\alpha, \dots, \alpha),
$$

and a Beta prior on each Bernoulli parameter,

$$
\theta_{cj} \sim \mathrm{Beta}(\beta_1, \beta_0),
$$

where $\beta_1$ behaves like a "pseudocount" of successes and $\beta_0$ a pseudocount of failures. With the default hyperparameters ($\alpha = \beta_0 = \beta_1 = 1$) the priors are uniform and the estimates below reduce to ordinary Laplace smoothing.

### 1.2 Counting

`fit` counts, for each class $c$:

- $N_c$ — the number of training samples of that class,
- $n_{cj}$ — the number of class-$c$ samples where feature $j$ is active.

### 1.3 Posterior means (the actual estimates)

Because the Dirichlet is conjugate to the multinomial and the Beta is conjugate to the Bernoulli, the posterior is the same family, and the posterior **mean** has a closed form:

$$
\pi_c = \frac{N_c + \alpha}{N + K\alpha}
$$

$$
\theta_{cj} = \frac{n_{cj} + \beta_1}{N_c + \beta_0 + \beta_1}
$$

These are exactly `pie_bar` and `theta_bar` in `fit`. Note that a feature that never appears in class $c$ still gets $\theta_{cj} = \frac{\beta_1}{\beta_0 + \beta_1}$, not $0$, so the model never assigns a hard zero probability to an unseen event.

---

## 2. Mutual-information feature selection

Passing `top_k_features` keeps only the $k$ features with the highest estimated mutual information with the label:

$$
I(X_j; C) = \sum_{c=1}^{K} \pi_c \left[
    \theta_{cj} \log \frac{\theta_{cj}}{\theta_j}
    + (1 - \theta_{cj}) \log \frac{1 - \theta_{cj}}{1 - \theta_j}
\right]
$$

where $\theta_j$ is the marginal (class-averaged) probability that feature $j$ is active:

$$
\theta_j = \frac{\sum_c n_{cj} + \beta_1}{N + \beta_0 + \beta_1}.
$$

A feature is informative when $\theta_{cj}$ differs a lot across classes; if $\theta_{cj} \approx \theta_j$ for every class, its contribution to $I$ is near zero. Note that `theta_j` is smoothed with the *same* Beta hyperparameters as the class-conditional estimates so the two stay consistent — a detail that matters when $\beta_1$ is large. The chosen features are stored in `selected_features` (sorted by MI, descending), and `theta_bar` is sliced down to just those columns before prediction.

---

## 3. Prediction

### 3.1 Log posterior (unnormalized)

For a new point $x$, we score each class in log space to avoid underflow:

$$
\log \tilde{P}(c \mid x) = \log \pi_c + \sum_{j} \left[
    x_j \log \theta_{cj} + (1 - x_j) \log(1 - \theta_{cj})
\right].
$$

Because we work with posterior means, these "posterior" scores already reflect the prior's regularizing effect.

### 3.2 Probabilities

The scores are normalized to a proper distribution with the softmax, computed stably via `scipy.special.logsumexp`:

$$
P(c \mid x) = \frac{\exp\left( \log \tilde{P}(c \mid x) \right)}
                  {\sum_{k} \exp\left( \log \tilde{P}(k \mid x) \right)}.
$$

### 3.3 Class label

`predict` returns $\arg\max_c \log \tilde{P}(c \mid \mathbf{x})$, i.e. the same class as `predict_proba` but without spending computational power on the normalization step.

For the two-class case, the decision rule simplifies to comparing the log odds:

$$
\log \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \log \frac{\bar{\pi}_1}{\bar{\pi}_0} + \sum_{j} x_j \log \frac{\bar{\theta}_{1j}(1 - \bar{\theta}_{0j})}{\bar{\theta}_{0j}(1 - \bar{\theta}_{1j})} + \sum_{j} \log \frac{1 - \bar{\theta}_{1j}}{1 - \bar{\theta}_{0j}}
$$

which makes it obvious that only features where $\bar{\theta}_{0j} \neq \bar{\theta}_{1j}$ contribute to the decision.

---

## 4. Metrics and diagnostics

All evaluation helpers live outside the class and operate on the model's own predictions — no `sklearn` is used.

- **Confusion matrix** (`_confusion`): simple cross-tabulation of true vs. predicted labels.
- **ROC curve** (`_roc`): walks the predicted probabilities from high to low, tracking true/false positive rates at each distinct threshold.
- **AUC** (`np.trapezoid`): area under the ROC curve via the trapezoid rule. 0.5 = coin flip, 1.0 = perfect ranking.

The demo prints the accuracy, the selected features, and the per-class activation rates for the top features, then draws two figures:

1. **Diagnostics (2×2)** — all-feature MI ranking (selected features highlighted), the $\theta_{cj}$ heatmap for selected features, the confusion matrix, and the ROC curve.
2. **Analysis (1×2)** — held-out accuracy vs. the number of kept features (a sweep that re-fits the model for each $k$), and a histogram of predicted probabilities split by true class.

---

## 5. Usage

```python
from bayesian_nb import BayesianNaiveBayes

model = BayesianNaiveBayes(alpha=1.0, beta_0=1.0, beta_1=1.0)
model.fit(X_train, y_train, top_k_features=10)   # binary X; k is optional

proba = model.predict_proba(X_test)              # soft class probabilities
labels = model.predict(X_test)                   # hard class labels
```

Running the file directly (`python bayesian_nb.py`) regenerates the synthetic demo data and both figures.

---

## 6. Tuning notes

- **$\alpha$** controls the smoothing of the class proportions. Large values pull `pie_bar` toward the uniform prior; small values let rare classes show through.
- **$\beta_1$** is the pseudocount for active features, **$\beta_0$** for inactive ones. Higher values shrink $\theta_{cj}$ toward $0.5$, dampening the influence of features with few observations — handy on tiny datasets.
- **`top_k_features`** is a genuine bias/variance dial. The demo's accuracy sweep shows that keeping a handful of informative features beats feeding the model all 50, because the extra dimensions are pure noise.
