import numpy as np
from scipy.special import betaln, gammaln
import matplotlib.pyplot as plt


class beta_binomial_model:
    def __init__(self, alpha, beta, error_coef: float) -> None:
        self.alpha_prior = alpha
        self.beta_prior = beta

        self.error_coef = error_coef

        self.alpha_posterior = alpha
        self.beta_posterior = beta

    def likelihood(self, n_trials, k_successes, p) -> tuple[float, float]:
        """
        n choose k = n! / (n-k)! * k! -> constant
        then : p^k * (1-p)^(n-k)
        we will use gammaln function for it

        returns:
            normal_likelihood_probability, log_likelihood_probability
        """
        log_norm = gammaln(n_trials + 1) - (
            gammaln(n_trials - k_successes + 1) + gammaln(k_successes + 1)
        )
        log_likelihood = (k_successes * np.log(p + self.error_coef)) + (
            (n_trials - k_successes) * (np.log(1 - p + self.error_coef))
        )
        log_prob = log_norm + log_likelihood
        return log_prob, np.exp(log_prob)

    def prior(self, p) -> tuple[float, float]:
        """
        Beta-prior = 1 / B(a, b) * [p^(a - 1) * (1 - p)^(b - 1)]
        """
        log_norm = -betaln(self.alpha_prior, self.beta_prior)
        log_probs = ((self.alpha_prior - 1) * np.log(p + self.error_coef)) + (
            (self.beta_prior - 1) * np.log(1 - p + self.error_coef)
        )
        final_log_probs = log_norm + log_probs
        return final_log_probs, np.exp(final_log_probs)

    def posterior_updation(self, k_successes, n_trials) -> None:
        self.alpha_posterior += k_successes
        self.beta_posterior += n_trials - k_successes

    def posterior(self, p) -> tuple[float, float]:
        """
        posterior Batch :
        => p(D | p) * p(p | D)
        => Bin(N1 | p, N1 + N0) * Beta(p | alpha_posterior, beta_posterior)
        => Beta(p | N1 + alpha_posterior, N0 + beta_posterior)
        """

        log_norm = -betaln(self.alpha_posterior, self.beta_posterior)
        log_prob_post = ((self.alpha_posterior - 1) * np.log(p + self.error_coef)) + (
            (self.beta_posterior - 1) * np.log(1 - p + self.error_coef)
        )
        final_log_probs = log_norm + log_prob_post
        return final_log_probs, np.exp(final_log_probs)

    def posterior_predictive(self, k_future_successes, n_future_trials):
        """
        Bb(k_successes_future | alpha_posterior, beta_posterior, n_trials_future) :
        => n_trials_future choose k_successes_future * B(k_successes_future + alpha_posterior, n_trials_future - k_successes_future + beta_posterior) / B(alpha_posterior, beta_posterior)
        """
        log_norm = gammaln(n_future_trials + 1) - (
            gammaln(n_future_trials - k_future_successes + 1)
            + gammaln(k_future_successes + 1)
        )

        beta_log_numerator = betaln(
            k_future_successes + self.alpha_posterior,
            n_future_trials - k_future_successes + self.beta_posterior,
        )
        beta_log_denominator = betaln(self.alpha_posterior, self.beta_posterior)
        log_probs = beta_log_numerator - beta_log_denominator
        final_log_probs = log_norm + log_probs
        return final_log_probs, np.exp(final_log_probs)

    def posterior_predictive_mean(self, n_future_trials) -> float:
        return n_future_trials * (
            self.alpha_posterior / (self.alpha_posterior + self.beta_posterior)
        )

    def posterior_predictive_variance(self, n_trials_future) -> float:
        term1_numerator = n_trials_future * self.alpha_posterior * self.beta_posterior
        term1_denominator = (self.alpha_posterior + self.beta_posterior) ** 2

        term2_numerator = self.alpha_posterior + self.beta_posterior + n_trials_future
        term2_denominator = self.alpha_posterior + self.beta_posterior + 1

        result = (term1_numerator / term1_denominator) * (
            term2_numerator / term2_denominator
        )
        return result

    def posterior_mean(self) -> float:
        return self.alpha_posterior / (self.alpha_posterior + self.beta_posterior)

    def posterior_variance(self) -> float:
        a = self.alpha_posterior
        b = self.beta_posterior
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


# 1. Initializing model
model = beta_binomial_model(alpha=2, beta=2, error_coef=1e-10)

# 2. Setting plotting
p_values = np.linspace(0.01, 0.99, 200)
fig, ax = plt.subplots(figsize=(10, 6))

# Defining data batches (e.g., [successes, trials])
batches = [[4, 5], [10, 10]]

# Plotting the process
for i, (k, n) in enumerate(batches):
    # Get current posterior
    _, post_probs = model.posterior(p_values)
    ax.plot(p_values, post_probs, label=f"Step {i}: After {k}/{n} obs")

    # Updating model with new batch
    model.posterior_updation(k, n)

# Final result
_, final_post = model.posterior(p_values)
ax.plot(p_values, final_post, label="Final Belief", linewidth=3, color="black")

ax.set_title("Evolution of Belief: Bayesian Sequential Updating")
ax.set_xlabel("Probability (p)")
ax.set_ylabel("Posterior Density")
ax.legend()
plt.show()
