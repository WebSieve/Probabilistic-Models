import numpy as np
from scipy.special import gammaln, betaln


class SimpleBetaBinomial:
    """
    A Beta-Binomial model from scratch with simplified math.
    We use log-space (gammaln, betaln) to prevent computer overflow with large exponents/factorials.
    """

    def __init__(self, alpha_prior, beta_prior):
        # 2. THE PRIOR
        # We define our prior belief BEFORE seeing data.
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # We start our posterior equal to our prior
        self.alpha_post = alpha_prior
        self.beta_post = beta_prior

    def likelihood(self, p, k_successes, n_trials):
        """
        1. THE LIKELIHOOD (Binomial Distribution)
        Answers: "If the true probability is 'p', how likely is it to see exactly 'k' successes in 'n' trials?"

        Math: (n choose k) * p^k * (1-p)^(n-k)
        Code translation: We use logarithms to turn multiplication into addition, which computers handle better.
        Library: scipy.special.gammaln is used because Gamma(n+1) = n! (factorial), and gammaln is log(Gamma).
        """
        # (n choose k) = n! / (k! * (n-k)!)
        # In log space: log(n!) - log(k!) - log((n-k)!)
        log_combinations = gammaln(n_trials + 1) - (
            gammaln(k_successes + 1) + gammaln(n_trials - k_successes + 1)
        )

        # Add the log of the probabilities
        log_prob = (
            log_combinations
            + k_successes * np.log(p)
            + (n_trials - k_successes) * np.log(1 - p)
        )

        # Return normal probability using np.exp
        return np.exp(log_prob)

    def prior_pdf(self, p):
        """
        2. THE PRIOR (Beta Distribution)
        Answers: "Before data, how much do I believe the true probability is 'p'?"

        Math: p^(alpha-1) * (1-p)^(beta-1) / Beta_Function(alpha, beta)
        Library: scipy.special.betaln gives us the log of the Beta Function denominator.
        """
        return self._beta_pdf(p, self.alpha_prior, self.beta_prior)

    def posterior_update(self, k_successes, n_trials):
        """
        3. THE POSTERIOR (Bayesian Update)
        Answers: "How does my belief change AFTER seeing 'k' successes in 'n' trials?"

        Math: Because Beta (prior) and Binomial (likelihood) are conjugate,
              the complex Bayes Theorem simplifies to simple addition!
        """
        self.alpha_post = self.alpha_prior + k_successes
        self.beta_post = self.beta_prior + (n_trials - k_successes)

    def posterior_pdf(self, p):
        """
        Calculates the shape of our new belief after the update.
        Uses the exact same math as the prior, just with updated alpha/beta.
        """
        return self._beta_pdf(p, self.alpha_post, self.beta_post)

    def posterior_predictive(self, k_future, n_future):
        """
        4. THE POSTERIOR PREDICTIVE (Beta-Binomial Distribution)
        Answers: "Given my updated beliefs, what is the probability of EXACTLY 'k' successes in the next 'n' future trials?"

        Math: (n choose k) * B(k + alpha, n - k + beta) / B(alpha, beta)
        """
        log_combinations = gammaln(n_future + 1) - (
            gammaln(k_future + 1) + gammaln(n_future - k_future + 1)
        )

        # Beta function ratio in log space: log(B_new) - log(B_old)
        log_beta_ratio = betaln(
            k_future + self.alpha_post, n_future - k_future + self.beta_post
        ) - betaln(self.alpha_post, self.beta_post)

        log_prediction = log_combinations + log_beta_ratio
        return np.exp(log_prediction)

    # --- SUMMARY STATISTICS ---

    def posterior_mean(self):
        """
        Mean: The "average" expected probability.
        Math: alpha / (alpha + beta)
        """
        return self.alpha_post / (self.alpha_post + self.beta_post)

    def posterior_mode(self):
        """
        Mode: The highest point of the distribution (most likely single value).
        Math: (alpha - 1) / (alpha + beta - 2)
        *Only valid if alpha, beta > 1
        """
        if self.alpha_post > 1 and self.beta_post > 1:
            return (self.alpha_post - 1) / (self.alpha_post + self.beta_post - 2)
        return None  # Undefined for alpha/beta <= 1

    def posterior_variance(self):
        """
        Variance: How spread out or "uncertain" our belief is.
        Math: (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        """
        total = self.alpha_post + self.beta_post
        return (self.alpha_post * self.beta_post) / ((total**2) * (total + 1))

    # --- HELPER METHOD ---
    def _beta_pdf(self, p, alpha, beta):
        """Helper to do the Beta distribution math safely."""
        # Handle edges (0 and 1) carefully to avoid log(0)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        log_num = (alpha - 1) * np.log(p) + (beta - 1) * np.log(1 - p)
        log_den = betaln(alpha, beta)
        return np.exp(log_num - log_den)
