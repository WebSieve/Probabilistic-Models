import matplotlib.pyplot as plt
import numpy as np
from beta_binomial_model import BetaBinomialModel

# --- A Practical Example: Estimating a Click-Through Rate (CTR) ---

# Let's imagine we are launching a new website feature and want to estimate its CTR.

# ==============================================================================
# Step 1: The Prior — What we believe BEFORE seeing any data.
# ==============================================================================
# Based on past experience, we believe the CTR is likely around 8%.
# We can represent this belief with a Beta distribution. A Beta(8, 92) has a mean of 8/(8+92) = 8%.
# α (alpha) can be seen as prior "successes" (clicks)
# β (beta) can be seen as prior "failures" (no clicks)

print("--- Step 1: Defining the Prior ---")
prior_alpha, prior_beta = 8, 92
model = BetaBinomialModel(alpha=prior_alpha, beta=prior_beta)
print(f"Prior belief: Beta(α={prior_alpha}, β={prior_beta})")
print(f"Prior expected CTR: {model.get_expected_value():.2%}\n")

# Let's visualize the Prior Distribution
p_values = np.linspace(0, 1, 1000)
prior_pdf_values = model.prior_pdf(p_values)

plt.figure(figsize=(15, 12))
plt.subplot(3, 1, 1)
plt.plot(p_values, prior_pdf_values, 'b--', label=f'Prior: Beta({prior_alpha}, {prior_beta})')
plt.title("Step 1: Prior Belief about CTR", fontsize=14)
plt.xlabel("Possible CTR (p)")
plt.ylabel("Density")
plt.xlim(0, 0.25) # Zoom in on the area of interest
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


# ==============================================================================
# Step 2: The Likelihood — The data we observe.
# ==============================================================================
# Now, we run an experiment. We show the feature to 200 users, and 25 of them click it.
# This is our data: k=25 successes in n=200 trials.
# The Likelihood is the probability of seeing this specific data *given* a certain fixed CTR.
# For example, if the true CTR was 10%, what's the likelihood of our data? P(data | p=0.10)
# If the true CTR was 15%, what's the likelihood? P(data | p=0.15)

print("--- Step 2: Observing Data (Likelihood) ---")
successes_observed, trials_observed = 25, 200
print(f"Observed data: {successes_observed} clicks in {trials_observed} trials.\n")

# Let's calculate and plot the likelihood for different values of p
likelihood_values = model.likelihood_pmf(p_values, successes_observed, trials_observed)
# We scale the likelihood to plot it on the same chart as the PDFs
# Note: np.trapz is deprecated, using np.trapezoid
scaled_likelihood = likelihood_values / np.trapezoid(likelihood_values, p_values)

plt.subplot(3, 1, 2)
plt.plot(p_values, scaled_likelihood, 'g', label=f'Likelihood of {successes_observed} clicks in {trials_observed} trials')
plt.title("Step 2: Likelihood of Observed Data", fontsize=14)
plt.xlabel("Possible CTR (p)")
plt.ylabel("Scaled Likelihood")
plt.xlim(0, 0.25)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


# ==============================================================================
# Step 3: The Posterior — Updating our belief AFTER seeing data.
# ==============================================================================
# Bayes' Theorem tells us: Posterior ∝ Likelihood × Prior
# P(p | data) ∝ P(data | p) × P(p)
#
# Because we used a Beta distribution for the prior and our data follows a Binomial
# distribution (conjugate priors!), the math is simple:
#
# α_posterior = α_prior + successes
# β_posterior = β_prior + (trials - successes)

print("--- Step 3: Calculating the Posterior ---")
model.update(successes=successes_observed, trials=trials_observed)
print(f"Posterior belief: Beta(α={model.alpha_posterior}, β={model.beta_posterior})")
print(f"Posterior expected CTR: {model.get_expected_value():.2%}\n")

# Visualize the Posterior
posterior_pdf_values = model.posterior_pdf(p_values)

plt.subplot(3, 1, 3)
plt.plot(p_values, prior_pdf_values, 'b--', label=f'Prior: Beta({prior_alpha}, {prior_beta})')
plt.plot(p_values, scaled_likelihood, 'g:', label='Likelihood (scaled)')
plt.plot(p_values, posterior_pdf_values, 'r-', linewidth=2, label=f'Posterior: Beta({model.alpha_posterior:.0f}, {model.beta_posterior:.0f})')
plt.title("Step 3: Posterior Belief (Prior + Likelihood)", fontsize=14)
plt.xlabel("Possible CTR (p)")
plt.ylabel("Density")
plt.xlim(0, 0.25)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()


# ==============================================================================
# Step 4: The Posterior Predictive Distribution — Predicting future outcomes.
# ==============================================================================
# Now, the main question: "If we show this to 100 more users, what is the
# probability we get exactly 15 clicks?"
#
# We don't use a single value for 'p'. Instead, we average over our entire
# posterior belief about 'p'. This is the Beta-Binomial distribution.
#
# P(k_future | data) = ∫ P(k_future | p) * P(p | data) dp

print("\n--- Step 4: Predicting Future Outcomes ---")
future_trials = 100
print(f"Predicting outcomes for {future_trials} future trials...\n")

# Calculate probability for a range of possible future successes
possible_future_successes = np.arange(0, 31)
predictive_pmf = [model.posterior_predictive_pmf(k, future_trials) for k in possible_future_successes]

# Find the most likely outcome
most_likely_k = np.argmax(predictive_pmf)
highest_prob = predictive_pmf[most_likely_k]

# Example: "What is the probability of exactly 10 clicks?"
prob_10_clicks = model.posterior_predictive_pmf(10, future_trials)
print(f"The probability of exactly 10 clicks in {future_trials} trials is: {prob_10_clicks:.2%}")

plt.figure(figsize=(12, 6))
plt.bar(possible_future_successes, predictive_pmf, color='skyblue', edgecolor='black', label='Posterior Predictive PMF')
plt.axvline(most_likely_k, color='red', linestyle='--', label=f'Most likely outcome: {most_likely_k} clicks ({highest_prob:.2%})')
plt.title(f"Posterior Predictive Distribution for {future_trials} Future Trials", fontsize=14)
plt.xlabel("Number of Future Clicks (k)")
plt.ylabel("Probability")
plt.xticks(np.arange(0, 31, 2))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Example: "What is the probability of 15 or more clicks?"
prob_15_or_more = sum(model.posterior_predictive_pmf(k, future_trials) for k in range(15, future_trials + 1))
print(f"The probability of 15 or more clicks in {future_trials} trials is: {prob_15_or_more:.2%}")
