import numpy as np
import matplotlib.pyplot as plt
from beta_binomial import SimpleBetaBinomial

# --- Scenario: A New Drug Trial ---
# We are testing a new medicine. 
# Prior Belief: Similar medicines usually cure 30% of people. So we set a prior centered around 30%.
# Data: We test it on 50 patients, and 22 are cured (44%).

# 1. Initialize our Prior (Mean = 30 / (30 + 70) = 0.30)
alpha_prior = 30
beta_prior = 70
model = SimpleBetaBinomial(alpha_prior, beta_prior)

# X-axis values for plotting probabilities (0% to 100%)
# np.linspace creates an array of 500 evenly spaced points between 0 and 1
p_values = np.linspace(0, 1, 500)

# Calculate Prior PDF
prior_y = model.prior_pdf(p_values)

# 2. Update with Observed Data (Likelihood happens here internally)
patients_tested = 50
patients_cured = 22
model.posterior_update(k_successes=patients_cured, n_trials=patients_tested)

# Calculate Posterior PDF
posterior_y = model.posterior_pdf(p_values)

# Calculate the Likelihood alone just for visualization
# We scale it down visually so it fits on the same graph as the PDFs
likelihood_y = model.likelihood(p_values, patients_cured, patients_tested)
likelihood_y = likelihood_y * (max(posterior_y) / max(likelihood_y)) 

# 3. Calculate Summary Statistics
mean_val = model.posterior_mean()
mode_val = model.posterior_mode()
var_val = model.posterior_variance()
std_dev = np.sqrt(var_val)

print(f"--- Posterior Statistics ---")
print(f"Mean (Expected probability): {mean_val:.2%}")
print(f"Mode (Most likely probability): {mode_val:.2%}")
print(f"Variance: {var_val:.5f} (Standard Deviation: {std_dev:.2%})\n")

# 4. Posterior Predictive Distribution
# If we test 20 MORE patients tomorrow, what's the chance EXACTLY 'x' are cured?
future_patients = 20
possible_cures = np.arange(0, future_patients + 1)
predictive_y = [model.posterior_predictive(k, future_patients) for k in possible_cures]

# --- PLOTTING ---
plt.style.use('bmh') # A clean plotting style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Prior, Likelihood, Posterior
ax1.plot(p_values, prior_y, '--', color='blue', label=f'Prior Belief (~30%)')
ax1.plot(p_values, likelihood_y, ':', color='green', label=f'Likelihood (Data: 22/50 cures)')
ax1.plot(p_values, posterior_y, '-', color='red', linewidth=2, label=f'Posterior Belief (Update)')
ax1.axvline(mean_val, color='black', alpha=0.5, linestyle='-.', label=f'Post. Mean ({mean_val:.1%})')

ax1.set_title("Bayesian Update: Drug Efficacy")
ax1.set_xlabel("Probability of Cure (p)")
ax1.set_ylabel("Density / Likelihood")
ax1.set_xlim(0, 0.7)
ax1.legend()

# Plot 2: Posterior Predictive
ax2.bar(possible_cures, predictive_y, color='skyblue', edgecolor='black')
ax2.set_title(f"Prediction for Next {future_patients} Patients")
ax2.set_xlabel("Number of Cures")
ax2.set_ylabel("Probability")
ax2.set_xticks(range(0, future_patients + 1, 2))

plt.tight_layout()
plt.savefig('analysis_output.png')
print("Graph saved as analysis_output.png")
