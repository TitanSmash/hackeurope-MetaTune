import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_samples = 20000

# 1. Generate synthetic dataset features and hyperparameters
data = {
    "ttr_score": np.random.uniform(0.1, 0.6, n_samples).round(3),
    "hapax_ratio": np.random.uniform(0.1, 0.7, n_samples).round(3),
    "zipf_alpha": np.random.uniform(1.0, 2.0, n_samples).round(3),
    "lora_r": np.random.choice([4, 8, 16, 32, 64], n_samples),
    "learning_rate": (10 ** np.random.uniform(-5, -3, n_samples)).round(6),
    "dropout": np.random.uniform(0.0, 0.3, n_samples).round(3),
}

# 2. Define the "True" Global Minima (Meta-Learning Logic)
optimal_log_lr = -3.3 - (2.0 * data["ttr_score"])
optimal_dropout = 0.05 + (0.3 * data["hapax_ratio"])
optimal_lora_r = 10 + 50 * (2.0 - data["zipf_alpha"])

# Calculate distance from the current hyperparameter to the "true" optimal
log_lr = np.log10(data["learning_rate"])
d_lr = log_lr - optimal_log_lr
d_drop = data["dropout"] - optimal_dropout
v_lora = (data["lora_r"] - optimal_lora_r) / 64.0  # Normalized distance for LoRA

# 3. Apply Higher-Degree Polynomial Penalties (Creating Local Optima)

# LEARNING RATE: 4th-Degree Double Well
# Global minimum at x=0. Local minimum around x=1.15
lr_penalty = 5.0 * (d_lr**4 - 2.4 * (d_lr**3) + 1.49 * (d_lr**2))

# LORA R: 6th-Degree Triple Well
# Global minimum at x=0. Local minima around x=±0.66
lora_penalty = 15.0 * (v_lora**6 - v_lora**4 + 0.3 * (v_lora**2))

# DROPOUT: Standard Quadratic
# We leave one parameter as a simple convex bowl to stabilize the overall landscape
dropout_penalty = 15.0 * (d_drop**2)

# 4. Inherent Dataset Difficulty
base_loss = 0.5 + (0.8 * data["ttr_score"]) - (0.2 * (data["zipf_alpha"] - 1.0))

# 5. Final Loss Calculation (Base + Polynomial Penalties + Noise)
noise = np.random.normal(0, 0.03, n_samples)
final_loss = base_loss + lr_penalty + dropout_penalty + lora_penalty + noise

# Clip to realistic loss ranges so the GP doesn't break on extreme outliers
data["val_loss"] = np.clip(final_loss, 0.1, 4.0).round(4)

# 6. Save to CSV
df = pd.DataFrame(data)
df.to_csv("history_runs.csv", index=False)

print(f"Saved {n_samples} non-convex synthetic runs to 'history_runs.csv'")
print("\nSample Data:")
print(df.head())
