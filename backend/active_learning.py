import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

# BoTorch & GPyTorch specific imports
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf


class PureBotorchOptimizer:
    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df
        self.metafeature_cols = ["ttr_score", "hapax_ratio", "zipf_alpha"]

        self.meta_scaler = StandardScaler()
        self.meta_scaler.fit(self.history_df[self.metafeature_cols])

        # Bounds: [lora_r, log10(learning_rate), dropout]
        self.bounds_real = torch.tensor(
            [[4.0, -5.0, 0.0], [64.0, -3.0, 0.3]], dtype=torch.float64
        )

    def _normalize(self, x_real):
        return (x_real - self.bounds_real[0]) / (
            self.bounds_real[1] - self.bounds_real[0]
        )

    def _unnormalize(self, x_norm):
        return (
            x_norm * (self.bounds_real[1] - self.bounds_real[0]) + self.bounds_real[0]
        )

    def recommend_next_configs(self, current_metafeatures: dict, n_suggestions=3):
        current_mf_array = self.meta_scaler.transform(
            [[current_metafeatures[col] for col in self.metafeature_cols]]
        )
        hist_mf_array = self.meta_scaler.transform(
            self.history_df[self.metafeature_cols]
        )

        distances = distance.cdist(current_mf_array, hist_mf_array, metric="euclidean")[
            0
        ]
        self.history_df["distance"] = distances

        similar_runs = self.history_df.sort_values(by=["distance", "val_loss"]).head(10)

        train_x_real = torch.tensor(
            similar_runs[["lora_r", "learning_rate", "dropout"]].values,
            dtype=torch.float64,
        )
        train_x_real[:, 1] = torch.log10(train_x_real[:, 1])

        train_X = self._normalize(train_x_real)
        train_Y = torch.tensor(
            -similar_runs["val_loss"].values, dtype=torch.float64
        ).unsqueeze(-1)

        gp_model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_mll(mll)

        self.gp_model = gp_model
        self.train_X = train_X
        self.train_Y = train_Y

        best_f = train_Y.max()
        qEI = qExpectedImprovement(model=gp_model, best_f=best_f)

        standard_bounds = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64
        )

        candidates_norm, _ = optimize_acqf(
            acq_function=qEI,
            bounds=standard_bounds,
            q=n_suggestions,
            num_restarts=5,
            raw_samples=20,
        )

        # Save the generated candidates for the plotting function
        self.candidates_norm = candidates_norm

        candidates_real = self._unnormalize(candidates_norm)

        formatted_suggestions = []
        for row in candidates_real:
            lora_continuous = float(row[0])
            lora_snapped = int(2 ** round(np.log2(lora_continuous)))
            lora_r = max(4, min(64, lora_snapped))
            lr = float(10 ** row[1])
            dropout = float(row[2])

            formatted_suggestions.append(
                {
                    "lora_r": lora_r,
                    "learning_rate": round(lr, 6),
                    "dropout": round(dropout, 3),
                }
            )

        return formatted_suggestions

    def plot_1d_surrogates(self, file_prefix="gp_surrogate"):
        """Plots a 1D slice of the GP for each hyperparameter and saves them separately."""
        if not hasattr(self, "gp_model"):
            print("Run recommend_next_configs first to train the GP.")
            return

        # Find the best observed point to keep other dimensions fixed
        best_idx = self.train_Y.argmax().item()
        best_x_norm = self.train_X[best_idx]

        param_names = ["LoRA r", "Learning Rate (log10)", "Dropout"]
        safe_file_names = ["lora_r", "learning_rate", "dropout"]

        for i in range(3):
            # Create a NEW figure for each hyperparameter (8x6 inches, high-res)
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

            # --- Visual Improvements: Grid & Spines ---
            ax.grid(True, linestyle="--", alpha=0.5, color="#CBD5E1")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#64748B")
            ax.spines["left"].set_color("#64748B")

            # Create a 1D test grid in the [0, 1] normalized space
            test_x_norm = best_x_norm.repeat(100, 1)
            grid_norm = torch.linspace(0, 1, 100, dtype=torch.float64)
            test_x_norm[:, i] = grid_norm

            # Predict using the GP
            with torch.no_grad():
                posterior = self.gp_model.posterior(test_x_norm)
                mean_norm = posterior.mean.detach().numpy().flatten()
                variance = posterior.variance.detach().numpy().flatten()

            std = np.sqrt(variance)

            # 1. Un-normalize the X-axis for plotting
            min_val = self.bounds_real[0, i].item()
            max_val = self.bounds_real[1, i].item()
            grid_real = grid_norm.numpy() * (max_val - min_val) + min_val
            obs_x_real = self.train_X[:, i].numpy() * (max_val - min_val) + min_val

            # 2. Un-negate the Y-axis (Loss)
            mean_loss = -mean_norm
            obs_y_loss = -self.train_Y.numpy().flatten()

            # Confidence interval bounds
            lower_loss_bound = mean_loss - 1.96 * std
            upper_loss_bound = mean_loss + 1.96 * std

            # --- Plotting with upgraded color palette ---

            # 1. Confidence Interval
            ax.fill_between(
                grid_real,
                lower_loss_bound,
                upper_loss_bound,
                color="#DBEAFE",  # Soft modern blue
                alpha=0.8,
                label="95% Confidence Interval",
            )

            # 2. GP Mean
            ax.plot(grid_real, mean_loss, color="#2563EB", lw=3.0, label="GP Mean")

            # 3. Observed Data
            ax.scatter(
                obs_x_real,
                obs_y_loss,
                color="#1E293B",  # Slate dark
                edgecolor="white",  # Adds separation from the background
                linewidth=1.2,
                s=80,
                zorder=5,
                label="Observed Values",
            )

            # 4. Suggestions
            if hasattr(self, "candidates_norm"):
                sugg_x_norm = self.candidates_norm[:, i]
                sugg_x_real = sugg_x_norm.numpy() * (max_val - min_val) + min_val

                test_sugg_norm = best_x_norm.repeat(self.candidates_norm.shape[0], 1)
                test_sugg_norm[:, i] = sugg_x_norm

                with torch.no_grad():
                    sugg_post = self.gp_model.posterior(test_sugg_norm)
                    sugg_mean_loss = -sugg_post.mean.detach().numpy().flatten()

                ax.scatter(
                    sugg_x_real,
                    sugg_mean_loss,
                    color="#DC2626",  # Crimson red
                    edgecolor="white",
                    linewidth=1.0,
                    marker="*",
                    s=350,  # Slightly larger for emphasis
                    zorder=10,
                    label="New Batch Suggestions",
                )

            # --- Labeling & Formatting ---
            ax.set_title(
                f"Bayesian Surrogate Landscape: {param_names[i]}",
                fontsize=14,
                fontweight="bold",
                color="#0F172A",
                pad=15,
            )
            ax.set_xlabel(
                param_names[i], fontsize=12, fontweight="medium", color="#334155"
            )
            ax.set_ylabel(
                "Validation Loss", fontsize=12, fontweight="medium", color="#334155"
            )

            # Style the ticks
            ax.tick_params(axis="both", colors="#475569", labelsize=10)

            # Refine Legend
            ax.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="#CBD5E1",
                fontsize=10,
            )

            # Save individual plot
            filename = f"{file_prefix}_{safe_file_names[i]}.png"
            plt.tight_layout()
            plt.savefig(filename, bbox_inches="tight")
            print(f"✅ Saved polished GP plot to '{filename}'")
            plt.close()


# --- Test the Native BoTorch Engine ---
if __name__ == "__main__":
    # Assuming you have generated history_runs.csv from the previous step
    try:
        mock_history = pd.read_csv("history_runs.csv")
    except FileNotFoundError:
        print("Please ensure 'history_runs.csv' exists in the directory.")
        exit()

    bo_engine = PureBotorchOptimizer(mock_history)
    new_dataset = {"ttr_score": 0.40, "hapax_ratio": 0.50, "zipf_alpha": 1.08}

    print("🧠 Native BoTorch Generating Suggestions...")
    candidates = bo_engine.recommend_next_configs(new_dataset, n_suggestions=3)

    for i, config in enumerate(candidates):
        print(f"Node {i+1}: {config}")

    # Generate the visual plots!
    bo_engine.plot_1d_surrogates(file_prefix="gp_surrogate")
